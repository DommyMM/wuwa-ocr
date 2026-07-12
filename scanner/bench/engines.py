"""Uniform wrappers around every local OCR engine we're evaluating.

Each engine exposes the same contract:

    Engine.name    -> str
    Engine.load()  -> None            (cold-start cost lives here)
    Engine.read(img_bgr) -> list[str] (text lines, top-to-bottom)

An engine that isn't installed / can't initialise reports itself unavailable
with a reason instead of raising, so the bench runs on whatever is present.

Deliberately excluded: Surya and the VLM-class readers (GOT-OCR, dots.ocr,
olmOCR). They are 650M-7B params to read a 6-digit number, and they'd contend
with the game for the GPU. Wrong tool.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


class Engine:
    name = "base"
    langs = "?"
    note = ""

    def load(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def read(self, img: np.ndarray) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def read_batch(self, imgs: list[np.ndarray]) -> list[list[str]]:
        """Read all of one echo's value cells. This is the real per-echo unit.

        Default: one call per cell. Tesseract overrides it, because its per-call
        cost is ~all process spawn and it can take N images in one invocation
        while still returning N separate results (so row alignment is preserved -
        we never batch cells into a single image, which is what lets an engine
        drop a line and shift every row below it).
        """
        return [self.read(i) for i in imgs]

    def available(self) -> tuple[bool, str]:
        try:
            self.load()
            return True, ""
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"


# --- Tesseract ---------------------------------------------------------------

class Tesseract(Engine):
    """psm 6 = uniform block of text. The values strip needs it (psm 3 drops
    short integers like 470 / 2280); see docs/echo-substat-tesseract-only.md.
    """
    langs = "100+ (all 9 WuWa langs); fine-tunable on the shipped game fonts"

    def __init__(self, psm: int = 6, whitelist: str | None = None, lang: str = "eng"):
        self.psm = psm
        self.whitelist = whitelist
        self.lang = lang
        self.name = f"tesseract:psm{psm}" + (":digits" if whitelist else "")

    def load(self) -> None:
        import pytesseract
        self._t = pytesseract
        self._t.get_tesseract_version()

    def _config(self) -> str:
        cfg = f"--psm {self.psm}"
        if self.whitelist:
            cfg += f" -c tessedit_char_whitelist={self.whitelist}"
        return cfg

    def read(self, img: np.ndarray) -> list[str]:
        txt = self._t.image_to_string(img, lang=self.lang, config=self._config())
        return [l.strip() for l in txt.splitlines() if l.strip()]

    def read_batch(self, imgs: list[np.ndarray]) -> list[list[str]]:
        """One tesseract process for all N cells, N separate results.

        pytesseract spawns a process per call and reloads eng.traineddata each
        time; that spawn is ~all of the measured 184 ms, not recognition.
        Tesseract's CLI accepts a file LIST and emits one form-feed-separated page
        per image, so we pay spawn once and keep per-cell alignment.

        Inventory Kamera solves the same problem by pooling 8 warm in-process
        Tesseract engines via the C API. tesserocr would give us that directly,
        but it has no Python 3.13 wheel, so this is the dependency-free equivalent.
        """
        import subprocess
        import tempfile

        import cv2

        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, im in enumerate(imgs):
                p = f"{td}/cell_{i:02d}.png"
                cv2.imwrite(p, im)
                paths.append(p)
            listing = f"{td}/cells.txt"
            with open(listing, "w") as fh:
                fh.write("\n".join(paths))

            cmd = ["tesseract", listing, "stdout", "-l", self.lang,
                   "--psm", str(self.psm)]
            if self.whitelist:
                cmd += ["-c", f"tessedit_char_whitelist={self.whitelist}"]
            out = subprocess.run(cmd, capture_output=True, text=True).stdout

        pages = out.split("\f")
        results = [[l.strip() for l in p.splitlines() if l.strip()] for p in pages]
        results = [r for r in results if True][: len(imgs)]
        while len(results) < len(imgs):
            results.append([])
        return results


# --- RapidOCR (PP-OCR models on ONNX) ----------------------------------------

class RapidOCRv1(Engine):
    """rapidocr-onnxruntime 1.x - what backend/data.py pins today.

    rec_only: the icon already localised the cell, so running the DETECTION model
    on a 285x68 crop is pure waste - and it actively hurts, because the detector
    fails to find a box in a tiny crop and returns nothing.
    """
    langs = "en/ch (per-model); PP-OCRv4 era"

    def __init__(self, rec_only: bool = True):
        self.rec_only = rec_only
        self.name = "rapidocr-onnx:1.x" + (":rec" if rec_only else ":det+rec")

    def load(self) -> None:
        from rapidocr_onnxruntime import RapidOCR
        if not hasattr(self, "_r"):
            self._r = RapidOCR(lang="en")

    def read(self, img: np.ndarray) -> list[str]:
        kw = dict(use_det=False, use_cls=False, use_rec=True) if self.rec_only else {}
        res, _ = self._r(img, **kw)
        return [t.strip() for _, t, _ in res] if res else []


class RapidOCRv3(Engine):
    """rapidocr 3.x - current line, runs PP-OCRv5/v6 incl. Thai."""
    langs = "~100 incl. th/ja/ko/ch_tra (PP-OCRv5/v6)"

    def __init__(self, rec_only: bool = True):
        self.rec_only = rec_only
        self.name = "rapidocr:3.x" + (":rec" if rec_only else ":det+rec")

    def load(self) -> None:
        from rapidocr import RapidOCR
        if not hasattr(self, "_r"):
            self._r = RapidOCR()

    def read(self, img: np.ndarray) -> list[str]:
        kw = dict(use_det=False, use_cls=False, use_rec=True) if self.rec_only else {}
        res = self._r(img, **kw)
        txts = getattr(res, "txts", None)
        return [t.strip() for t in txts] if txts else []


# --- PaddleOCR (same models, heavier runtime) --------------------------------

class Paddle(Engine):
    name = "paddleocr"
    langs = "106 incl. th"
    note = "same PP-OCR models as RapidOCR; ~1GB paddlepaddle runtime"

    def load(self) -> None:
        from paddleocr import PaddleOCR
        if not hasattr(self, "_r"):
            self._r = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

    def read(self, img: np.ndarray) -> list[str]:
        res = self._r.predict(img)
        out: list[str] = []
        for page in res or []:
            out.extend(t.strip() for t in page.get("rec_texts", []))
        return out


# --- EasyOCR -----------------------------------------------------------------

class Easy(Engine):
    langs = "80+ incl. th/ja/ko/ch_tra"
    note = "PyTorch; ~2GB+ in an exe"

    def __init__(self, rec_only: bool = True, gpu: bool = False):
        self.rec_only = rec_only
        self.gpu = gpu
        self.name = "easyocr" + (":rec" if rec_only else ":det+rec") + (":gpu" if gpu else "")

    def load(self) -> None:
        import easyocr
        if not hasattr(self, "_r"):
            self._r = easyocr.Reader(["en"], gpu=self.gpu, verbose=False)

    def read(self, img: np.ndarray) -> list[str]:
        if not self.rec_only:
            return [t.strip() for _, t, _ in self._r.readtext(img)]
        import cv2
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # horizontal_list=None + free_list=None => treat the whole crop as one box
        return [t.strip() for _, t, _ in self._r.recognize(grey, None, None)]


# --- OneOCR (Windows Snipping Tool engine) -----------------------------------

class OneOCR(Engine):
    name = "oneocr"
    langs = "CJK-strong; exact list undocumented"
    note = "needs oneocr.dll + .onemodel lifted from Windows ScreenSketch"

    def load(self) -> None:
        import oneocr
        if not hasattr(self, "_r"):
            self._r = oneocr.OcrEngine()

    def read(self, img: np.ndarray) -> list[str]:
        res = self._r.recognize_cv2(img)
        return [l["text"].strip() for l in res.get("lines", [])]


# --- WinRT OCR (built into Windows) ------------------------------------------

class WinRT(Engine):
    name = "winrt"
    langs = "depends on installed Windows language packs (~25)"
    note = "zero bundle size"

    def load(self) -> None:
        import winocr
        import cv2
        from PIL import Image
        self._winocr = winocr
        self._cv2 = cv2
        self._Image = Image

    def read(self, img: np.ndarray) -> list[str]:
        rgb = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)
        res = self._winocr.recognize_pil_sync(self._Image.fromarray(rgb), "en")
        lines = res["lines"] if isinstance(res, dict) else res.lines
        out = []
        for l in lines:
            t = l["text"] if isinstance(l, dict) else l.text
            out.append(t.strip())
        return out


ALL_ENGINES: list[Engine] = [
    Tesseract(psm=7, whitelist="0123456789.%"),
    Tesseract(psm=6, whitelist="0123456789.%"),
    RapidOCRv1(rec_only=True),
    RapidOCRv3(rec_only=True),
    Paddle(),
    Easy(rec_only=True),
    Easy(rec_only=True, gpu=True),
    OneOCR(),
    WinRT(),
]


@dataclass
class Timing:
    cold_ms: float = 0.0
    warm_ms: list[float] = field(default_factory=list)

    @property
    def warm_median(self) -> float:
        if not self.warm_ms:
            return 0.0
        s = sorted(self.warm_ms)
        return s[len(s) // 2]


def time_load(engine: Engine) -> float:
    t0 = time.perf_counter()
    engine.load()
    return (time.perf_counter() - t0) * 1000


def time_read(engine: Engine, img: np.ndarray, runs: int = 5) -> tuple[list[str], Timing]:
    t = Timing()
    lines = engine.read(img)  # warm the path once, discard
    for _ in range(runs):
        t0 = time.perf_counter()
        lines = engine.read(img)
        t.warm_ms.append((time.perf_counter() - t0) * 1000)
    return lines, t
