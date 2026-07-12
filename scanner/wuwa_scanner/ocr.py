"""The only OCR in the scanner: read a number.

Everything else is icon/template matching, so this is all that is language-dependent -
and digits are identical in all 9 WuWa text languages, so in practice it is not.

Scope, after the design collapsed the problem:
  * stat NAMES come from the 17 stat icons                -> no OCR
  * the '%' is implied by the stat family                 -> no OCR
  * main + innate values derive from cost (EchoStats.json)-> no OCR
  * echo identity/cost/set/level come from the tile       -> no OCR
  => 5 substat numbers per echo. That is the entire OCR surface.

Engine bench (5 substat cells/echo, on real 4K captures). Once the crops were correct,
EVERY engine scored 5/5, so accuracy is not a differentiator and the choice is purely
speed and packaging:

    WinRT (Windows.Media.Ocr)      24 ms/echo   zero bundle      <- primary
    EasyOCR rec-only (GPU)        154 ms/echo   ~2 GB torch
    Tesseract (1 spawn, N images) 213 ms/echo   ~10 MB           <- fallback
    RapidOCR 3.x rec-only         399 ms/echo   ~50 MB
    PaddleOCR                     crashes       ~1 GB

Two traps worth remembering:
  * `pytesseract` costs ~154 ms PER CALL, and that is process spawn reloading
    eng.traineddata, not recognition. Handing Tesseract all N cells in ONE invocation
    via a file list is 5-6x faster and still returns N separate results.
  * Engines must run RECOGNITION-ONLY. The cell is already localised, so letting
    RapidOCR/Paddle run text DETECTION on a 285x68 crop is waste, and it actively fails
    (RapidOCR went from 2/7 to 7/7 the moment detection was disabled).

NEVER concatenate the cells into one image to save a call. That lets the engine drop a
line and shift every row below it, which is exactly the drift that
card.py::reconcile_echo_substat_rows exists to survive. One cell in, one result out.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

NUM_RX = re.compile(r"\d+(?:[.,]\d+)?")


def _prep(img: np.ndarray) -> np.ndarray:
    up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return th


def parse_number(text: str) -> float | None:
    """Digits only. The '%' is never read - the stat family already implies it."""
    m = NUM_RX.search(text.replace(" ", "").replace("%", ""))
    if not m:
        return None
    try:
        return float(m.group().replace(",", "."))
    except ValueError:
        return None


class Reader:
    """Reads numbers from a batch of value cells. One result per cell, always."""

    def read(self, cells: list[np.ndarray | None]) -> list[float | None]:
        raise NotImplementedError


class WinRTReader(Reader):
    """Windows.Media.Ocr. Built into Windows, zero bundle size, ~5 ms/cell.

    Its usual weakness (weak CJK without language packs) is irrelevant here: we only
    ever read digits.
    """

    name = "winrt"

    def __init__(self) -> None:
        import winocr
        from PIL import Image
        self._winocr = winocr
        self._Image = Image

    def read(self, cells: list[np.ndarray | None]) -> list[float | None]:
        out: list[float | None] = []
        for c in cells:
            if c is None:
                out.append(None)
                continue
            rgb = cv2.cvtColor(_prep(c), cv2.COLOR_GRAY2RGB)
            res = self._winocr.recognize_pil_sync(self._Image.fromarray(rgb), "en")
            lines = res["lines"] if isinstance(res, dict) else res.lines
            txt = "".join((l["text"] if isinstance(l, dict) else l.text) for l in lines)
            out.append(parse_number(txt))
        return out


class TesseractReader(Reader):
    """Tesseract via ONE process for N cells (file list), N separate results.

    Inventory Kamera solves the same spawn problem by pooling 8 warm in-process engines
    through the C API. `tesserocr` would give us that directly but has no Python 3.13
    wheel, so this is the dependency-free equivalent.
    """

    name = "tesseract"

    def read(self, cells: list[np.ndarray | None]) -> list[float | None]:
        idx = [i for i, c in enumerate(cells) if c is not None]
        if not idx:
            return [None] * len(cells)

        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i in idx:
                p = f"{td}/c{i:02d}.png"
                cv2.imwrite(p, _prep(cells[i]))
                paths.append(p)
            Path(f"{td}/list.txt").write_text("\n".join(paths))
            out = subprocess.run(
                ["tesseract", f"{td}/list.txt", "stdout", "--psm", "7",
                 "-c", "tessedit_char_whitelist=0123456789.%"],
                capture_output=True, text=True,
            ).stdout

        pages = out.split("\f")
        vals: list[float | None] = [None] * len(cells)
        for n, i in enumerate(idx):
            if n < len(pages):
                vals[i] = parse_number(pages[n].replace("\n", ""))
        return vals


def default_reader() -> Reader:
    """WinRT if available (24 ms/echo, zero bundle), else Tesseract (213 ms/echo)."""
    try:
        return WinRTReader()
    except Exception:
        return TesseractReader()
