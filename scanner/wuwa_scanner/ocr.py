"""The scanner's only OCR, which reads numbers

Stat names, '%', identity, cost and set come from icons and templates, so OCR covers substat values and tile levels
Every engine read 5/5 on correct crops, so WinRT leads for speed (24 ms/echo, no bundle) with Tesseract as fallback
Tesseract takes all cells in one process via a file list, since each pytesseract call spends ~154 ms on spawn
Engines run recognition-only since cells are already localised, and RapidOCR with detection read 2/7
Never concatenate cells into one image, since the engine can drop a line and shift every row below it
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

NUM_RX = re.compile(r"\d+(?:[.,]\d+)?")


def _prep(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """Upscale, grayscale and Otsu

    Value cells read fine at 2x but level-digit crops need 4x (89/90 at 2x), a plateau where every psm agrees
    """
    up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return th


def parse_number(text: str) -> float | None:
    """Digits only, since the stat family already implies the '%'"""
    m = NUM_RX.search(text.replace(" ", "").replace("%", ""))
    if not m:
        return None
    try:
        return float(m.group().replace(",", "."))
    except ValueError:
        return None


class Reader:
    """Reads numbers from a batch of cells, always one result per cell"""

    def __init__(self, upscale: int = 2) -> None:
        self.upscale = upscale

    def read(self, cells: list[np.ndarray | None]) -> list[float | None]:
        raise NotImplementedError


class WinRTReader(Reader):
    """Windows.Media.Ocr, built into Windows with zero bundle size at ~5 ms/cell"""

    name = "winrt"

    def __init__(self, upscale: int = 2) -> None:
        super().__init__(upscale)
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
            rgb = cv2.cvtColor(_prep(c, self.upscale), cv2.COLOR_GRAY2RGB)
            res = self._winocr.recognize_pil_sync(self._Image.fromarray(rgb), "en")
            lines = res["lines"] if isinstance(res, dict) else res.lines
            txt = "".join((l["text"] if isinstance(l, dict) else l.text) for l in lines)
            out.append(parse_number(txt))
        return out


class TesseractReader(Reader):
    """Tesseract reading N cells in one process via a file list, returning N separate results

    Warm in-process engines would skip the spawn, but tesserocr has no Python 3.13 wheel
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
                cv2.imwrite(p, _prep(cells[i], self.upscale))
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
    """Substat value cells: WinRT if available (24 ms/echo, zero bundle), else Tesseract"""
    try:
        return WinRTReader()
    except Exception:
        return TesseractReader()


def level_reader() -> Reader:
    """Tile level pills, always Tesseract at 4x

    WinRT returns no lines on the one- or two-digit pill at 2x or 4x (0/18) while Tesseract reads 18/18
    default_reader here would silently return no levels, which reads as nothing worth clicking
    """
    return TesseractReader(upscale=4)
