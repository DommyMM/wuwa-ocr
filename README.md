# Wuthering Waves OCR Backend

FastAPI OCR service for WuWaBuilds import scans.  
Hosted at `https://ocr.wuwabuilds.moe`.

## Runtime Model

- Single OCR mode: split-card region processing (`card.py`)
- Legacy full-screen mode (`char.py` / `echo.py`) has been removed
- Data is loaded from local `backend/Data/*.json` and template PNGs at startup import time (`data.py`)
- Echo, character, and weapon OCR results include IDs for robust frontend matching

## Start

```bash
py server.py
```

Default port is `5000` (`PORT` env var supported).

## API

### `POST /api/ocr`

Process one cropped region image.

Request body:

```json
{
  "image": "base64_encoded_image"
}
```

Recommended header:

```http
X-OCR-Region: character | weapon | watermark | forte | sequences | echo1 | echo2 | echo3 | echo4 | echo5
```

Compatibility fallbacks (optional body fields):

- `region`: direct region key
- `type`: legacy `import-<region>` format

`char-*` legacy mode is not supported.

### Example Responses

Character region:

```json
{
  "success": true,
  "analysis": {
    "name": "Aemeath",
    "id": "53",
    "level": 90
  }
}
```

Weapon region:

```json
{
  "success": true,
  "analysis": {
    "name": "Everbright Polestar",
    "id": "21020076",
    "level": 90
  }
}
```

Echo region (`echo1`-`echo5`):

```json
{
  "success": true,
  "analysis": {
    "name": {
      "name": "Sigillum",
      "id": "60001915",
      "confidence": 0.87
    },
    "main": { "name": "Crit DMG", "value": "44%" },
    "substats": [{ "name": "Crit Rate", "value": "8.7%" }],
    "element": "Trailblazing"
  }
}
```

### Other Endpoints

- `GET /health` -> health check
- `GET /` -> API status metadata
- `GET /ocr-results` -> serves `../ocr_results.json` (output of `batch_ocr.py`); 404 when missing

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | HTTP listen port |
| `INTERNAL_API_KEY` | — | Trusted proxy key (Railway → OCR); skips per-IP rate limiting and uses forwarded client IP |
| `OCR_WORKERS` | `8` | `ProcessPoolExecutor` size — parallel Tesseract processes. Match to CPU thread count locally (e.g. `16` for 7800X3D). Railway is capped at `8` vCPU. |
| `OCR_RATE_LIMIT` | `60` | Requests per minute per IP. Set to `10000` locally to disable effective limiting during batch import. |
| `OCR_TIMEOUT` | `60` | Seconds before a single OCR request times out. |
| `OCR_OPENCV_THREADS` | `1` | Per-worker `cv2.setNumThreads` value. |
| `USE_GPU` | `1` locally, `0` on Railway | Enables RapidOCR CUDA providers in `data.py` when `onnxruntime-gpu` is available. |
| `RAILWAY_ENVIRONMENT_NAME` | — | Auto-set on Railway; toggles GPU default and is used to log environment context. |

## Limits and Errors

- Rate limit: `OCR_RATE_LIMIT` requests/minute per IP (default `60`)
- Timeout: `OCR_TIMEOUT` per OCR request (default `60s`)
- Common statuses:
  - `400` invalid image/region/request
  - `408` processing timeout
  - `429` rate limit exceeded
  - `500` internal server error

## Data Sync Expectations

The backend does not fetch runtime game data from production frontend URLs.  
Keep `backend/Data` synchronized from `wuwabuilds/scripts`:

1. `py sync_all.py`
2. `py download_echo_icons.py --clean --force` (when templates need refresh)

