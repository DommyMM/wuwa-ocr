# Wuthering Waves OCR Backend

FastAPI OCR service for WuWaBuilds import scans.  
Hosted at `https://ocr.wuwa.build`.

## Runtime Model

- Single OCR mode: English full-card import processing. The API receives the original
  screenshot, decodes it once, crops fixed regions server-side, then fans out
  region recognition through `card.py`.
- Legacy full-screen mode (`char.py` / `echo.py`) has been removed
- Data is loaded from local `backend/Data/*.json` and image templates at startup import time (`data.py`)
- Echo, character, and weapon OCR results include IDs for robust frontend matching
- Echo and element templates may be PNG or WebP; current element templates are WebP-only.

## Start

```bash
py server.py
```

Default port is `5000` (`PORT` env var supported).

## API

### `POST /api/ocr`

Process one full build-card screenshot.

Request body can be `multipart/form-data`:

```http
image=<file>
```

or a raw image body with an image `Content-Type`.

The response is always an `application/x-ndjson` stream. Each line is one JSON
event: `meta`, zero or more per-region `region` events, then a final `done`
event that contains the merged import analysis, per-region status, and timing
data.

```json
{"type":"meta","trainingImageKey":"training-images/00055f05eb843ecf.jpg","image":{"width":1920,"height":1080,"bytes":271977}}
{"type":"region","region":"watermark","status":"done","analysis":{"username":"Player","uid":500000000},"elapsedMs":180.2}
{"type":"region","region":"echo1","status":"done","analysis":{"main":{"name":"ATK%","value":"18%"}},"elapsedMs":1111.0}
{"type":"done","success":true,"analysis":{},"progress":{},"timings":{},"trainingImageKey":"training-images/00055f05eb843ecf.jpg"}
```

Interactive import consumes the `region` events for live UI updates. Bulk import
uses the same stream parser and only consumes the final `done` event.

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
| `OMP_THREAD_LIMIT` | `1` in Dockerfile | Keeps each Tesseract subprocess single-threaded while the service parallelizes across regions/workers. |
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

## Railway Operations

Production runs in Railway project `wuwa-backend` as service `WuWa OCR`.
Operational commands, latency/cost snapshots, and log
query examples live in [`docs/railway-observability.md`](docs/railway-observability.md).

## Data Sync Expectations

The backend does not fetch runtime game data from production frontend URLs.  
Keep `backend/Data` synchronized from `wuwabuilds/scripts`:

1. From `wuwabuilds/scripts`, run the data sync (`py sync_all.py`, or the targeted Encore merge path documented in `wuwabuilds/docs/sync-sources.md`).
2. Run `py sync_backend.py` to refresh backend JSONs and Encore element badge templates.
3. Run `py download_echo_icons.py --clean --force` only when echo icon templates need a full refresh. The downloader stores backend echo templates as WebP by default, converting PNG source assets when needed.
4. To validate a full echo-template WebP swap without committing converted assets, run `py backend\regress_echo_webp.py --limit 500` from the workspace root after installing `backend/requirements.txt`.

## Local Backfill Helpers

- `py backend\r2_date_summary.py --since 2026-06-07T19:00:00-07:00` counts local R2 screenshots in a patch window.
- `py backend\stage_r2_backfill.py --since 2026-06-07T19:00:00-07:00 --clean` stages a filtered folder for the frontend `/bulk-import` page.
