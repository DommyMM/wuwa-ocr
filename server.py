from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import hashlib
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional, cast
from card import process_card
import time
from collections import defaultdict
import os
import asyncio
from contextlib import asynccontextmanager
import ipaddress
import sys
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on env vars being set externally

IS_RAILWAY = bool(os.getenv("RAILWAY_ENVIRONMENT_NAME"))
USE_GPU = os.getenv("USE_GPU", "0" if IS_RAILWAY else "1") == "1"

MAX_WORKERS = int(os.getenv("OCR_WORKERS", "8"))
OPENCV_THREADS = int(os.getenv("OCR_OPENCV_THREADS", "1"))
PROCESS_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "60"))
REQUESTS_PER_MINUTE = int(os.getenv("OCR_RATE_LIMIT", "60"))
PORT = int(os.getenv("PORT", "5000"))
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "").strip()
WARM_WORKERS = os.getenv("OCR_WARM_WORKERS", "1") == "1"
consecutive_500s = 0
MAX_CONSECUTIVE_500S = 3

# Ensure output is flushed for Railway
if hasattr(sys.stdout, "reconfigure"):
    cast(Any, sys.stdout).reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    cast(Any, sys.stderr).reconfigure(line_buffering=True)

cv2.setNumThreads(OPENCV_THREADS)

def force_restart(reason: str):
    print(f"FORCING RESTART: {reason}", flush=True)
    time.sleep(1)  # Give time for log to be written
    os._exit(1)  # Hard exit that Railway will detect
    
class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        self.requests[ip] = [req_time for req_time in self.requests[ip] if req_time > minute_ago]
        if len(self.requests[ip]) < REQUESTS_PER_MINUTE:
            self.requests[ip].append(now)
            return True
        return False

def normalize_ip(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    candidate = value.split(",", 1)[0].strip()
    if not candidate:
        return None

    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None

def is_trusted_proxy_request(request: Request) -> bool:
    if not INTERNAL_API_KEY:
        return False

    return request.headers.get("x-internal-key", "").strip() == INTERNAL_API_KEY

def get_rate_limit_identity(request: Request) -> str:
    if is_trusted_proxy_request(request):
        forwarded_ip = normalize_ip(request.headers.get("x-ocr-client-ip"))
        if forwarded_ip:
            return forwarded_ip

    direct_ip = normalize_ip(request.client.host if request.client else None)
    if direct_ip:
        return direct_ip

    return request.client.host if request.client and request.client.host else "unknown"

class OCRResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    analysis: Optional[dict] = None
    progress: Optional[dict] = None
    timings: Optional[dict] = None
    trainingImageKey: Optional[str] = None

class APIStatus(BaseModel):
    status: str = "running"
    endpoints: dict = {
        "ocr": {
            "path": "/api/ocr",
            "method": "POST",
            "request": {
                "image": "multipart file field or raw image body"
            },
            "response": "full import analysis with progress and timings",
        }
    }

IMPORT_REGIONS: dict[str, dict[str, float]] = {
    # Use asset-bearing crops for character/weapon now that those are SIFT
    # recognizers. The old title-strip/name OCR crop is intentionally skipped.
    "character": {"x1": 0.0200, "x2": 0.2700, "y1": 0.1000, "y2": 0.4500},
    "watermark": {"x1": 0.0073, "x2": 0.1304, "y1": 0.0741, "y2": 0.1370},
    "forte": {"x1": 0.4057, "x2": 0.7422, "y1": 0.0222, "y2": 0.5917},
    "sequences": {"x1": 0.0703, "x2": 0.3318, "y1": 0.4787, "y2": 0.5843},
    "weapon": {"x1": 0.7590, "x2": 0.8310, "y1": 0.4120, "y2": 0.5380},
    "echo1": {"x1": 0.0125, "x2": 0.2042, "y1": 0.6019, "y2": 0.9843},
    "echo2": {"x1": 0.2057, "x2": 0.3974, "y1": 0.6019, "y2": 0.9843},
    "echo3": {"x1": 0.4016, "x2": 0.5938, "y1": 0.6019, "y2": 0.9843},
    "echo4": {"x1": 0.5969, "x2": 0.7891, "y1": 0.6019, "y2": 0.9843},
    "echo5": {"x1": 0.7911, "x2": 0.9833, "y1": 0.6019, "y2": 0.9843},
}

REGION_KEYS = tuple(IMPORT_REGIONS.keys())

def warm_worker() -> int:
    import card
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    card.recognize_character_asset(blank)
    card.recognize_weapon_asset(blank)
    return os.getpid()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Server starting on port {PORT} | railway={IS_RAILWAY} gpu={USE_GPU} workers={MAX_WORKERS} opencv_threads={OPENCV_THREADS}", flush=True)
    if WARM_WORKERS:
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        await asyncio.gather(*[
            loop.run_in_executor(executor, warm_worker)
            for _ in range(MAX_WORKERS)
        ])
        print(f"Warmed {MAX_WORKERS} OCR workers in {(time.perf_counter() - started):.2f}s", flush=True)
    yield
    print("Server shutting down", flush=True)
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)
def worker_init():
    """Ensure worker output is flushed."""
    if hasattr(sys.stdout, "reconfigure"):
        cast(Any, sys.stdout).reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        cast(Any, sys.stderr).reconfigure(line_buffering=True)
    cv2.setNumThreads(OPENCV_THREADS)

def crop_region(image: np.ndarray, region: dict[str, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = round(region["x1"] * w)
    x2 = round(region["x2"] * w)
    y1 = round(region["y1"] * h)
    y2 = round(region["y2"] * h)
    return np.ascontiguousarray(image[y1:y2, x1:x2])

def process_region_task(task: tuple[str, np.ndarray]) -> dict[str, Any]:
    region, crop = task
    started = time.perf_counter()
    try:
        result = process_card(crop, region)
        return {
            "region": region,
            "success": bool(result.get("success")),
            "analysis": result.get("analysis"),
            "error": result.get("error"),
            "elapsedMs": (time.perf_counter() - started) * 1000,
        }
    except Exception as exc:
        return {
            "region": region,
            "success": False,
            "analysis": None,
            "error": str(exc),
            "elapsedMs": (time.perf_counter() - started) * 1000,
        }

executor = ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=worker_init
)
rate_limiter = RateLimiter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/api/ocr":
        rate_limit_identity = get_rate_limit_identity(request)
        if not rate_limiter.is_allowed(rate_limit_identity):
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "Rate limit exceeded. Please try again later.",
                }
            )
    response = await call_next(request)
    return response

async def read_upload_image_bytes(request: Request) -> bytes:
    content_type = request.headers.get("content-type", "").lower()

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        value = form.get("image")
        if not isinstance(value, UploadFile) and not hasattr(value, "read"):
            raise HTTPException(status_code=400, detail="Missing multipart file field 'image'.")
        image_bytes = await value.read()
    else:
        image_bytes = await request.body()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Missing image bytes.")

    return image_bytes

async def process_full_import_image(image_bytes: bytes) -> dict[str, Any]:
    timing_start = time.perf_counter()

    hash_started = time.perf_counter()
    image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    hashed_at = time.perf_counter()

    decode_started = time.perf_counter()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    decoded_at = time.perf_counter()

    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image.")

    crop_started = time.perf_counter()
    crops = {
        region: crop_region(image, coords)
        for region, coords in IMPORT_REGIONS.items()
    }
    cropped_at = time.perf_counter()

    loop = asyncio.get_event_loop()
    recognition_started = time.perf_counter()
    tasks = [
        loop.run_in_executor(executor, process_region_task, (region, crops[region]))
        for region in REGION_KEYS
    ]

    try:
        region_results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=PROCESS_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail=f"Processing timeout exceeded ({PROCESS_TIMEOUT} seconds)")
    except Exception as exc:
        error_msg = str(exc)
        if "terminated abruptly" in error_msg.lower():
            force_restart(f"ProcessPool worker terminated abruptly: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Image processing error: {error_msg}")

    recognized_at = time.perf_counter()

    analysis: dict[str, Any] = {}
    progress: dict[str, str] = {}
    region_timings: dict[str, float] = {}
    region_errors: dict[str, str] = {}

    for result in region_results:
        region = result["region"]
        region_timings[region] = round(float(result["elapsedMs"]), 2)
        if result["success"] and result["analysis"] is not None:
            analysis[region] = result["analysis"]
            progress[region] = "done"
        else:
            progress[region] = "error"
            if result.get("error"):
                region_errors[region] = str(result["error"])

    finished_at = time.perf_counter()
    timings = {
        "hashMs": round((hashed_at - hash_started) * 1000, 2),
        "decodeMs": round((decoded_at - decode_started) * 1000, 2),
        "cropMs": round((cropped_at - crop_started) * 1000, 2),
        "recognitionWallMs": round((recognized_at - recognition_started) * 1000, 2),
        "totalMs": round((finished_at - timing_start) * 1000, 2),
        "regions": region_timings,
    }

    return {
        "success": True,
        "analysis": analysis,
        "progress": progress,
        "timings": timings,
        "trainingImageKey": f"training-images/{image_hash}.jpg",
        "regionErrors": region_errors,
        "image": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "bytes": len(image_bytes),
        },
    }

@app.post("/api/ocr", response_model=OCRResponse)
async def process_image_request(request: Request):
    global consecutive_500s

    request_start = time.perf_counter()
        
    try:
        print("import: Processing full-image request", flush=True)

        body_read_started = time.perf_counter()
        image_bytes = await read_upload_image_bytes(request)
        body_read_ms = (time.perf_counter() - body_read_started) * 1000
        result = await process_full_import_image(image_bytes)
        if result.get("timings"):
            result["timings"]["bodyReadMs"] = round(body_read_ms, 2)
            result["timings"]["wallMs"] = round((time.perf_counter() - request_start) * 1000, 2)
            
        timings = result.get("timings", {})
        region_timings = timings.get("regions") if isinstance(timings, dict) else None
        if isinstance(region_timings, dict):
            slow_regions = ",".join(
                f"{name}:{elapsed:.0f}"
                for name, elapsed in sorted(region_timings.items(), key=lambda item: item[1], reverse=True)[:4]
            )
        else:
            slow_regions = ""
        print(
            "import: Completed "
            f"wall={timings.get('wallMs')}ms "
            f"body={timings.get('bodyReadMs')}ms "
            f"decode={timings.get('decodeMs')}ms "
            f"crop={timings.get('cropMs')}ms "
            f"recognition={timings.get('recognitionWallMs')}ms "
            f"bytes={result.get('image', {}).get('bytes')} "
            f"slow={slow_regions}",
            flush=True,
        )
        
        consecutive_500s = 0
        return result
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "success": False,
                "error": e.detail
            }
        )
        
    except Exception as e:
        print(f"import: Failed - {str(e)}", flush=True)
        
        consecutive_500s += 1
        if consecutive_500s > 1:
            print(f"Consecutive errors: {consecutive_500s}/{MAX_CONSECUTIVE_500S}", flush=True)
        
        if consecutive_500s >= MAX_CONSECUTIVE_500S:
            force_restart(f"Too many consecutive 500 errors ({consecutive_500s})")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.get("/", response_model=APIStatus)
async def homepage():
    return APIStatus()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ocr-results")
async def ocr_results():
    """Serve the batch OCR results JSON for frontend bulk submission."""
    import json as _json
    results_path = Path(__file__).parent.parent / "ocr_results.json"
    if not results_path.exists():
        return JSONResponse(status_code=404, content={"error": "ocr_results.json not found — run batch_ocr.py first"})
    with open(results_path, encoding="utf-8") as f:
        return _json.load(f)

if __name__ == "__main__":
    import uvicorn
    print(f"Uvicorn starting on 0.0.0.0:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
