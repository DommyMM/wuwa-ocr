from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional, cast
from card import process_card
from r2_storage import ImageIdentity, R2ImageStore, R2Settings, StorageResult, UnsupportedImageType, identify_image
from issue_reports import MAX_IMAGE_BYTES, handle_issue_report
from image_integrity import echo_bed_score, validate_header_dimensions, validate_image_integrity
from log_events import log_event
import time
from collections import defaultdict
import os
import asyncio
from contextlib import asynccontextmanager
import ipaddress
import inspect
import hmac
import sys
import uuid
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, so env vars must be set externally

IS_RAILWAY = bool(os.getenv("RAILWAY_ENVIRONMENT_NAME"))

# One worker per heavy region (5 echoes and forte) so those six run in one wave and light regions follow
# Raise if uploads regularly exceed one per second
MAX_WORKERS = int(os.getenv("OCR_WORKERS", "6"))
OPENCV_THREADS = int(os.getenv("OCR_OPENCV_THREADS", "1"))
PROCESS_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "60"))
REQUESTS_PER_MINUTE = int(os.getenv("OCR_RATE_LIMIT", "10"))
REPORTS_PER_MINUTE = int(os.getenv("OCR_REPORT_RATE_LIMIT", "5"))
PORT = int(os.getenv("PORT", "5000"))
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "").strip()
R2_SETTINGS = R2Settings.from_env()
r2_image_store = R2ImageStore.disabled(R2_SETTINGS.timeout_seconds)
active_storage_tasks: set[asyncio.Task[StorageResult]] = set()
consecutive_500s = 0
MAX_CONSECUTIVE_500S = 3

# Non-English thresholds (see card.echo_language_signal): enough real values to be a build card, just not in English
NONENGLISH_NAME_MATCH_FLOOR = float(os.getenv("OCR_NONENGLISH_NAME_FLOOR", "0.35"))
NONENGLISH_MIN_VALUES = int(os.getenv("OCR_NONENGLISH_MIN_VALUES", "15"))

# Echo-bed score worth logging: genuine cards sit ~1.6, labelled pastes start at 3.2
BED_OBSERVE_SCORE_FLOOR = float(os.getenv("OCR_BED_OBSERVE_FLOOR", "2.5"))

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
    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        self.requests[ip] = [req_time for req_time in self.requests[ip] if req_time > minute_ago]
        if len(self.requests[ip]) < self.requests_per_minute:
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

    return hmac.compare_digest(
        request.headers.get("x-internal-key", "").strip(),
        INTERNAL_API_KEY,
    )

def get_rate_limit_identity(request: Request) -> str:
    if is_trusted_proxy_request(request):
        forwarded_ip = normalize_ip(request.headers.get("x-ocr-client-ip"))
        if forwarded_ip:
            return forwarded_ip

    direct_ip = normalize_ip(request.client.host if request.client else None)
    if direct_ip:
        return direct_ip

    return request.client.host if request.client and request.client.host else "unknown"

class APIStatus(BaseModel):
    status: str = "running"
    endpoints: dict = {
        "ocr": {
            "path": "/api/ocr",
            "method": "POST",
            "request": {
                "image": "multipart file field or raw image body",
            },
            "response": "application/x-ndjson stream with meta, region, and done events",
        },
        "reportOcrIssue": {
            "path": "/api/report-ocr-issue",
            "method": "POST",
            "request": "multipart report JSON plus one confirmed image key or fallback image",
            "response": "JSON report receipt",
        },
    }


IMPORT_REGIONS: dict[str, dict[str, float]] = {
    # x2 reaches 0.38 so the LV. pill after the longest names ("Yangyang: Xuanling") stays in crop
    # card.py's CHAR_* sub-boxes are fractions of this width
    "character": {"x1": 0.0000, "x2": 0.3800, "y1": 0.0000, "y2": 0.5500},
    "watermark": {"x1": 0.0073, "x2": 0.1304, "y1": 0.0741, "y2": 0.1370},
    "forte": {"x1": 0.4057, "x2": 0.7422, "y1": 0.0222, "y2": 0.5917},
    "sequences": {"x1": 0.0703, "x2": 0.3318, "y1": 0.4787, "y2": 0.5843},
    "weapon": {"x1": 0.7542, "x2": 0.9828, "y1": 0.3843, "y2": 0.5843},
    "echo1": {"x1": 0.0125, "x2": 0.2042, "y1": 0.6019, "y2": 0.9843},
    "echo2": {"x1": 0.2057, "x2": 0.3974, "y1": 0.6019, "y2": 0.9843},
    "echo3": {"x1": 0.4016, "x2": 0.5938, "y1": 0.6019, "y2": 0.9843},
    "echo4": {"x1": 0.5969, "x2": 0.7891, "y1": 0.6019, "y2": 0.9843},
    "echo5": {"x1": 0.7911, "x2": 0.9833, "y1": 0.6019, "y2": 0.9843},
}

REGION_KEYS = tuple(IMPORT_REGIONS.keys())

# Region order for the per-request server log, while events still stream to the client in completion order
LOG_ORDER = ("character", "watermark", "weapon", "forte", "sequences", "echo1", "echo2", "echo3", "echo4", "echo5")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global r2_image_store

    r2_image_store = R2ImageStore(R2_SETTINGS)
    print(
        f"Server starting on port {PORT} | railway={IS_RAILWAY} "
        f"workers={MAX_WORKERS} opencv_threads={OPENCV_THREADS} "
        f"r2_upload={R2_SETTINGS.enabled} r2_timeout={R2_SETTINGS.timeout_seconds}s",
        flush=True,
    )
    # Warm every worker in the background so the first request skips the ~3-7s module and SIFT template load
    # Not awaited so a failure is non-fatal and doesn't block the port bind or healthcheck
    loop = asyncio.get_running_loop()

    async def _warm():
        try:
            started = time.perf_counter()
            await asyncio.gather(*[
                loop.run_in_executor(executor, warm_worker) for _ in range(MAX_WORKERS)
            ])
            print(f"import: warmed {MAX_WORKERS} workers in {(time.perf_counter()-started)*1000:.0f}ms", flush=True)
        except Exception as exc:
            print(f"import: warmup failed (non-fatal): {exc}", flush=True)

    warm_task = asyncio.create_task(_warm())
    try:
        yield
    finally:
        warm_task.cancel()
        if active_storage_tasks:
            await asyncio.gather(*active_storage_tasks, return_exceptions=True)
        r2_image_store.close()
        print("Server shutting down", flush=True)
        executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)
def worker_init():
    """Ensure worker output is flushed"""
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
            "logs": result.get("logs", []),
            "elapsedMs": (time.perf_counter() - started) * 1000,
        }
    except Exception as exc:
        return {
            "region": region,
            "success": False,
            "analysis": None,
            "error": str(exc),
            "logs": [],
            "elapsedMs": (time.perf_counter() - started) * 1000,
        }

def warm_worker(hold: float = 2.0) -> bool:
    """Load a worker's modules and SIFT templates at boot instead of on the first request

    Random noise so SIFT finds keypoints and the echo and Tesseract paths all run
    Trailing sleep keeps the worker busy so the pool spawns all MAX_WORKERS instead of reusing a warm one
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (400, 360, 3), dtype=np.uint8)
    for region in ("character", "weapon", "echo1"):
        try:
            process_card(img, region)
        except Exception:
            pass
    time.sleep(hold)
    return True

executor = ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=worker_init
)
rate_limiter = RateLimiter(REQUESTS_PER_MINUTE)
report_rate_limiter = RateLimiter(REPORTS_PER_MINUTE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/api/report-ocr-issue":
        if INTERNAL_API_KEY and not is_trusted_proxy_request(request):
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "reason": "Report endpoint requires the trusted gateway.",
                },
            )

        rate_limit_identity = get_rate_limit_identity(request)
        if not report_rate_limiter.is_allowed(rate_limit_identity):
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "reason": "Too many issue reports. Please try again later.",
                },
                headers={"Retry-After": "60"},
            )
    elif request.url.path == "/api/ocr":
        rate_limit_identity = get_rate_limit_identity(request)
        if not rate_limiter.is_allowed(rate_limit_identity):
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "Rate limit exceeded. Please try again later.",
                },
                headers={"Retry-After": "60"},
            )
    response = await call_next(request)
    return response

# Multipart framing (boundaries, part header) around one 5 MiB image
MAX_MULTIPART_BYTES = MAX_IMAGE_BYTES + 64 * 1024

def reject_oversized_declaration(request: Request, limit: int) -> None:
    """Turn away an upload that declares up front it is too large"""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail="Image exceeds the 5 MiB limit.")


async def read_bounded_body(request: Request, limit: int) -> bytes:
    """Read the raw body, stopping as soon as it passes the limit

    request.body() buffers everything before any length check, making the limit a report rather than a cap
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="Image exceeds the 5 MiB limit.")
        chunks.append(chunk)
    return b"".join(chunks)


async def read_upload_image_bytes(request: Request) -> bytes:
    content_type = request.headers.get("content-type", "").lower()

    if content_type.startswith("multipart/form-data"):
        reject_oversized_declaration(request, MAX_MULTIPART_BYTES)
        form = await request.form()
        value = form.get("image")
        if not isinstance(value, UploadFile) and not hasattr(value, "read"):
            raise HTTPException(status_code=400, detail="Missing multipart file field 'image'.")
        try:
            image_bytes = await value.read()
        finally:
            close = getattr(value, "close", None)
            if callable(close):
                close_result = close()
                if inspect.isawaitable(close_result):
                    await close_result
    else:
        image_bytes = await read_bounded_body(request, MAX_IMAGE_BYTES)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Missing image bytes.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 5 MiB limit.",
        )

    return image_bytes


@app.post("/api/report-ocr-issue")
async def report_ocr_issue(request: Request):
    return await handle_issue_report(request, r2_image_store)

def ndjson_event(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"


def new_scan_id() -> str:
    """Cross-service correlation id for one scan"""
    return str(uuid.uuid4())


def start_storage_task(
    image_bytes: bytes,
    image_identity: ImageIdentity,
    scan_id: str,
) -> asyncio.Task[StorageResult]:
    """Start and retain storage even if recognition returns an early error"""
    task = asyncio.create_task(
        r2_image_store.store(image_bytes, image_identity),
        name=f"r2-store-{scan_id}",
    )
    active_storage_tasks.add(task)
    task.add_done_callback(active_storage_tasks.discard)
    return task


def log_deferred_storage_result(
    scan_id: str,
    image_identity: ImageIdentity,
    task: asyncio.Task[StorageResult],
) -> None:
    """Log the eventual result of an upload that outlived OCR recognition"""
    try:
        result = task.result()
    except asyncio.CancelledError:
        # A cancelled upload strands a build whose key was already handed out, so always log it
        log_event(
            "ocr_image_storage_completed",
            "ocr_image_storage_completed r2=cancelled",
            level="warning",
            scan_id=scan_id,
            hash_prefix=image_identity.hash_prefix,
            r2_result="cancelled",
        )
        return
    except Exception as exc:
        log_event(
            "ocr_image_storage_completed",
            f"ocr_image_storage_completed r2=failed ({type(exc).__name__})",
            level="warning",
            scan_id=scan_id,
            hash_prefix=image_identity.hash_prefix,
            r2_result="failed",
            r2_error_code=type(exc).__name__,
        )
        return

    log_event(
        "ocr_image_storage_completed",
        (
            f"ocr_image_storage_completed r2={result.result} "
            f"in {round(result.elapsed_ms)}ms"
        ),
        level="warning" if result.result in {"failed", "timed_out"} else "info",
        scan_id=scan_id,
        hash_prefix=image_identity.hash_prefix,
        r2_result=result.result,
        r2_ms=round(result.elapsed_ms, 2),
        r2_error_code=result.error_code,
    )

def slow_region_summary(timings: dict[str, Any]) -> str:
    region_timings = timings.get("regions") if isinstance(timings, dict) else None
    if not isinstance(region_timings, dict):
        return ""
    return ",".join(
        f"{name}:{elapsed:.0f}"
        for name, elapsed in sorted(region_timings.items(), key=lambda item: item[1], reverse=True)[:4]
    )

def detect_unsupported_language(analysis: dict[str, Any]) -> bool:
    """Aggregate card.py's per-echo langSignal into a card-level non-English verdict

    Needs NONENGLISH_MIN_VALUES readable values so wrong screenshots and odd layouts aren't mislabeled
    Non-English when the English name match rate falls below NONENGLISH_NAME_MATCH_FLOOR
    """
    good = total = values = 0
    for region in ("echo1", "echo2", "echo3", "echo4", "echo5"):
        entry = analysis.get(region)
        sig = entry.get("langSignal") if isinstance(entry, dict) else None
        if isinstance(sig, dict):
            good += int(sig.get("nameGood", 0))
            total += int(sig.get("nameTotal", 0))
            values += int(sig.get("numValues", 0))
    if total == 0 or values < NONENGLISH_MIN_VALUES:
        return False
    return (good / total) < NONENGLISH_NAME_MATCH_FLOOR

def log_import_completed(
    result: dict[str, Any],
    region_logs: dict[str, list],
    hash_prefix: str,
    storage_result: StorageResult,
) -> None:
    """Emit ordered recognition diagnostics and one structured completion event"""
    timings = result.get("timings", {})
    lines = [
        f"  {region}: {entry}"
        for region in LOG_ORDER
        for entry in region_logs.get(region, [])
    ]
    scan_id = result.get("scanId")
    if lines:
        print(
            f"import: regions scan_id={scan_id}\n" + "\n".join(lines),
            flush=True,
        )

    log_event(
        "ocr_import_completed",
        (
            f"ocr_import_completed r2={storage_result.result} "
            f"wall={timings.get('wallMs')}ms"
        ),
        level="warning" if storage_result.result in {"failed", "timed_out"} else "info",
        scan_id=scan_id,
        bytes=result.get("image", {}).get("bytes"),
        media_type=result.get("image", {}).get("mediaType"),
        hash_prefix=hash_prefix,
        r2_result=storage_result.result,
        r2_ms=timings.get("r2Ms"),
        r2_error_code=storage_result.error_code,
        storage_wait_ms=timings.get("storageWaitMs"),
        hash_ms=timings.get("hashMs"),
        ocr_wall_ms=timings.get("recognitionWallMs"),
        wall_ms=timings.get("wallMs"),
        unsupported_language=bool(result.get("unsupportedLanguage")),
        slow_regions=slow_region_summary(timings),
    )

async def stream_full_import_image(
    image_bytes: bytes,
    body_read_ms: float,
    request_start: float,
    scan_id: str,
):
    global consecutive_500s

    timing_start = time.perf_counter()

    hash_started = time.perf_counter()
    try:
        image_identity = identify_image(image_bytes)
    except UnsupportedImageType as exc:
        log_event(
            "ocr_import_rejected",
            "ocr_import_rejected unsupported_image_type",
            scan_id=scan_id,
            reason="unsupported_image_type",
            bytes=len(image_bytes),
        )
        yield ndjson_event({
            "type": "error",
            "success": False,
            "scanId": scan_id,
            "error": str(exc),
        })
        return
    hashed_at = time.perf_counter()

    # Dimensions gated on header before decoding because cv2.imdecode allocates whatever the header declares
    integrity = validate_header_dimensions(image_bytes)

    decode_started = time.perf_counter()
    image = None
    if integrity is None:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    decoded_at = time.perf_counter()

    if integrity is None:
        if image is None:
            yield ndjson_event({
                "type": "error",
                "success": False,
                "scanId": scan_id,
                "error": "Failed to decode image.",
            })
            return
        integrity = validate_image_integrity(image)

    if not integrity["accepted"]:
        log_event(
            "ocr_import_rejected",
            f"ocr_import_rejected {','.join(integrity['reasons'])}",
            level="warning",
            scan_id=scan_id,
            reason=",".join(integrity["reasons"]),
            bytes=len(image_bytes),
            media_type=image_identity.content_type,
            hash_prefix=image_identity.hash_prefix,
            integrity=integrity,
        )
        # Client gets the message only, since a live integrity score gives cheaters a reference
        yield ndjson_event({
            "type": "error",
            "success": False,
            "scanId": scan_id,
            "error": (
                integrity.get("message")
                or "This image does not match a supported build card."
            ),
        })
        return

    crop_started = time.perf_counter()
    crops = {
        region: crop_region(image, coords)
        for region, coords in IMPORT_REGIONS.items()
    }
    cropped_at = time.perf_counter()

    image_meta = {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "bytes": len(image_bytes),
        "mediaType": image_identity.content_type,
    }

    loop = asyncio.get_running_loop()
    recognition_started = time.perf_counter()

    async def run_region(region: str) -> dict[str, Any]:
        return await loop.run_in_executor(executor, process_region_task, (region, crops[region]))

    storage_enabled = r2_image_store.settings.enabled
    # Storage starts alongside recognition so the upload is usually done by the time the response hands out its key
    storage_started_at = time.perf_counter()
    storage_task = start_storage_task(image_bytes, image_identity, scan_id)
    tasks = [asyncio.create_task(run_region(region)) for region in REGION_KEYS]
    yield ndjson_event({
        "type": "meta",
        "scanId": scan_id,
        "sourceImageKey": image_identity.key if storage_enabled else None,
        "image": image_meta,
    })
    analysis: dict[str, Any] = {}
    progress: dict[str, str] = {}
    region_timings: dict[str, float] = {}
    region_errors: dict[str, str] = {}
    region_logs: dict[str, list] = {}

    try:
        for completed in asyncio.as_completed(tasks, timeout=PROCESS_TIMEOUT):
            result = await completed
            region = result["region"]
            elapsed_ms = round(float(result["elapsedMs"]), 2)
            region_timings[region] = elapsed_ms
            region_logs[region] = result.get("logs") or []

            if result["success"] and result["analysis"] is not None:
                analysis[region] = result["analysis"]
                progress[region] = "done"
                yield ndjson_event({
                    "type": "region",
                    "scanId": scan_id,
                    "region": region,
                    "status": "done",
                    "analysis": result["analysis"],
                    "elapsedMs": elapsed_ms,
                })
            else:
                error = str(result.get("error") or "Region recognition failed")
                progress[region] = "error"
                region_errors[region] = error
                yield ndjson_event({
                    "type": "region",
                    "scanId": scan_id,
                    "region": region,
                    "status": "error",
                    "error": error,
                    "elapsedMs": elapsed_ms,
                })
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()
        yield ndjson_event({
            "type": "error",
            "success": False,
            "scanId": scan_id,
            "error": f"Processing timeout exceeded ({PROCESS_TIMEOUT} seconds)",
        })
        return
    except Exception as exc:
        error_msg = str(exc)
        if "terminated abruptly" in error_msg.lower():
            force_restart(f"ProcessPool worker terminated abruptly: {error_msg}")
        yield ndjson_event({
            "type": "error",
            "success": False,
            "scanId": scan_id,
            "error": f"Image processing error: {error_msg}",
        })
        return

    recognized_at = time.perf_counter()
    for region in REGION_KEYS:
        if region not in progress:
            progress[region] = "error"
            region_errors[region] = "Region recognition did not complete"

    # Echo-bed integrity logged but not enforced: pasted stat cells break the gradient, but so do wrapped names
    bed = echo_bed_score(image)
    if bed["score"] >= BED_OBSERVE_SCORE_FLOOR:
        log_event(
            "echo_bed_observed",
            f"echo_bed_observed (not enforced) score={bed['score']:.2f}",
            scan_id=scan_id,
            hash_prefix=image_identity.hash_prefix,
            bed_score=round(bed["score"], 2),
            bed_panels=[round(p, 2) for p in bed["panels"]],
            chrome_score=integrity.get("chromeScore"),
        )

    # Key is the SHA-256 of the request bytes, so OCR returns it without waiting on a stalled R2 upload
    # trainingImageKey stays confirmation-only for consumers that need the object to exist (issue reports)
    # One scheduler turn lets an already-finishing upload return its confirmed key
    await asyncio.sleep(0)
    if storage_task.done():
        storage_result = await storage_task
    else:
        storage_result = StorageResult(
            result="pending",
            elapsed_ms=(time.perf_counter() - storage_started_at) * 1000,
            key=image_identity.key,
        )
        storage_task.add_done_callback(
            lambda task: log_deferred_storage_result(scan_id, image_identity, task)
        )
    finished_at = time.perf_counter()
    timings = {
        "hashMs": round((hashed_at - hash_started) * 1000, 2),
        "decodeMs": round((decoded_at - decode_started) * 1000, 2),
        "cropMs": round((cropped_at - crop_started) * 1000, 2),
        "recognitionWallMs": round((recognized_at - recognition_started) * 1000, 2),
        "r2Ms": (
            round(storage_result.elapsed_ms, 2)
            if storage_result.result != "pending"
            else None
        ),
        "storageWaitMs": 0.0,
        "totalMs": round((finished_at - timing_start) * 1000, 2),
        "bodyReadMs": round(body_read_ms, 2),
        "wallMs": round((time.perf_counter() - request_start) * 1000, 2),
        "regions": region_timings,
    }
    result = {
        "success": True,
        "scanId": scan_id,
        "analysis": analysis,
        "progress": progress,
        "timings": timings,
        "sourceImageKey": (
            image_identity.key
            if storage_enabled and storage_result.result not in {"disabled", "failed"}
            else None
        ),
        "trainingImageKey": (
            storage_result.key
            if storage_result.result in {"stored", "already_present"}
            else None
        ),
        "storage": storage_result.public_payload(),
        "regionErrors": region_errors,
        "image": image_meta,
        "unsupportedLanguage": detect_unsupported_language(analysis),
    }
    log_import_completed(
        result,
        region_logs,
        image_identity.hash_prefix,
        storage_result,
    )
    consecutive_500s = 0
    yield ndjson_event({"type": "done", **result})

@app.post("/api/ocr")
async def process_image_request(request: Request):
    global consecutive_500s

    request_start = time.perf_counter()
    scan_id = new_scan_id()
        
    try:
        log_event("ocr_import_started", "ocr_import_started", scan_id=scan_id)

        body_read_started = time.perf_counter()
        image_bytes = await read_upload_image_bytes(request)
        body_read_ms = (time.perf_counter() - body_read_started) * 1000
        return StreamingResponse(
            stream_full_import_image(
                image_bytes,
                body_read_ms,
                request_start,
                scan_id,
            ),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "success": False,
                "scanId": scan_id,
                "error": e.detail,
            }
        )
        
    except Exception as e:
        log_event(
            "ocr_import_failed",
            f"ocr_import_failed {type(e).__name__}: {e}",
            level="error",
            scan_id=scan_id,
            error_code=type(e).__name__,
        )

        consecutive_500s += 1
        if consecutive_500s > 1:
            print(f"Consecutive errors: {consecutive_500s}/{MAX_CONSECUTIVE_500S}", flush=True)
        
        if consecutive_500s >= MAX_CONSECUTIVE_500S:
            force_restart(f"Too many consecutive 500 errors ({consecutive_500s})")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "scanId": scan_id,
                "error": str(e),
            }
        )

@app.get("/", response_model=APIStatus)
async def homepage():
    return APIStatus()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    print(f"Uvicorn starting on 0.0.0.0:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
