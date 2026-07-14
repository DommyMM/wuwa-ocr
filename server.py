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
from image_integrity import validate_image_integrity, validate_ocr_integrity
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
REQUESTS_PER_MINUTE = int(os.getenv("OCR_RATE_LIMIT", "10"))
REPORTS_PER_MINUTE = int(os.getenv("OCR_REPORT_RATE_LIMIT", "5"))
PORT = int(os.getenv("PORT", "5000"))
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "").strip()
R2_SETTINGS = R2Settings.from_env()
r2_image_store = R2ImageStore.disabled(R2_SETTINGS.timeout_seconds)
active_storage_tasks: set[asyncio.Task[StorageResult]] = set()
consecutive_500s = 0
MAX_CONSECUTIVE_500S = 3

# Non-English detection thresholds (see card.echo_language_signal). A card is flagged
# non-English when its echo panels carry real values (so it IS a build card, not a wrong
# screenshot/odd layout) yet too few substat names match the English vocabulary.
NONENGLISH_NAME_MATCH_FLOOR = float(os.getenv("OCR_NONENGLISH_NAME_FLOOR", "0.35"))
NONENGLISH_MIN_VALUES = int(os.getenv("OCR_NONENGLISH_MIN_VALUES", "15"))

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
    "character": {"x1": 0.0000, "x2": 0.3200, "y1": 0.0000, "y2": 0.5500},
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

# Order for the consolidated per-request server-side log block. Region events stil stream to the client in completion order 
LOG_ORDER = ("character", "watermark", "weapon", "forte", "sequences", "echo1", "echo2", "echo3", "echo4", "echo5")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global r2_image_store

    r2_image_store = R2ImageStore(R2_SETTINGS)
    print(
        f"Server starting on port {PORT} | railway={IS_RAILWAY} gpu={USE_GPU} "
        f"workers={MAX_WORKERS} opencv_threads={OPENCV_THREADS} "
        f"r2_upload={R2_SETTINGS.enabled} r2_timeout={R2_SETTINGS.timeout_seconds}s",
        flush=True,
    )
    # Warm every worker in the background: each worker process loads RapidOCR and
    # the SIFT templates on its first task (~3-7s cold). Doing it at boot moves
    # that cost off the first user's request. Backgrounded (not awaited) so it
    # never blocks the port bind / Railway healthcheck, and a failure is non-fatal.
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
    """Force a worker to load its OCR engines and SIFT templates.

    Models load lazily on a worker's first real task; running a throwaway
    recognition here at boot pays that import cost off the user path. Random
    noise (not black) so SIFT finds keypoints and the echo sweep + RapidOCR +
    Tesseract paths all execute. Errors are swallowed: the goal is to warm the
    process, not to produce a result.

    The trailing sleep holds the worker busy so that when MAX_WORKERS of these
    run concurrently the pool is forced to spawn *every* worker (each paying its
    model load) instead of reusing one already-warm worker for all the warm
    tasks. Without it the pool drains the quick tasks on one or two workers and
    the rest stay cold, so the first real request still eats a ~3s cold load
    (observed in prod as character:3064ms).
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
        try:
            image_bytes = await value.read()
        finally:
            close = getattr(value, "close", None)
            if callable(close):
                close_result = close()
                if inspect.isawaitable(close_result):
                    await close_result
    else:
        image_bytes = await request.body()

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
    """Return the one canonical cross-service correlation identifier format."""

    return str(uuid.uuid4())


def start_storage_task(
    image_bytes: bytes,
    image_identity: ImageIdentity,
    scan_id: str,
) -> asyncio.Task[StorageResult]:
    """Start and retain storage even if recognition returns an early error."""

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
    """Log the eventual result of an upload that outlived OCR recognition."""

    try:
        result = task.result()
    except asyncio.CancelledError:
        # A cancelled retained upload is the one case that silently strands a
        # build: the response already handed out the optimistic key, so the row
        # points at an object nobody will ever write. Never swallow it.
        print(
            json.dumps({
                "event": "ocr_image_storage_completed",
                "message": "ocr_image_storage_completed r2=cancelled",
                "level": "warning",
                "scan_id": scan_id,
                "hash_prefix": image_identity.hash_prefix,
                "r2_result": "cancelled",
            }, separators=(",", ":")),
            flush=True,
        )
        return
    except Exception as exc:
        print(
            json.dumps({
                "event": "ocr_image_storage_completed",
                "message": f"ocr_image_storage_completed r2=failed ({type(exc).__name__})",
                "level": "warning",
                "scan_id": scan_id,
                "hash_prefix": image_identity.hash_prefix,
                "r2_result": "failed",
                "r2_error_code": type(exc).__name__,
            }, separators=(",", ":")),
            flush=True,
        )
        return

    print(
        json.dumps({
            "event": "ocr_image_storage_completed",
            "message": (
                f"ocr_image_storage_completed r2={result.result} "
                f"in {round(result.elapsed_ms)}ms"
            ),
            "level": (
                "warning"
                if result.result in {"failed", "timed_out"}
                else "info"
            ),
            "scan_id": scan_id,
            "hash_prefix": image_identity.hash_prefix,
            "r2_result": result.result,
            "r2_ms": round(result.elapsed_ms, 2),
            "r2_error_code": result.error_code,
        }, separators=(",", ":")),
        flush=True,
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
    """Aggregate card.py's per-echo langSignal into a card-level non-English verdict.

    Non-English = a real build card (>= NONENGLISH_MIN_VALUES readable values across the
    echoes) whose substat NAMES match the English vocabulary below NONENGLISH_NAME_MATCH_FLOOR.
    The value gate keeps wrong screenshots and non-standard layouts from being mislabeled.
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
    """Emit ordered recognition diagnostics and one structured completion event."""

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

    completion = {
        "event": "ocr_import_completed",
        # Railway parses any JSON log line as a structured log and renders
        # `message`. Without it the whole event ships as a blank line and
        # matches no log search, which hid every r2_result in production.
        "message": (
            f"ocr_import_completed r2={storage_result.result} "
            f"wall={timings.get('wallMs')}ms"
        ),
        "level": (
            "warning"
            if storage_result.result in {"failed", "timed_out"}
            else "info"
        ),
        "scan_id": scan_id,
        "bytes": result.get("image", {}).get("bytes"),
        "media_type": result.get("image", {}).get("mediaType"),
        "hash_prefix": hash_prefix,
        "r2_result": storage_result.result,
        "r2_ms": timings.get("r2Ms"),
        "r2_error_code": storage_result.error_code,
        "storage_wait_ms": timings.get("storageWaitMs"),
        "hash_ms": timings.get("hashMs"),
        "ocr_wall_ms": timings.get("recognitionWallMs"),
        "wall_ms": timings.get("wallMs"),
        "unsupported_language": bool(result.get("unsupportedLanguage")),
        "slow_regions": slow_region_summary(timings),
    }
    print(
        json.dumps(completion, separators=(",", ":"), ensure_ascii=False),
        flush=True,
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
        print(
            json.dumps({
                "event": "ocr_import_rejected",
                "message": "ocr_import_rejected unsupported_image_type",
                "level": "info",
                "scan_id": scan_id,
                "reason": "unsupported_image_type",
                "bytes": len(image_bytes),
            }, separators=(",", ":")),
            flush=True,
        )
        yield ndjson_event({
            "type": "error",
            "success": False,
            "scanId": scan_id,
            "error": str(exc),
        })
        return
    hashed_at = time.perf_counter()

    decode_started = time.perf_counter()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    decoded_at = time.perf_counter()

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
        print(
            json.dumps({
                "event": "ocr_import_rejected",
                "message": f"ocr_import_rejected {','.join(integrity['reasons'])}",
                "level": "warning",
                "scan_id": scan_id,
                "reason": ",".join(integrity["reasons"]),
                "bytes": len(image_bytes),
                "media_type": image_identity.content_type,
                "hash_prefix": image_identity.hash_prefix,
                "integrity": integrity,
            }, separators=(",", ":")),
            flush=True,
        )
        yield ndjson_event({
            "type": "error",
            "success": False,
            "scanId": scan_id,
            "error": (
                integrity.get("message")
                or "This image does not match a supported build card."
            ),
            "integrity": integrity,
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
    requires_ocr_validation = bool(integrity.get("requiresOcrValidation"))
    # Storage always runs concurrently with recognition, including for suspect
    # images. Deferring it until after an OCR verdict guaranteed a `pending`
    # result at response time, which hands the frontend an optimistic key for an
    # upload that has barely started.
    storage_started_at = time.perf_counter()
    storage_task = start_storage_task(image_bytes, image_identity, scan_id)
    tasks = [asyncio.create_task(run_region(region)) for region in REGION_KEYS]
    yield ndjson_event({
        "type": "meta",
        "scanId": scan_id,
        "sourceImageKey": image_identity.key if storage_enabled else None,
        "image": image_meta,
        "integrity": integrity,
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

    if requires_ocr_validation:
        # OBSERVE-ONLY. This gate rejected 88% of the images the fast triage
        # escalates, measured over the r2-backup corpus — ~3% of all real
        # uploads, against the ~0.06% that are actually invalid. Escalation
        # correlates with lossy re-encodes (Discord/WebP), which degrade OCR
        # without making the card fake, and the editor already lets a user fix a
        # misread. Nothing here may reject until the signal is rebuilt around
        # the QR anchor and echo-panel geometry.
        ocr_integrity = validate_ocr_integrity(analysis)
        integrity["ocr"] = ocr_integrity
        integrity["initialVerdict"] = integrity["verdict"]
        integrity["verdict"] = "ok"
        integrity["accepted"] = True
        integrity["requiresOcrValidation"] = False
        integrity["resolvedBy"] = (
            "ocr_structure" if ocr_integrity["accepted"] else "ocr_structure_observed"
        )
        if not ocr_integrity["accepted"]:
            print(
                json.dumps({
                    "event": "ocr_integrity_observed",
                    "message": (
                        "ocr_integrity_observed (not enforced) "
                        f"{','.join(ocr_integrity['reasons'])}"
                    ),
                    "level": "info",
                    "scan_id": scan_id,
                    "reason": ",".join(ocr_integrity["reasons"]),
                    "hash_prefix": image_identity.hash_prefix,
                    "integrity": integrity,
                }, separators=(",", ":")),
                flush=True,
            )

    # The object name is already definitive because it is the SHA-256 of the
    # exact request bytes. Do not hold completed OCR behind a stalled R2 call:
    # return the optimistic sourceImageKey and retain the upload in the process.
    # trainingImageKey remains confirmation-only for consumers that must know
    # the object exists (notably issue reports).
    # Give an already-completing storage coroutine one non-blocking scheduler
    # turn so normal fast uploads retain the confirmed-key response shape.
    await asyncio.sleep(0)
    assert storage_task is not None
    assert storage_started_at is not None
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
        "integrity": integrity,
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
        print(
            json.dumps({
                "event": "ocr_import_started",
                "message": "ocr_import_started",
                "level": "info",
                "scan_id": scan_id,
            }, separators=(",", ":")),
            flush=True,
        )

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
        print(
            json.dumps({
                "event": "ocr_import_failed",
                "message": f"ocr_import_failed {type(e).__name__}: {e}",
                "level": "error",
                "scan_id": scan_id,
                "error_code": type(e).__name__,
            }, separators=(",", ":")),
            flush=True,
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
