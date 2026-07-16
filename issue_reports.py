"""Validated OCR issue-report ingestion and R2 persistence."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from log_events import log_event
from r2_storage import UnsupportedImageType, identify_image


MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_REPORT_JSON_BYTES = 256 * 1024
MAX_REPORT_MULTIPART_BYTES = MAX_IMAGE_BYTES + MAX_REPORT_JSON_BYTES + 64 * 1024

SOURCE_IMAGE_KEY_PATTERN = re.compile(r"^[a-f0-9]{64}\.(?:jpg|png)$")
SCAN_ID_PATTERN = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$"
)


class OcrIssueReport(BaseModel):
    """A report is diagnostic material, so validation only guards what becomes
    an R2 key or an unbounded write. Unknown fields and unknown progress regions
    are carried through: a report about an unexpected client state is exactly the
    report worth keeping."""

    route: Literal["/import"]
    reason: Literal[
        "illegal_echo",
        "ocr_error",
        "validation_error",
        "manual_report",
    ]
    schemaVersion: int = 1
    scanId: str | None = Field(default=None, pattern=SCAN_ID_PATTERN.pattern)
    trainingImageKey: str | None = Field(
        default=None,
        pattern=SOURCE_IMAGE_KEY_PATTERN.pattern,
    )
    note: str = Field(default="", max_length=2000)
    progress: dict[str, str] = Field(default_factory=dict)
    analysisData: Any = None
    importedState: Any = None
    validationError: str | None = None
    ocrError: str | None = None
    lbUploadError: str | None = None
    uploadToLb: bool = False
    watermark: dict[str, Any] = Field(default_factory=dict)
    client: dict[str, Any] = Field(default_factory=dict)


class OcrIssueRequestError(ValueError):
    def __init__(self, status_code: int, reason: str):
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason


def issue_report_error(status_code: int, reason: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "reason": reason},
    )


def validate_declared_report_size(request: Request) -> None:
    raw_length = request.headers.get("content-length")
    if raw_length is None:
        return
    try:
        declared_length = int(raw_length)
    except ValueError as exc:
        raise OcrIssueRequestError(400, "Invalid Content-Length header.") from exc
    if declared_length < 0:
        raise OcrIssueRequestError(400, "Invalid Content-Length header.")
    if declared_length > MAX_REPORT_MULTIPART_BYTES:
        raise OcrIssueRequestError(413, "Issue report request is too large.")


async def read_report_image(upload: Any) -> bytes:
    image_bytes = await upload.read()
    if not image_bytes:
        raise OcrIssueRequestError(400, "Fallback image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise OcrIssueRequestError(413, "Image exceeds the 5 MiB limit.")
    return image_bytes


def parse_issue_report_json(raw_report: str) -> OcrIssueReport:
    if len(raw_report.encode("utf-8")) > MAX_REPORT_JSON_BYTES:
        raise OcrIssueRequestError(
            413,
            "Report metadata exceeds the 256 KiB limit.",
        )
    def reject_nonstandard_number(value: str) -> None:
        # NaN/Infinity would round-trip into R2 as invalid JSON and silently
        # corrupt the report dataset, so they are refused at the door.
        raise ValueError(f"Non-standard JSON number: {value}")

    try:
        payload = json.loads(raw_report, parse_constant=reject_nonstandard_number)
        if not isinstance(payload, dict):
            raise ValueError("report must be a JSON object")
        return OcrIssueReport.model_validate(payload)
    except (ValueError, ValidationError) as exc:
        raise OcrIssueRequestError(400, "Invalid issue report metadata.") from exc


async def read_issue_report_request(
    request: Request,
) -> tuple[OcrIssueReport, bytes | None]:
    content_type = request.headers.get("content-type", "").lower()
    if not content_type.startswith("multipart/form-data"):
        raise OcrIssueRequestError(
            415,
            "Issue reports must use multipart/form-data.",
        )
    validate_declared_report_size(request)

    try:
        # max_part_size bounds non-file parts only, so the report JSON is capped
        # here while the image part stays bounded by the gateway and the read below.
        form = await request.form(
            max_files=1,
            max_fields=1,
            max_part_size=MAX_REPORT_JSON_BYTES,
        )
    except Exception as exc:
        if "maximum size" in str(exc).lower():
            raise OcrIssueRequestError(
                413,
                "Report metadata exceeds the 256 KiB limit.",
            ) from exc
        raise OcrIssueRequestError(400, "Invalid issue report multipart body.") from exc

    try:
        report_parts = form.getlist("report")
        if len(report_parts) != 1 or not isinstance(report_parts[0], str):
            raise OcrIssueRequestError(
                400,
                "Issue report requires one report JSON field.",
            )
        report = parse_issue_report_json(report_parts[0])

        image_parts = form.getlist("image")
        if len(image_parts) > 1:
            raise OcrIssueRequestError(400, "Issue report accepts only one image.")
        image_bytes: bytes | None = None
        if image_parts:
            upload = image_parts[0]
            if not hasattr(upload, "read"):
                raise OcrIssueRequestError(400, "Image must be a file upload.")
            image_bytes = await read_report_image(upload)
    finally:
        await form.close()

    if report.trainingImageKey is None and image_bytes is None:
        raise OcrIssueRequestError(
            400,
            "Provide either a trainingImageKey or the original image.",
        )
    return report, image_bytes


def build_report_object_key(report_id: str, created_at: datetime) -> str:
    return (
        f"reports/{created_at.year:04d}/{created_at.month:02d}/"
        f"{created_at.day:02d}/{report_id}.json"
    )


def issue_report_payload(
    report: OcrIssueReport,
    report_id: str,
    created_at: datetime,
    training_image_key: str,
    image_storage: str,
) -> dict[str, Any]:
    created_at_iso = created_at.isoformat().replace("+00:00", "Z")
    client = dict(report.client)
    client["submittedAt"] = client.get("submittedAt") or created_at_iso
    return {
        "schemaVersion": report.schemaVersion,
        "reportId": report_id,
        "createdAt": created_at_iso,
        "source": "import",
        "scanId": report.scanId,
        "reason": report.reason,
        "note": report.note.strip(),
        "trainingImageKey": training_image_key,
        "imageStorage": image_storage,
        "route": "/import",
        "uploadToLb": report.uploadToLb,
        "watermark": report.watermark,
        "errors": {
            "validationError": report.validationError,
            "ocrError": report.ocrError,
            "lbUploadError": report.lbUploadError,
        },
        "progress": report.progress,
        "analysisData": report.analysisData,
        "importedState": report.importedState,
        "client": client,
    }


async def persist_issue_report(
    report: OcrIssueReport,
    image_bytes: bytes | None,
    image_store: Any,
) -> dict[str, str | bool]:
    if not image_store.settings.enabled:
        raise OcrIssueRequestError(
            503,
            "Issue report storage is not configured.",
        )

    # A confirmed key is one this backend already minted and stored, so it is
    # taken at face value rather than re-confirmed with an extra R2 round-trip.
    if report.trainingImageKey is not None:
        training_image_key = report.trainingImageKey
        image_storage = "referenced"
    else:
        if image_bytes is None:
            raise OcrIssueRequestError(400, "Fallback image is required.")
        try:
            identity = identify_image(image_bytes)
        except UnsupportedImageType as exc:
            raise OcrIssueRequestError(415, str(exc)) from exc
        storage_result = await image_store.store(image_bytes, identity)
        if (
            storage_result.result not in {"stored", "already_present"}
            or not storage_result.key
        ):
            raise OcrIssueRequestError(
                503,
                "Fallback image storage is temporarily unavailable.",
            )
        training_image_key = storage_result.key
        image_storage = storage_result.result

    report_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    report_key = build_report_object_key(report_id, created_at)
    payload = issue_report_payload(
        report,
        report_id,
        created_at,
        training_image_key,
        image_storage,
    )
    report_body = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    await image_store.put_json_object(report_key, report_body)

    log_event(
        "ocr_issue_report_stored",
        f"ocr_issue_report_stored reason={report.reason} image={image_storage}",
        report_id=report_id,
        scan_id=report.scanId,
        reason=report.reason,
        image_storage=image_storage,
    )
    return {
        "success": True,
        "reportId": report_id,
        "reportKey": report_key,
        "trainingImageKey": training_image_key,
        "imageStorage": image_storage,
    }


async def handle_issue_report(request: Request, image_store: Any) -> JSONResponse:
    try:
        report, image_bytes = await read_issue_report_request(request)
        receipt = await persist_issue_report(report, image_bytes, image_store)
        return JSONResponse(status_code=201, content=receipt)
    except OcrIssueRequestError as exc:
        return issue_report_error(exc.status_code, exc.reason)
    except Exception as exc:
        log_event(
            "ocr_issue_report_failed",
            f"ocr_issue_report_failed {type(exc).__name__}",
            level="error",
            error_code=type(exc).__name__,
        )
        return issue_report_error(
            503,
            "Issue report storage is temporarily unavailable.",
        )
