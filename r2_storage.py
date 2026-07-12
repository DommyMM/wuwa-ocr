"""Bounded, content-addressed persistence of OCR inputs in Cloudflare R2."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from botocore.config import Config
from botocore.exceptions import ClientError


JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

StorageStatus = Literal[
    "stored",
    "already_present",
    "failed",
    "timed_out",
    "disabled",
]


class UnsupportedImageType(ValueError):
    """Raised when an OCR input is not a supported JPEG or PNG by magic bytes."""


class R2ConfigurationError(ValueError):
    """Raised when R2 upload is enabled without a complete, valid configuration."""


class ExistingObjectMismatch(RuntimeError):
    """Raised if a content-addressed key exists with an impossible size mismatch."""


class R2OperationTimeout(TimeoutError):
    """Raised when an auxiliary R2 operation exceeds the configured deadline."""


@dataclass(frozen=True)
class ImageIdentity:
    digest_hex: str
    digest_bytes: bytes = field(repr=False)
    extension: Literal["jpg", "png"]
    content_type: Literal["image/jpeg", "image/png"]

    @property
    def key(self) -> str:
        return f"{self.digest_hex}.{self.extension}"

    @property
    def hash_prefix(self) -> str:
        return self.digest_hex[:12]


@dataclass(frozen=True)
class StorageResult:
    result: StorageStatus
    elapsed_ms: float
    key: str | None = None
    error_code: str | None = None

    def public_payload(self) -> dict[str, str | float]:
        return {
            "result": self.result,
            "elapsedMs": round(self.elapsed_ms, 2),
        }


@dataclass(frozen=True)
class R2Settings:
    enabled: bool
    timeout_seconds: float
    account_id: str = ""
    access_key_id: str = field(default="", repr=False)
    secret_access_key: str = field(default="", repr=False)
    bucket_name: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "R2Settings":
        values = os.environ if env is None else env
        enabled = _parse_bool(values.get("OCR_R2_UPLOAD_ENABLED", "0"))

        raw_timeout = values.get("OCR_R2_TIMEOUT_SECONDS", "5").strip()
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError as exc:
            raise R2ConfigurationError(
                "OCR_R2_TIMEOUT_SECONDS must be a positive number"
            ) from exc
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise R2ConfigurationError(
                "OCR_R2_TIMEOUT_SECONDS must be a positive number"
            )

        settings = cls(
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            account_id=values.get("CLOUDFLARE_ACCOUNT_ID", "").strip(),
            access_key_id=values.get("R2_ACCESS_KEY_ID", "").strip(),
            secret_access_key=values.get("R2_SECRET_ACCESS_KEY", "").strip(),
            bucket_name=values.get("R2_BUCKET_NAME", "").strip(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.enabled:
            return

        missing = [
            name
            for name, value in (
                ("CLOUDFLARE_ACCOUNT_ID", self.account_id),
                ("R2_ACCESS_KEY_ID", self.access_key_id),
                ("R2_SECRET_ACCESS_KEY", self.secret_access_key),
                ("R2_BUCKET_NAME", self.bucket_name),
            )
            if not value
        ]
        if missing:
            raise R2ConfigurationError(
                "OCR_R2_UPLOAD_ENABLED requires: " + ", ".join(missing)
            )


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise R2ConfigurationError(
        "OCR_R2_UPLOAD_ENABLED must be one of 1/0, true/false, yes/no, or on/off"
    )


def identify_image(image_bytes: bytes) -> ImageIdentity:
    """Return the canonical identity for the exact supported input bytes."""

    if image_bytes.startswith(PNG_MAGIC):
        extension: Literal["jpg", "png"] = "png"
        content_type: Literal["image/jpeg", "image/png"] = "image/png"
    elif image_bytes.startswith(JPEG_MAGIC):
        extension = "jpg"
        content_type = "image/jpeg"
    else:
        raise UnsupportedImageType(
            "Unsupported image type. Upload a JPEG or PNG image."
        )

    digest = hashlib.sha256(image_bytes).digest()
    return ImageIdentity(
        digest_hex=digest.hex(),
        digest_bytes=digest,
        extension=extension,
        content_type=content_type,
    )


class R2ImageStore:
    """A reused boto3 client with a response-bounded asynchronous facade."""

    def __init__(self, settings: R2Settings, client: Any | None = None):
        settings.validate()
        self.settings = settings
        self._client = client
        if settings.enabled and self._client is None:
            self._client = self._create_client(settings)

    @classmethod
    def disabled(cls, timeout_seconds: float = 5.0) -> "R2ImageStore":
        return cls(R2Settings(enabled=False, timeout_seconds=timeout_seconds))

    @staticmethod
    def _create_client(settings: R2Settings) -> Any:
        import boto3

        # The outer asyncio deadline is authoritative. Matching SDK timeouts and
        # disabling retries prevents a timed-out to_thread call from occupying a
        # worker thread for substantially longer in the background.
        client_config = Config(
            connect_timeout=settings.timeout_seconds,
            read_timeout=settings.timeout_seconds,
            max_pool_connections=16,
            region_name="auto",
            retries={"mode": "standard", "total_max_attempts": 1},
            signature_version="s3v4",
        )
        return boto3.client(
            "s3",
            endpoint_url=(
                f"https://{settings.account_id}.r2.cloudflarestorage.com"
            ),
            aws_access_key_id=settings.access_key_id,
            aws_secret_access_key=settings.secret_access_key,
            region_name="auto",
            config=client_config,
        )

    async def store(
        self,
        image_bytes: bytes,
        identity: ImageIdentity,
    ) -> StorageResult:
        started = time.perf_counter()
        if not self.settings.enabled:
            return StorageResult(result="disabled", elapsed_ms=0.0)

        try:
            status = await asyncio.wait_for(
                asyncio.to_thread(self._store_sync, image_bytes, identity),
                timeout=self.settings.timeout_seconds,
            )
            return StorageResult(
                result=status,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                key=identity.key,
            )
        except TimeoutError:
            return StorageResult(
                result="timed_out",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error_code="deadline_exceeded",
            )
        except Exception as exc:
            return StorageResult(
                result="failed",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error_code=_error_code(exc),
            )

    async def put_json_object(self, key: str, body: bytes) -> None:
        """Create one JSON object without overwriting an existing report."""

        self._require_client()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._put_json_object_sync, key, body),
                timeout=self.settings.timeout_seconds,
            )
        except TimeoutError as exc:
            raise R2OperationTimeout("R2 report write timed out") from exc

    def _require_client(self) -> None:
        if not self.settings.enabled or self._client is None:
            raise R2ConfigurationError("R2 client is not configured")

    def _put_json_object_sync(self, key: str, body: bytes) -> None:
        self._require_client()
        self._client.put_object(
            Bucket=self.settings.bucket_name,
            Key=key,
            Body=body,
            ContentType="application/json; charset=utf-8",
            IfNoneMatch="*",
        )

    def _store_sync(
        self,
        image_bytes: bytes,
        identity: ImageIdentity,
    ) -> Literal["stored", "already_present"]:
        self._require_client()

        try:
            existing = self._client.head_object(
                Bucket=self.settings.bucket_name,
                Key=identity.key,
            )
        except ClientError as exc:
            if not _is_missing_object(exc):
                raise
        else:
            self._validate_existing(existing, image_bytes, identity)
            return "already_present"

        try:
            self._client.put_object(
                Bucket=self.settings.bucket_name,
                Key=identity.key,
                Body=image_bytes,
                ContentType=identity.content_type,
                ChecksumSHA256=base64.b64encode(identity.digest_bytes).decode("ascii"),
                Metadata={"sha256": identity.digest_hex},
                IfNoneMatch="*",
            )
        except ClientError as exc:
            if not _is_precondition_failed(exc):
                raise
            # Another identical request won the HEAD→PUT race. Re-read and
            # validate instead of overwriting a content-addressed object.
            existing = self._client.head_object(
                Bucket=self.settings.bucket_name,
                Key=identity.key,
            )
            self._validate_existing(existing, image_bytes, identity)
            return "already_present"
        return "stored"

    @staticmethod
    def _validate_existing(
        existing: Mapping[str, Any],
        image_bytes: bytes,
        identity: ImageIdentity,
    ) -> None:
        content_length = existing.get("ContentLength")
        if content_length is not None and int(content_length) != len(image_bytes):
            raise ExistingObjectMismatch(
                "Existing content-addressed object has an unexpected size"
            )
        stored_digest = existing.get("Metadata", {}).get("sha256")
        if stored_digest is not None and stored_digest != identity.digest_hex:
            raise ExistingObjectMismatch(
                "Existing content-addressed object has an unexpected digest"
            )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def _is_missing_object(exc: ClientError) -> bool:
    response = exc.response
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status == 404 or code in {"404", "NoSuchKey", "NotFound"}


def _is_precondition_failed(exc: ClientError) -> bool:
    response = exc.response
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status == 412 or code in {"412", "PreconditionFailed"}


def _error_code(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code:
            return code
    if isinstance(exc, ExistingObjectMismatch):
        return "existing_object_mismatch"
    return type(exc).__name__
