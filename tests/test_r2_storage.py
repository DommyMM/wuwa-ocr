from __future__ import annotations

import asyncio
import base64
import hashlib
import time
import unittest

from botocore.exceptions import ClientError

from r2_storage import (
    R2ConfigurationError,
    R2ImageStore,
    R2OperationTimeout,
    R2Settings,
    UnsupportedImageType,
    identify_image,
)


JPEG_BYTES = b"\xff\xd8\xff\xe0exact-jpeg-input\xff\xd9"
PNG_BYTES = b"\x89PNG\r\n\x1a\nexact-png-input"


def missing_object_error() -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "404", "Message": "Not Found"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        "HeadObject",
    )


def service_error() -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "InternalError", "Message": "failed"},
            "ResponseMetadata": {"HTTPStatusCode": 500},
        },
        "HeadObject",
    )


def precondition_error() -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "PreconditionFailed", "Message": "exists"},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        },
        "PutObject",
    )


def enabled_settings(timeout: float = 1.0) -> R2Settings:
    return R2Settings(
        enabled=True,
        timeout_seconds=timeout,
        account_id="account",
        access_key_id="access",
        secret_access_key="secret",
        bucket_name="bucket",
    )


class FakeClient:
    def __init__(self, head_result=None, head_error: Exception | None = None):
        self.head_result = head_result
        self.head_error = head_error
        self.head_calls = []
        self.put_calls = []

    def head_object(self, **kwargs):
        self.head_calls.append(kwargs)
        if self.head_error:
            raise self.head_error
        return self.head_result

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        return {"ETag": '"etag"'}


class SlowMissingClient(FakeClient):
    def __init__(self, delay: float):
        super().__init__(head_error=missing_object_error())
        self.delay = delay

    def head_object(self, **kwargs):
        time.sleep(self.delay)
        return super().head_object(**kwargs)


class SlowPutClient(FakeClient):
    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay

    def put_object(self, **kwargs):
        time.sleep(self.delay)
        return super().put_object(**kwargs)


class ConcurrentWriterClient(FakeClient):
    def __init__(self, image_bytes: bytes, digest: str):
        super().__init__()
        self.image_bytes = image_bytes
        self.digest = digest

    def head_object(self, **kwargs):
        self.head_calls.append(kwargs)
        if len(self.head_calls) == 1:
            raise missing_object_error()
        return {
            "ContentLength": len(self.image_bytes),
            "Metadata": {"sha256": self.digest},
        }

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        raise precondition_error()


class ImageIdentityTests(unittest.TestCase):
    def test_jpeg_uses_full_sha256_and_magic_extension(self):
        identity = identify_image(JPEG_BYTES)

        self.assertEqual(identity.digest_hex, hashlib.sha256(JPEG_BYTES).hexdigest())
        self.assertEqual(identity.key, f"{identity.digest_hex}.jpg")
        self.assertEqual(identity.content_type, "image/jpeg")
        self.assertEqual(len(identity.digest_hex), 64)

    def test_png_uses_png_extension(self):
        identity = identify_image(PNG_BYTES)

        self.assertEqual(identity.key, f"{hashlib.sha256(PNG_BYTES).hexdigest()}.png")
        self.assertEqual(identity.content_type, "image/png")

    def test_declared_filename_or_mime_cannot_override_magic(self):
        self.assertEqual(identify_image(JPEG_BYTES).extension, "jpg")
        self.assertEqual(identify_image(PNG_BYTES).extension, "png")

    def test_unsupported_magic_is_rejected(self):
        with self.assertRaises(UnsupportedImageType):
            identify_image(b"RIFFxxxxWEBPunsupported")

    def test_same_exact_bytes_are_stable_across_retries(self):
        self.assertEqual(identify_image(JPEG_BYTES).key, identify_image(JPEG_BYTES).key)


class SettingsTests(unittest.TestCase):
    def test_disabled_defaults_to_five_second_timeout(self):
        settings = R2Settings.from_env({})

        self.assertFalse(settings.enabled)
        self.assertEqual(settings.timeout_seconds, 5.0)

    def test_enabled_requires_every_r2_setting(self):
        with self.assertRaises(R2ConfigurationError):
            R2Settings.from_env({"OCR_R2_UPLOAD_ENABLED": "1"})

    def test_timeout_must_be_positive(self):
        with self.assertRaises(R2ConfigurationError):
            R2Settings.from_env({"OCR_R2_TIMEOUT_SECONDS": "0"})


class R2ImageStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_report_is_created_without_overwrite(self):
        client = FakeClient()
        store = R2ImageStore(enabled_settings(), client=client)
        body = b'{"schemaVersion":1}'

        await store.put_json_object("reports/2026/07/11/report.json", body)

        self.assertEqual(client.put_calls, [{
            "Bucket": "bucket",
            "Key": "reports/2026/07/11/report.json",
            "Body": body,
            "ContentType": "application/json; charset=utf-8",
            "IfNoneMatch": "*",
        }])

    async def test_json_report_write_uses_the_r2_deadline(self):
        store = R2ImageStore(
            enabled_settings(timeout=0.01),
            client=SlowPutClient(delay=0.08),
        )
        started = time.perf_counter()

        with self.assertRaises(R2OperationTimeout):
            await store.put_json_object("reports/report.json", b"{}")

        self.assertLess(time.perf_counter() - started, 0.06)

    async def test_disabled_store_rejects_report_operations(self):
        store = R2ImageStore.disabled()

        with self.assertRaises(R2ConfigurationError):
            await store.put_json_object("reports/report.json", b"{}")

    async def test_missing_object_uploads_exact_bytes_and_checksum(self):
        client = FakeClient(head_error=missing_object_error())
        store = R2ImageStore(enabled_settings(), client=client)
        identity = identify_image(JPEG_BYTES)

        result = await store.store(JPEG_BYTES, identity)

        self.assertEqual(result.result, "stored")
        self.assertEqual(result.key, identity.key)
        self.assertEqual(len(client.put_calls), 1)
        put = client.put_calls[0]
        self.assertEqual(put["Key"], identity.key)
        self.assertEqual(put["Body"], JPEG_BYTES)
        self.assertEqual(put["ContentType"], "image/jpeg")
        self.assertEqual(
            put["ChecksumSHA256"],
            base64.b64encode(hashlib.sha256(JPEG_BYTES).digest()).decode("ascii"),
        )
        self.assertEqual(put["Metadata"], {"sha256": identity.digest_hex})
        self.assertEqual(put["IfNoneMatch"], "*")

    async def test_concurrent_writer_is_revalidated_without_overwrite(self):
        identity = identify_image(JPEG_BYTES)
        client = ConcurrentWriterClient(JPEG_BYTES, identity.digest_hex)
        store = R2ImageStore(enabled_settings(), client=client)

        result = await store.store(JPEG_BYTES, identity)

        self.assertEqual(result.result, "already_present")
        self.assertEqual(result.key, identity.key)
        self.assertEqual(len(client.head_calls), 2)
        self.assertEqual(len(client.put_calls), 1)

    async def test_existing_object_skips_put(self):
        client = FakeClient(head_result={"ContentLength": len(PNG_BYTES)})
        store = R2ImageStore(enabled_settings(), client=client)
        identity = identify_image(PNG_BYTES)

        result = await store.store(PNG_BYTES, identity)

        self.assertEqual(result.result, "already_present")
        self.assertEqual(result.key, identity.key)
        self.assertEqual(client.put_calls, [])

    async def test_existing_size_mismatch_is_failure(self):
        client = FakeClient(head_result={"ContentLength": len(PNG_BYTES) + 1})
        store = R2ImageStore(enabled_settings(), client=client)

        result = await store.store(PNG_BYTES, identify_image(PNG_BYTES))

        self.assertEqual(result.result, "failed")
        self.assertEqual(result.error_code, "existing_object_mismatch")
        self.assertIsNone(result.key)
        self.assertEqual(client.put_calls, [])

    async def test_existing_digest_metadata_mismatch_is_failure(self):
        client = FakeClient(head_result={
            "ContentLength": len(PNG_BYTES),
            "Metadata": {"sha256": "0" * 64},
        })
        store = R2ImageStore(enabled_settings(), client=client)

        result = await store.store(PNG_BYTES, identify_image(PNG_BYTES))

        self.assertEqual(result.result, "failed")
        self.assertEqual(result.error_code, "existing_object_mismatch")
        self.assertIsNone(result.key)

    async def test_r2_error_is_a_structured_failure(self):
        store = R2ImageStore(
            enabled_settings(),
            client=FakeClient(head_error=service_error()),
        )

        result = await store.store(JPEG_BYTES, identify_image(JPEG_BYTES))

        self.assertEqual(result.result, "failed")
        self.assertEqual(result.error_code, "InternalError")
        self.assertIsNone(result.key)

    async def test_timeout_bounds_the_async_result(self):
        timeout = 0.01
        store = R2ImageStore(
            enabled_settings(timeout),
            client=SlowMissingClient(delay=0.08),
        )
        started = time.perf_counter()

        result = await store.store(JPEG_BYTES, identify_image(JPEG_BYTES))

        wall = time.perf_counter() - started
        self.assertEqual(result.result, "timed_out")
        self.assertEqual(result.error_code, "deadline_exceeded")
        self.assertIsNone(result.key)
        self.assertLess(wall, 0.06)

    async def test_disabled_storage_never_needs_a_client(self):
        result = await R2ImageStore.disabled().store(
            JPEG_BYTES,
            identify_image(JPEG_BYTES),
        )

        self.assertEqual(result.result, "disabled")
        self.assertIsNone(result.key)


if __name__ == "__main__":
    unittest.main()
