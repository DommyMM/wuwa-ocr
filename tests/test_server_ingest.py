from __future__ import annotations

import asyncio
import io
import json
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from unittest.mock import patch

import cv2
import numpy as np
from starlette.requests import Request

import server
from r2_storage import StorageResult


def encoded_image(extension: str) -> bytes:
    image = np.full((64, 96, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise AssertionError(f"could not encode {extension}")
    return encoded.tobytes()


def make_request(body: bytes, content_type: str) -> Request:
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/ocr",
        "raw_path": b"/api/ocr",
        "query_string": b"",
        "headers": [(b"content-type", content_type.encode("ascii"))],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    return Request(scope, receive)


class FakeStore:
    def __init__(self, status: str, delay: float = 0.0):
        self.status = status
        self.delay = delay
        self.calls = 0
        self.started_at: float | None = None
        self.finished_at: float | None = None

    async def store(self, image_bytes, identity):
        self.calls += 1
        self.started_at = time.perf_counter()
        if self.delay:
            await asyncio.sleep(self.delay)
        self.finished_at = time.perf_counter()
        return StorageResult(
            result=self.status,
            elapsed_ms=self.delay * 1000,
            key=(
                identity.key
                if self.status in {"stored", "already_present"}
                else None
            ),
            error_code=("test_failure" if self.status == "failed" else None),
        )


async def collect_stream(
    image_bytes: bytes,
    store: FakeStore,
    process_region,
    scan_id: str = "scan-test",
) -> list[dict]:
    events = []
    with ThreadPoolExecutor(max_workers=len(server.REGION_KEYS)) as test_executor:
        with (
            patch.object(server, "executor", test_executor),
            patch.object(server, "process_region_task", process_region),
            patch.object(server, "r2_image_store", store),
            redirect_stdout(io.StringIO()),
        ):
            async for line in server.stream_full_import_image(
                image_bytes,
                body_read_ms=3.5,
                request_start=time.perf_counter(),
                scan_id=scan_id,
            ):
                events.append(json.loads(line))
    return events


def successful_region(task):
    region, _crop = task
    return {
        "region": region,
        "success": True,
        "analysis": {"region": region},
        "error": None,
        "logs": [],
        "elapsedMs": 1.0,
    }


class RequestBodyTests(unittest.IsolatedAsyncioTestCase):
    async def test_generated_scan_id_is_a_canonical_uuid(self):
        scan_id = server.new_scan_id()

        parsed = uuid.UUID(scan_id)
        self.assertEqual(str(parsed), scan_id)
        self.assertEqual(parsed.version, 4)

    async def test_raw_image_body_is_preserved_exactly(self):
        body = encoded_image(".jpg")
        request = make_request(body, "image/jpeg")

        self.assertEqual(await server.read_upload_image_bytes(request), body)

    async def test_multipart_image_body_is_preserved_exactly(self):
        image = encoded_image(".png")
        boundary = "ocr-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="image"; filename="card.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode("ascii") + image + f"\r\n--{boundary}--\r\n".encode("ascii")
        request = make_request(body, f"multipart/form-data; boundary={boundary}")

        self.assertEqual(await server.read_upload_image_bytes(request), image)

    async def test_exact_image_byte_limit_is_enforced(self):
        accepted = b"x" * server.MAX_IMAGE_BYTES
        oversized = accepted + b"x"

        self.assertEqual(
            await server.read_upload_image_bytes(
                make_request(accepted, "image/jpeg")
            ),
            accepted,
        )
        with self.assertRaisesRegex(server.HTTPException, "5 MiB") as raised:
            await server.read_upload_image_bytes(
                make_request(oversized, "image/jpeg")
            )
        self.assertEqual(raised.exception.status_code, 413)


class StreamContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_meta_has_scan_id_but_never_an_unconfirmed_key(self):
        store = FakeStore("stored")

        events = await collect_stream(
            encoded_image(".jpg"),
            store,
            successful_region,
        )

        meta = events[0]
        done = events[-1]
        self.assertEqual(meta["type"], "meta")
        self.assertEqual(meta["scanId"], "scan-test")
        self.assertNotIn("trainingImageKey", meta)
        self.assertEqual(meta["image"]["mediaType"], "image/jpeg")
        self.assertEqual(done["type"], "done")
        self.assertTrue(done["success"])
        self.assertEqual(done["scanId"], "scan-test")
        self.assertRegex(done["trainingImageKey"], r"^[a-f0-9]{64}\.jpg$")
        self.assertEqual(done["storage"]["result"], "stored")
        self.assertIn("r2Ms", done["timings"])
        self.assertIn("storageWaitMs", done["timings"])

    async def test_storage_and_recognition_overlap(self):
        store = FakeStore("stored", delay=0.2)
        recognition_times = []

        def timed_region(task):
            started = time.perf_counter()
            result = successful_region(task)
            recognition_times.append((started, time.perf_counter()))
            return result

        events = await collect_stream(
            encoded_image(".jpg"),
            store,
            timed_region,
        )

        self.assertEqual(events[-1]["type"], "done")
        self.assertIsNotNone(store.started_at)
        self.assertIsNotNone(store.finished_at)
        self.assertTrue(recognition_times)
        self.assertLess(min(start for start, _ in recognition_times), store.finished_at)
        self.assertLess(store.started_at, max(end for _, end in recognition_times))

    async def test_r2_failure_does_not_fail_ocr(self):
        events = await collect_stream(
            encoded_image(".png"),
            FakeStore("failed"),
            successful_region,
        )

        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertTrue(done["success"])
        self.assertIsNone(done["trainingImageKey"])
        self.assertEqual(done["storage"]["result"], "failed")
        self.assertEqual(done["image"]["mediaType"], "image/png")

    async def test_r2_timeout_does_not_fail_ocr(self):
        events = await collect_stream(
            encoded_image(".jpg"),
            FakeStore("timed_out"),
            successful_region,
        )

        done = events[-1]
        self.assertTrue(done["success"])
        self.assertIsNone(done["trainingImageKey"])
        self.assertEqual(done["storage"]["result"], "timed_out")

    async def test_wrong_magic_never_starts_storage_or_recognition(self):
        store = FakeStore("stored")
        recognition_calls = []

        def should_not_run(task):
            recognition_calls.append(task)
            return successful_region(task)

        events = await collect_stream(b"not-an-image", store, should_not_run)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertFalse(events[0]["success"])
        self.assertEqual(events[0]["scanId"], "scan-test")
        self.assertEqual(store.calls, 0)
        self.assertEqual(recognition_calls, [])


if __name__ == "__main__":
    unittest.main()
