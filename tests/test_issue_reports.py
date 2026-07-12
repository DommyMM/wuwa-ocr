from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from starlette.requests import Request

import server
import issue_reports
from r2_storage import StorageResult


CONFIRMED_KEY = "a" * 64 + ".jpg"
SCAN_ID = "76078ac4-9ac5-4b52-a933-4fb724f62659"
REGION_KEYS = (
    "character",
    "watermark",
    "forte",
    "sequences",
    "weapon",
    "echo1",
    "echo2",
    "echo3",
    "echo4",
    "echo5",
)


def encoded_image(extension: str = ".jpg") -> bytes:
    image = np.full((64, 96, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise AssertionError(f"could not encode {extension}")
    return encoded.tobytes()


def valid_report(**overrides):
    report = {
        "schemaVersion": 1,
        "route": "/import",
        "reason": "manual_report",
        "scanId": SCAN_ID,
        "trainingImageKey": CONFIRMED_KEY,
        "note": "Echo 3 was read incorrectly.",
        "progress": {key: "done" for key in REGION_KEYS},
        "analysisData": {"echo3": {"main": {"name": "ATK%"}}},
        "importedState": {"characterId": "1409"},
        "validationError": None,
        "ocrError": None,
        "lbUploadError": None,
        "uploadToLb": True,
        "watermark": {"username": "Player", "uid": "500000000"},
        "client": {
            "url": "https://wuwa.build/import",
            "userAgent": "test-agent",
            "submittedAt": "2026-07-11T20:00:00.000Z",
        },
    }
    report.update(overrides)
    return report


def multipart_body(
    report: dict | str,
    image: bytes | None = None,
    extra_fields: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    boundary = "ocr-report-boundary"
    raw_report = report if isinstance(report, str) else json.dumps(report)
    parts = [
        f"--{boundary}\r\n".encode("ascii"),
        b'Content-Disposition: form-data; name="report"\r\n',
        b"Content-Type: application/json; charset=utf-8\r\n\r\n",
        raw_report.encode("utf-8"),
        b"\r\n",
    ]
    for name, value in (extra_fields or {}).items():
        parts.extend([
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
            value.encode("utf-8"),
            b"\r\n",
        ])
    if image is not None:
        parts.extend([
            f"--{boundary}\r\n".encode("ascii"),
            b'Content-Disposition: form-data; name="image"; filename="card.png"\r\n',
            b"Content-Type: image/png\r\n\r\n",
            image,
            b"\r\n",
        ])
    parts.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def make_request(
    body: bytes,
    content_type: str,
    *,
    path: str = "/api/report-ocr-issue",
    extra_headers: dict[str, str] | None = None,
    include_length: bool = True,
) -> Request:
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    headers = [(b"content-type", content_type.encode("ascii"))]
    if include_length:
        headers.append((b"content-length", str(len(body)).encode("ascii")))
    for name, value in (extra_headers or {}).items():
        headers.append((name.lower().encode("ascii"), value.encode("ascii")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers,
        "client": ("203.0.113.10", 12345),
        "server": ("testserver", 80),
    }
    return Request(scope, receive)


class FakeReportStore:
    def __init__(
        self,
        *,
        storage_status: str = "stored",
        put_exception: Exception | None = None,
        enabled: bool = True,
    ):
        self.settings = SimpleNamespace(enabled=enabled)
        self.storage_status = storage_status
        self.put_exception = put_exception
        self.store_calls: list[tuple[bytes, object]] = []
        self.report_calls: list[tuple[str, bytes]] = []

    async def store(self, image_bytes, identity):
        self.store_calls.append((image_bytes, identity))
        return StorageResult(
            result=self.storage_status,
            elapsed_ms=1.0,
            key=(
                identity.key
                if self.storage_status in {"stored", "already_present"}
                else None
            ),
            error_code=("test_failure" if self.storage_status == "failed" else None),
        )

    async def put_json_object(self, key: str, body: bytes) -> None:
        if self.put_exception is not None:
            raise self.put_exception
        self.report_calls.append((key, body))


async def submit(report, *, image: bytes | None = None, store=None):
    body, content_type = multipart_body(report, image)
    active_store = store or FakeReportStore()
    with (
        patch.object(server, "r2_image_store", active_store),
        redirect_stdout(io.StringIO()),
    ):
        response = await server.report_ocr_issue(make_request(body, content_type))
    return response, json.loads(response.body), active_store


class IssueReportContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_key_writes_report_without_uploading_image(self):
        response, receipt, store = await submit(valid_report())

        self.assertEqual(response.status_code, 201)
        self.assertTrue(receipt["success"])
        self.assertEqual(receipt["trainingImageKey"], CONFIRMED_KEY)
        self.assertEqual(receipt["imageStorage"], "referenced")
        self.assertEqual(store.store_calls, [])
        self.assertEqual(len(store.report_calls), 1)
        report_key, report_body = store.report_calls[0]
        self.assertRegex(
            report_key,
            r"^reports/\d{4}/\d{2}/\d{2}/[a-f0-9-]{36}\.json$",
        )
        persisted = json.loads(report_body)
        self.assertEqual(persisted["schemaVersion"], 1)
        self.assertEqual(persisted["scanId"], SCAN_ID)
        self.assertEqual(persisted["trainingImageKey"], CONFIRMED_KEY)
        self.assertEqual(persisted["imageStorage"], "referenced")
        self.assertEqual(persisted["errors"]["ocrError"], None)
        self.assertNotIn("image", persisted)

    async def test_fallback_stores_exact_bytes_and_returns_canonical_key(self):
        image = encoded_image(".png")
        response, receipt, store = await submit(
            valid_report(trainingImageKey=None),
            image=image,
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(receipt["imageStorage"], "stored")
        self.assertRegex(receipt["trainingImageKey"], r"^[a-f0-9]{64}\.png$")
        self.assertEqual(store.store_calls[0][0], image)
        self.assertEqual(store.store_calls[0][1].key, receipt["trainingImageKey"])

    async def test_fallback_deduplicates_an_already_present_image(self):
        response, receipt, _store = await submit(
            valid_report(trainingImageKey=None),
            image=encoded_image(),
            store=FakeReportStore(storage_status="already_present"),
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(receipt["imageStorage"], "already_present")

    async def test_scan_id_may_be_absent_but_must_be_canonical_when_present(self):
        response, receipt, store = await submit(valid_report(scanId=None))

        self.assertEqual(response.status_code, 201)
        persisted = json.loads(store.report_calls[0][1])
        self.assertIsNone(persisted["scanId"])
        self.assertTrue(receipt["success"])

        response, payload, _store = await submit(valid_report(scanId="not-a-uuid"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["reason"], "Invalid issue report metadata.")

    async def test_an_image_source_is_required(self):
        response, payload, _store = await submit(
            valid_report(trainingImageKey=None),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("trainingImageKey or the original image", payload["reason"])

    async def test_a_confirmed_key_wins_over_a_redundant_image(self):
        response, receipt, store = await submit(
            valid_report(),
            image=encoded_image(),
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(receipt["trainingImageKey"], CONFIRMED_KEY)
        self.assertEqual(store.store_calls, [])

    async def test_metadata_rejects_storage_keys_and_closed_enums(self):
        invalid_reports = [
            valid_report(trainingImageKey="not-a-key"),
            valid_report(reason="other"),
            valid_report(route="/other"),
        ]
        for report in invalid_reports:
            with self.subTest(report=report):
                response, payload, _store = await submit(report)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(payload["reason"], "Invalid issue report metadata.")

    async def test_unknown_regions_and_extra_fields_are_kept_not_rejected(self):
        # A report describing an unexpected client state is exactly the report
        # worth keeping, so shape drift must not turn into a 400.
        response, _receipt, store = await submit({
            **valid_report(),
            "progress": {**valid_report()["progress"], "echo6": "unknown"},
            "unexpected": True,
        })

        self.assertEqual(response.status_code, 201)
        persisted = json.loads(store.report_calls[0][1])
        self.assertEqual(persisted["progress"]["echo6"], "unknown")

    async def test_wrong_magic_image_is_rejected(self):
        response, payload, _store = await submit(
            valid_report(trainingImageKey=None),
            image=b"not-an-image",
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn("Unsupported image type", payload["reason"])

    async def test_storage_failures_are_sanitized_and_never_write_dangling_report(self):
        response, payload, store = await submit(
            valid_report(trainingImageKey=None),
            image=encoded_image(),
            store=FakeReportStore(storage_status="failed"),
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            payload["reason"],
            "Fallback image storage is temporarily unavailable.",
        )
        self.assertEqual(store.report_calls, [])

        response, payload, _store = await submit(
            valid_report(),
            store=FakeReportStore(put_exception=RuntimeError("secret detail")),
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            payload,
            {
                "success": False,
                "reason": "Issue report storage is temporarily unavailable.",
            },
        )
        self.assertNotIn("secret", json.dumps(payload))

    async def test_disabled_storage_fails_closed(self):
        response, payload, store = await submit(
            valid_report(),
            store=FakeReportStore(enabled=False),
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["reason"], "Issue report storage is not configured.")
        self.assertEqual(store.report_calls, [])


class IssueReportParsingTests(unittest.IsolatedAsyncioTestCase):
    async def test_content_type_and_declared_envelope_are_bounded(self):
        response = await server.report_ocr_issue(
            make_request(b"{}", "application/json")
        )
        self.assertEqual(response.status_code, 415)

        body, content_type = multipart_body(valid_report())
        request = make_request(
            body,
            content_type,
            extra_headers={
                "content-length": str(issue_reports.MAX_REPORT_MULTIPART_BYTES + 1),
            },
            include_length=False,
        )
        response = await server.report_ocr_issue(request)
        self.assertEqual(response.status_code, 413)

    async def test_metadata_limit_and_nonstandard_json_numbers_are_rejected(self):
        oversized = "x" * (issue_reports.MAX_REPORT_JSON_BYTES + 1)
        with self.assertRaises(issue_reports.OcrIssueRequestError) as raised:
            issue_reports.parse_issue_report_json(oversized)
        self.assertEqual(raised.exception.status_code, 413)

        raw = json.dumps(valid_report()).replace(
            '"analysisData": {',
            '"analysisData": {"invalid": NaN,',
        )
        with self.assertRaises(issue_reports.OcrIssueRequestError) as raised:
            issue_reports.parse_issue_report_json(raw)
        self.assertEqual(raised.exception.status_code, 400)

    async def test_image_reader_accepts_exactly_five_mib_and_rejects_more(self):
        class Upload:
            def __init__(self, body: bytes):
                self.body = body

            async def read(self):
                return self.body

        exact = Upload(b"x" * issue_reports.MAX_IMAGE_BYTES)
        self.assertEqual(
            len(await issue_reports.read_report_image(exact)),
            issue_reports.MAX_IMAGE_BYTES,
        )

        oversized = Upload(b"x" * (issue_reports.MAX_IMAGE_BYTES + 1))
        with self.assertRaises(issue_reports.OcrIssueRequestError) as raised:
            await issue_reports.read_report_image(oversized)
        self.assertEqual(raised.exception.status_code, 413)


class IssueReportAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_key_is_required_when_configured(self):
        request = make_request(b"", "multipart/form-data; boundary=x")
        called = False

        async def call_next(_request):
            nonlocal called
            called = True
            return server.JSONResponse({"ok": True})

        with patch.object(server, "INTERNAL_API_KEY", "trusted-secret"):
            response = await server.rate_limit_middleware(request, call_next)

        self.assertEqual(response.status_code, 403)
        self.assertFalse(called)
        self.assertEqual(json.loads(response.body)["success"], False)

    async def test_report_limiter_is_independent_and_uses_forwarded_ip(self):
        limiter = server.RateLimiter(1)

        async def call_next(_request):
            return server.JSONResponse({"ok": True})

        def trusted_request():
            return make_request(
                b"",
                "multipart/form-data; boundary=x",
                extra_headers={
                    "x-internal-key": "trusted-secret",
                    "x-ocr-client-ip": "198.51.100.42",
                },
            )

        with (
            patch.object(server, "INTERNAL_API_KEY", "trusted-secret"),
            patch.object(server, "report_rate_limiter", limiter),
        ):
            first = await server.rate_limit_middleware(trusted_request(), call_next)
            second = await server.rate_limit_middleware(trusted_request(), call_next)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.headers["retry-after"], "60")
        self.assertIn("198.51.100.42", limiter.requests)


if __name__ == "__main__":
    unittest.main()
