"""Local browser UI for reviewing scan_image_integrity.py's review_queue.json

Decisions save to review_decisions.json next to the queue
Delete decisions export to backend/invalid_images.json for clean_invalid.py

Usage:
  py review_integrity_gui.py ../forensics/integrity_scan/review_queue.json ../r2-backup
  py review_integrity_gui.py ../forensics/integrity_scan/review_queue.json ../r2-backup --verdict suspect
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import shutil
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).parent
DEFAULT_INVALID_FILE = BACKEND_DIR / "invalid_images.json"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_invalid(decisions: dict[str, Any], invalid_file: Path) -> None:
    invalid = sorted(
        key
        for key, value in decisions.items()
        if isinstance(value, dict) and value.get("decision") == "delete"
    )
    write_json(invalid_file, invalid)


def filter_items(items: list[dict[str, Any]], verdicts: set[str]) -> list[dict[str, Any]]:
    if not verdicts:
        return items
    return [item for item in items if str(item.get("verdict")) in verdicts]


def pct(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return ""


def metric(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return ""


class ReviewApp:
    def __init__(self, queue: Path, image_dir: Path, decisions_path: Path, invalid_file: Path, archive_dir: Path, verdicts: set[str]):
        self.queue = queue
        self.image_dir = image_dir
        self.decisions_path = decisions_path
        self.invalid_file = invalid_file
        self.archive_dir = archive_dir
        raw_items = load_json(queue, [])
        if not isinstance(raw_items, list):
            raise ValueError(f"{queue} is not a review queue list")
        self.items = filter_items(raw_items, verdicts)
        self.decisions = load_json(decisions_path, {})
        if not isinstance(self.decisions, dict):
            self.decisions = {}
        export_invalid(self.decisions, self.invalid_file)

    def archive_image(self, key: str, decision: str) -> None:
        if decision not in {"delete", "review"}:
            return
        src = self.image_dir / key
        if not src.exists():
            return
        dst = self.archive_dir / decision / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    def save_decision(self, key: str, decision: str) -> None:
        item = self.item_by_key(key)
        self.decisions[key] = {
            "decision": decision,
            "verdict": item.get("verdict") if item else "",
            "reasons": item.get("reasons", []) if item else [],
        }
        self.archive_image(key, decision)
        write_json(self.decisions_path, self.decisions)
        export_invalid(self.decisions, self.invalid_file)

    def item_by_key(self, key: str) -> dict[str, Any] | None:
        for item in self.items:
            if item.get("key") == key:
                return item
        return None

    def counts(self) -> dict[str, int]:
        out = {"total": len(self.items), "keep": 0, "delete": 0, "review": 0, "undecided": 0}
        for item in self.items:
            key = str(item.get("key", ""))
            decision = self.decisions.get(key, {}).get("decision") if isinstance(self.decisions.get(key), dict) else None
            if decision in {"keep", "delete", "review"}:
                out[decision] += 1
            else:
                out["undecided"] += 1
        return out

    def next_index(self, current: int) -> int:
        if not self.items:
            return 0
        for idx in range(current, len(self.items)):
            key = str(self.items[idx].get("key", ""))
            if key not in self.decisions:
                return idx
        return min(current, len(self.items) - 1)


def make_handler(app: ReviewApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def send_bytes(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, data: Any, status: int = 200) -> None:
            self.send_bytes(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                query = urllib.parse.parse_qs(parsed.query)
                idx = int(query.get("i", ["0"])[0] or 0)
                self.render_page(idx)
                return
            if parsed.path == "/image":
                query = urllib.parse.parse_qs(parsed.query)
                key = query.get("key", [""])[0]
                path = (app.image_dir / key).resolve()
                if not str(path).lower().startswith(str(app.image_dir.resolve()).lower()) or not path.exists():
                    self.send_bytes(404, b"not found", "text/plain")
                    return
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self.send_bytes(200, path.read_bytes(), content_type)
                return
            if parsed.path == "/api/status":
                self.send_json(app.counts())
                return
            self.send_bytes(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/api/decision":
                self.send_bytes(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length)
            try:
                payload = json.loads(body.decode("utf-8"))
                key = str(payload["key"])
                decision = str(payload["decision"])
            except Exception:
                self.send_json({"ok": False, "error": "invalid payload"}, 400)
                return
            if decision not in {"keep", "delete", "review"}:
                self.send_json({"ok": False, "error": "invalid decision"}, 400)
                return
            app.save_decision(key, decision)
            self.send_json({"ok": True, "counts": app.counts()})

        def render_page(self, idx: int) -> None:
            total = len(app.items)
            if total == 0:
                self.send_bytes(200, b"<h1>No review items</h1>", "text/html; charset=utf-8")
                return
            idx = max(0, min(idx, total - 1))
            item = app.items[idx]
            key = str(item.get("key", ""))
            image = item.get("image") if isinstance(item.get("image"), dict) else {}
            decision = app.decisions.get(key, {}).get("decision") if isinstance(app.decisions.get(key), dict) else ""
            counts = app.counts()
            panels = item.get("panels") if isinstance(item.get("panels"), dict) else {}

            panel_rows = []
            for panel in ("echo3", "echo4", "echo5"):
                data = panels.get(panel) if isinstance(panels.get(panel), dict) else {}
                hit = "yes" if data.get("hit") else "no"
                panel_rows.append(
                    f"<tr><td>{panel}</td><td>{metric(data.get('darkAvg'))}</td>"
                    f"<td>{pct(data.get('darkPercentile'))}</td><td>{metric(data.get('runAvg'))}</td>"
                    f"<td>{pct(data.get('runPercentile'))}</td><td>{hit}</td></tr>"
                )

            prev_idx = max(0, idx - 1)
            next_idx = min(total - 1, idx + 1)
            next_undecided = app.next_index(idx + 1)
            escaped_key = html.escape(key)
            reasons = ", ".join(str(r) for r in item.get("reasons", []))

            page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Integrity Review</title>
  <style>
    body {{ margin: 0; font-family: system-ui, Segoe UI, sans-serif; background: #141217; color: #eee; }}
    .wrap {{ display: grid; grid-template-columns: minmax(0, 1fr) 390px; height: 100vh; }}
    .imagePane {{ display: flex; align-items: center; justify-content: center; background: #09080b; overflow: auto; }}
    img {{ max-width: 100%; max-height: 100vh; object-fit: contain; }}
    aside {{ padding: 18px; border-left: 1px solid #333; overflow: auto; }}
    h1 {{ font-size: 18px; margin: 0 0 12px; word-break: break-all; }}
    .meta, table {{ color: #cfcbd6; font-size: 13px; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #2b2631; margin-right: 6px; }}
    .decision {{ color: #fff; background: #51415e; }}
    button, a.nav {{ border: 0; color: #fff; padding: 10px 12px; margin: 5px 4px 5px 0; border-radius: 6px; cursor: pointer; text-decoration: none; display: inline-block; font-weight: 650; }}
    .keep {{ background: #237a4b; }}
    .delete {{ background: #a83333; }}
    .review {{ background: #9a6b1e; }}
    .nav {{ background: #35303d; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 14px; }}
    td, th {{ border-bottom: 1px solid #333; padding: 7px 5px; text-align: right; }}
    td:first-child, th:first-child {{ text-align: left; }}
    code {{ color: #e7d39a; }}
    .small {{ color: #aaa; font-size: 12px; margin-top: 16px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <main class="imagePane">
      <img src="/image?key={urllib.parse.quote(key)}" alt="{escaped_key}" />
    </main>
    <aside>
      <h1>{escaped_key}</h1>
      <div>
        <span class="pill">{idx + 1} / {total}</span>
        <span class="pill">{html.escape(str(item.get("verdict", "")))}</span>
        <span class="pill decision">decision: <span id="decision">{html.escape(str(decision or "undecided"))}</span></span>
      </div>
      <p class="meta">Reasons: <code>{html.escape(reasons)}</code></p>
      <p class="meta">
        {image.get("width")}x{image.get("height")} |
        aspect {float(image.get("aspect", 0)):.4f} |
        {image.get("bytes")} bytes |
        {html.escape(str(image.get("extension", "")))}
      </p>
      <div>
        <button class="keep" onclick="decide('keep')">Keep (A)</button>
        <button class="review" onclick="decide('review')">Review Later (S)</button>
        <button class="delete" onclick="decide('delete')">Delete (D)</button>
      </div>
      <div>
        <a class="nav" href="/?i={prev_idx}">Prev</a>
        <a class="nav" href="/?i={next_idx}">Next</a>
        <a class="nav" href="/?i={next_undecided}">Next Undecided</a>
      </div>
      <table>
        <thead><tr><th>Panel</th><th>Dark</th><th>Dark %</th><th>Run</th><th>Run %</th><th>Hit</th></tr></thead>
        <tbody>{''.join(panel_rows)}</tbody>
      </table>
      <p class="small" id="counts">
        total {counts['total']} | keep {counts['keep']} | delete {counts['delete']} |
        review {counts['review']} | undecided {counts['undecided']}
      </p>
      <p class="small">Keyboard: A keep, S review, D delete, ←/→ navigate. Marking auto-advances.</p>
    </aside>
  </div>
  <script>
    const key = {json.dumps(key)};
    async function decide(decision) {{
      const res = await fetch('/api/decision', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{key, decision}})
      }});
      const data = await res.json();
      if (!data.ok) {{
        alert(data.error || 'failed');
        return;
      }}
      document.getElementById('decision').textContent = decision;
      const c = data.counts;
      document.getElementById('counts').textContent =
        `total ${{c.total}} | keep ${{c.keep}} | delete ${{c.delete}} | review ${{c.review}} | undecided ${{c.undecided}}`;
      setTimeout(() => location.href = '/?i={next_undecided}', 120);
    }}
    document.addEventListener('keydown', (event) => {{
      if (event.key === 'a' || event.key === 'A') decide('keep');
      if (event.key === 'd' || event.key === 'D') decide('delete');
      if (event.key === 's' || event.key === 'S') decide('review');
      if (event.key === 'ArrowLeft') location.href = '/?i={prev_idx}';
      if (event.key === 'ArrowRight') location.href = '/?i={next_idx}';
    }});
  </script>
</body>
</html>"""
            self.send_bytes(200, page.encode("utf-8"), "text/html; charset=utf-8")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--invalid-file", type=Path, default=DEFAULT_INVALID_FILE)
    parser.add_argument("--archive-dir", type=Path, help="where delete/review-marked local copies are archived; defaults next to queue")
    parser.add_argument("--verdict", action="append", choices=["review", "suspect", "reject"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    decisions_path = args.decisions or (args.queue.parent / "review_decisions.json")
    archive_dir = args.archive_dir or (args.queue.parent / "marked_images")
    app = ReviewApp(args.queue, args.image_dir, decisions_path, args.invalid_file, archive_dir, set(args.verdict or []))
    handler = make_handler(app)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"

    print(f"Reviewing {len(app.items)} item(s)")
    print(f"URL: {url}")
    print(f"Decisions: {decisions_path}")
    print(f"Invalid export: {args.invalid_file}")
    print(f"Marked-image archive: {archive_dir}")
    print("Press Ctrl+C to stop.")

    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
