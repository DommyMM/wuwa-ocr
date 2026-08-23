"""
sync_r2.py — mirror the R2 bucket into local r2-backup/ with true original mtimes.

Downloads only keys missing locally, and stamps each local file with the
original upload time so date-based backfills can select screenshots by patch
window without relisting R2.

Usage:
  py sync_r2.py                         # dry run — reports downloads and mtime drift
  py sync_r2.py --run                   # download missing keys, stamp drifted mtimes
  py sync_r2.py --run --recheck-legacy  # re-close the original-time table after
                                        # another migration-style copy job
"""
import sys
import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
WORKSPACE   = BACKEND_DIR.parent
R2_BACKUP   = WORKSPACE / "r2-backup"

# Frozen table of original upload times for the objects the 2026-07-11
# source-image migration copied. That copy reset each object's R2 LastModified
# to the copy date, so the mirror phase stamped the original time into
# `legacy-last-modified` metadata; this file is the local read-through of it.
#
# `coveredThrough` is what makes the table closed rather than open-ended: it
# records the newest LastModified the table accounts for. Any object newer than
# that postdates the migration, so it provably carries no legacy metadata and
# its own LastModified is already the original upload time. That replaces the
# old per-run "assuming no legacy metadata" guess with a checked invariant, and
# keeps a normal sync at zero HEAD requests no matter how much the bucket grows.
# A canonical key at or below the boundary that is missing from the table is
# therefore an anomaly, not a routine cache miss.
# Losing this file costs a one-off full sweep, never a wrong answer: the
# objects themselves remain the source of truth for every value in it.
ORIGINAL_MTIMES = WORKSPACE / "r2-original-mtimes.json"

DRY_RUN        = "--run" not in sys.argv
RECHECK_LEGACY = "--recheck-legacy" in sys.argv
WORKERS        = 128
# utime writes a float that NTFS rounds, so re-reading never matches exactly.
# A second of slack keeps steady-state runs at zero writes.
MTIME_TOLERANCE = 1.0

# Only the migration ever wrote 64-hex keys, so nothing else can carry the
# metadata. Everything else (db-backups/, reports/, a few date-named strays) is
# trusted at its own LastModified.
CANONICAL_KEY_RE = re.compile(r"^[a-f0-9]{64}\.(?:jpg|png)$")

ENV_CANDIDATES = [
    WORKSPACE / "wuwabuilds" / ".env",
    BACKEND_DIR / ".env",
]


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def find_env_file() -> Path:
    for path in ENV_CANDIDATES:
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in ENV_CANDIDATES)
    raise FileNotFoundError(f"No .env file found. Checked: {checked}")


def parse_iso(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def walk_local() -> dict[str, float]:
    """Key -> current mtime for every file under r2-backup. os.scandir hands back
    the mtime from the directory entry itself, so this costs one walk instead of
    a walk plus a stat call per file."""
    found: dict[str, float] = {}
    stack = [R2_BACKUP]
    while stack:
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    key = Path(entry.path).relative_to(R2_BACKUP).as_posix()
                    found[key] = entry.stat().st_mtime
    return found


def load_table() -> tuple[dict[str, str | None], str | None]:
    """Returns (key -> original ISO time or None, coveredThrough or None).

    A None value is load-bearing: it records a canonical object inside the
    covered window that was checked and genuinely has no legacy metadata, which
    is what keeps it from tripping the anomaly branch on every run.

    An unreadable or absent table is not an error. Every value in it is
    re-derivable from the objects' own metadata, so the caller just falls back
    to a full sweep."""
    if ORIGINAL_MTIMES.exists():
        try:
            raw = json.loads(ORIGINAL_MTIMES.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "times" in raw:
                return raw["times"], raw.get("coveredThrough")
        except (json.JSONDecodeError, OSError):
            pass
    return {}, None


def save_table(times: dict[str, str | None], covered_through: str | None) -> None:
    ORIGINAL_MTIMES.write_text(
        json.dumps(
            {"version": 2, "coveredThrough": covered_through, "times": times},
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def resolve_original_mtimes(s3, bucket: str, objects: list, recheck: bool, dry_run: bool) -> dict[str, float]:
    """Key -> original upload timestamp, for the keys where it differs from
    LastModified. Resolution order per key:

      1. in the table            -> its recorded time (a None entry falls
                                    through to LastModified, already correct)
      2. not canonical           -> LastModified
      3. newer than the boundary -> LastModified, with no network call
      4. anything left           -> a canonical key inside the covered window
                                    that the table misses; HEAD it and say so
    """
    times, covered_through = load_table()
    by_key = {obj["Key"]: obj for obj in objects}
    canonical = [obj for obj in objects if CANONICAL_KEY_RE.match(obj["Key"])]
    boundary = parse_iso(covered_through) if covered_through else None
    to_head = [
        obj["Key"] for obj in canonical
        if obj["Key"] not in times
        and (recheck or boundary is None or obj["LastModified"] <= boundary)
    ]
    anomalous = bool(to_head) and boundary is not None and not recheck

    if to_head and dry_run:
        print(f"[DRY RUN] would HEAD {len(to_head)} canonical objects")
        to_head = []

    if to_head:
        if anomalous:
            print(
                f"  ANOMALY: {len(to_head)} canonical objects sit at or below the coverage "
                f"boundary {covered_through} yet are missing from the table (e.g. {to_head[0]}). "
                f"Something wrote a backdated object; checking them directly."
            )
        else:
            print(f"Checking {len(to_head)} canonical objects for original timestamps ...")

        def head(key: str):
            try:
                resp = s3.head_object(Bucket=bucket, Key=key)
                return key, resp.get("Metadata", {}).get("legacy-last-modified") or None, True
            except Exception:
                return key, None, False

        failed = []
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for key, raw, ok in pool.map(head, to_head):
                if ok:
                    times[key] = raw
                else:
                    failed.append(key)
        if failed:
            print(f"  WARNING: {len(failed)} HEAD requests failed, will retry next run (e.g. {failed[0]})")
        # A full sweep re-closes the table, so the boundary can advance to the
        # newest object it now accounts for. An anomaly patch fills a hole
        # underneath the existing boundary and leaves it where it was.
        if (recheck or boundary is None) and canonical:
            covered_through = max(obj["LastModified"] for obj in canonical).isoformat()

    # The table only ever changes when a HEAD actually resolved something, so a
    # steady-state run leaves it untouched on disk.
    if to_head and not dry_run:
        save_table(times, covered_through)

    resolved: dict[str, float] = {}
    for key, raw in times.items():
        if not raw:
            continue
        try:
            resolved[key] = parse_iso(raw).timestamp()
        except ValueError:
            continue

    trusted = len(by_key) - sum(1 for key in resolved if key in by_key)
    print(
        f"Original mtimes: {len(resolved)} from table | {trusted} from R2 LastModified "
        f"| {len(to_head)} HEAD requests (boundary {covered_through})"
    )
    return resolved


def main():
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    R2_BACKUP.mkdir(exist_ok=True)
    env        = load_env(find_env_file())
    account_id = env["CLOUDFLARE_ACCOUNT_ID"]
    bucket     = env.get("R2_BUCKET_NAME", "wuwabuilds")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(max_pool_connections=WORKERS),
    )

    print(f"Listing objects in R2 bucket '{bucket}' ...")
    started = time.perf_counter()
    all_objects = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        all_objects.extend(page.get("Contents", []))
    by_key = {obj["Key"]: obj for obj in all_objects}
    print(f"Found {len(by_key)} objects ({time.perf_counter() - started:.1f}s)")

    local = walk_local()
    to_download = [key for key in by_key if key not in local]
    print(f"Local r2-backup: {len(local)} files | to download: {len(to_download)}")

    original = resolve_original_mtimes(s3, bucket, all_objects, RECHECK_LEGACY, DRY_RUN)

    def target_mtime(key: str) -> float:
        return original.get(key, by_key[key]["LastModified"].timestamp())

    # Only files whose stamp actually drifted get rewritten. In steady state
    # that is zero, which keeps the sync from dirtying 26k files' metadata on
    # every run just to write back the values they already hold.
    drifted = [
        key for key, mtime in local.items()
        if key in by_key and abs(mtime - target_mtime(key)) > MTIME_TOLERANCE
    ]

    if DRY_RUN:
        print(f"\n[DRY RUN] {len(to_download)} to download, {len(drifted)} mtimes to correct.")
        for key in to_download[:20]:
            print(f"  + {key}")
        if len(to_download) > 20:
            print(f"  ... and {len(to_download) - 20} more")
        print("Re-run with --run to apply.")
        return

    for key in drifted:
        stamp = target_mtime(key)
        os.utime(R2_BACKUP / key, (stamp, stamp))
    print(f"Stamped {len(drifted)} drifted mtimes ({len(local) - len(drifted)} already correct)")

    if not to_download:
        print("Already in sync.")
        return

    print(f"\nDownloading with {WORKERS} parallel workers ...\n")
    downloaded = 0
    failed     = 0
    total      = len(to_download)

    def fetch(idx_key):
        idx, key = idx_key
        try:
            destination = R2_BACKUP / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(destination))
            stamp = target_mtime(key)
            os.utime(destination, (stamp, stamp))
            return idx, key, None
        except Exception as e:
            return idx, key, str(e)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch, (i, k)): k for i, k in enumerate(to_download, 1)}
        for fut in as_completed(futures):
            idx, key, err = fut.result()
            if err:
                print(f"  [{idx}/{total}] FAILED {key}: {err}")
                failed += 1
            else:
                print(f"  [{idx}/{total}] {key}")
                downloaded += 1

    print(f"\n{'-' * 50}")
    print(f"  Downloaded: {downloaded}  Failed: {failed}")


if __name__ == "__main__":
    main()
