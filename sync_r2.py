"""
sync_r2.py — download all images from R2 bucket into local r2-backup/.

Only fetches files not already present locally (skips existing).
Run this before benchmark to ensure r2-backup is up to date.

Usage:
  py sync_r2.py             # dry run — shows what would be downloaded
  py sync_r2.py --run       # actually downloads and stamps file mtimes
  py sync_r2.py --run --manifest  # also write compact r2-backup-manifest.json
  py sync_r2.py --run --recheck-legacy  # also re-verify uncached canonical
                                         # keys against R2 instead of assuming
                                         # they have no legacy metadata
"""
import sys
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
R2_BACKUP   = BACKEND_DIR.parent / "r2-backup"
MANIFEST    = BACKEND_DIR.parent / "r2-backup-manifest.json"
# Owned entirely by this script (unlike the old migration manifest): canonical
# objects are content-addressed and never rewritten after creation, so once a
# key's been checked its result never goes stale. Losing this file just means
# a full re-sweep next run, never a wrong answer.
LEGACY_MTIME_CACHE = BACKEND_DIR.parent / "r2-legacy-mtime-cache.json"
DRY_RUN        = "--run" not in sys.argv
WRITE_MANIFEST = "--manifest" in sys.argv
RECHECK_LEGACY = "--recheck-legacy" in sys.argv
WORKERS        = 128

# R2 can't have its own LastModified overridden, but the source-image migration
# stamps this custom metadata field on every canonical object it creates so the
# original screenshot time survives the copy. Reading it straight off the
# object (instead of the migration manifest file) means local mtime accuracy
# doesn't depend on that file still existing on disk.
CANONICAL_KEY_RE = re.compile(r"^[a-f0-9]{64}\.(?:jpg|png)$")

ENV_CANDIDATES = [
    BACKEND_DIR.parent / "wuwabuilds" / ".env",
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


def local_keys() -> set[str]:
    return {
        str(path.relative_to(R2_BACKUP)).replace("\\", "/")
        for path in R2_BACKUP.rglob("*")
        if path.is_file()
    }


def load_legacy_mtime_cache() -> dict[str, str | None]:
    if not LEGACY_MTIME_CACHE.exists():
        return {}
    try:
        return json.loads(LEGACY_MTIME_CACHE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_legacy_mtime_cache(cache: dict[str, str | None]) -> None:
    LEGACY_MTIME_CACHE.write_text(
        json.dumps(cache, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )


def fetch_legacy_mtimes(s3, bucket: str, keys: list[str], recheck: bool = False) -> dict[str, float]:
    """Read each canonical-keyed object's own `legacy-last-modified` metadata
    via HeadObject. Legacy 16-hex-named objects were never given this field
    (they're untouched originals, their own LastModified is already correct),
    so only canonical-shaped keys are worth checking.

    Results are cached locally by key (`None` means confirmed-absent, a real
    marker not just a missing entry) since the answer can never change once
    checked. The 16->64 hex migration that populates this field ran once and
    is closed, so any canonical key not already in the cache is, by
    construction, a normal upload through r2_storage.py — which never sets
    legacy-last-modified. By default that's trusted instead of paying a
    network round trip to confirm it; pass recheck=True (--recheck-legacy) to
    verify anyway, e.g. after another migration-style copy job runs.
    """
    canonical_keys = [k for k in keys if CANONICAL_KEY_RE.match(k)]
    if not canonical_keys:
        return {}

    cache = load_legacy_mtime_cache()
    uncached = [k for k in canonical_keys if k not in cache]
    to_check = uncached if recheck else []

    def head(key: str):
        try:
            resp = s3.head_object(Bucket=bucket, Key=key)
            return key, resp.get("Metadata", {}).get("legacy-last-modified") or None, True
        except Exception:
            return key, None, False

    if to_check:
        print(
            f"Checking {len(to_check)} new canonical objects for original timestamps "
            f"({len(canonical_keys) - len(to_check)} already cached) ..."
        )
        failed = []
        surprises = []
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for key, raw, ok in pool.map(head, to_check):
                if ok:
                    cache[key] = raw
                    if raw:
                        surprises.append(key)
                else:
                    failed.append(key)
        save_legacy_mtime_cache(cache)
        if failed:
            print(f"  WARNING: {len(failed)} HEAD requests failed, will retry next run (e.g. {failed[0]})")
        if surprises:
            print(
                f"  NOTICE: {len(surprises)} objects assumed to have no legacy metadata actually "
                f"had it (e.g. {surprises[0]}) — the migration may not be fully closed"
            )
    elif uncached:
        print(
            f"{len(uncached)} canonical objects not legacy-checked — assuming no legacy metadata "
            f"since the migration is closed (pass --recheck-legacy to verify)"
        )
    else:
        print(f"All {len(canonical_keys)} canonical objects already cached, skipping sweep")

    result: dict[str, float] = {}
    for key in canonical_keys:
        raw = cache.get(key)
        if not raw:
            continue
        try:
            result[key] = datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue

    if result:
        print(f"Recovered original timestamps for {len(result)} migrated images")
    return result


def main():
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    R2_BACKUP.mkdir(exist_ok=True)
    env_file   = find_env_file()
    env        = load_env(env_file)
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

    # List all objects in bucket (paginated)
    print(f"Listing objects in R2 bucket '{bucket}' ...")
    paginator = s3.get_paginator("list_objects_v2")
    all_objects = []
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            all_objects.append(obj)

    all_keys = [obj["Key"] for obj in all_objects]
    print(f"Found {len(all_keys)} objects in R2")

    manifest = {
        obj["Key"]: {
            "size": obj.get("Size"),
            "etag": str(obj.get("ETag", "")).strip('"'),
            "last_modified": obj["LastModified"].isoformat(),
        }
        for obj in all_objects
    }
    by_key = {obj["Key"]: obj for obj in all_objects}

    existing_keys = local_keys()
    to_download = [k for k in all_keys if k not in existing_keys]

    print(f"Local r2-backup: {len(existing_keys)} files")
    print(f"To download:     {len(to_download)} new files")

    if DRY_RUN:
        print(f"\n[DRY RUN] Would download {len(to_download)} files. Re-run with --run to apply.")
        for k in to_download[:20]:
            print(f"  {k}")
        if len(to_download) > 20:
            print(f"  ... and {len(to_download) - 20} more")
        return

    if WRITE_MANIFEST:
        MANIFEST.write_text(json.dumps(manifest, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        print(f"Wrote manifest: {MANIFEST}")

    migration_mtimes = fetch_legacy_mtimes(s3, bucket, all_keys, recheck=RECHECK_LEGACY)

    def effective_mtime(key: str) -> float:
        return migration_mtimes.get(key, by_key[key]["LastModified"].timestamp())

    # Keep local mtimes aligned to true original time so date-based backfills can
    # select screenshots by patch window without relisting R2.
    touched_existing = 0
    for key in existing_keys:
        if key not in by_key:
            continue
        mtime = effective_mtime(key)
        os.utime(R2_BACKUP / key, (mtime, mtime))
        touched_existing += 1
    print(f"Updated mtimes for {touched_existing} existing files")

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
            last_modified = effective_mtime(key)
            os.utime(destination, (last_modified, last_modified))
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

    print(f"\n{'─'*50}")
    print(f"  Downloaded: {downloaded}  Failed: {failed}")
    print(f"  r2-backup now has {len(local_keys())} files")


if __name__ == "__main__":
    main()
