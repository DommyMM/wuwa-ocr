"""
sync_r2.py — download all images from R2 bucket into local r2-backup/.

Only fetches files not already present locally (skips existing).
Run this before benchmark to ensure r2-backup is up to date.

Usage:
  py sync_r2.py          # dry run — shows what would be downloaded
  py sync_r2.py --run    # actually downloads
"""
import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
R2_BACKUP   = BACKEND_DIR.parent / "r2-backup"
MANIFEST    = BACKEND_DIR.parent / "r2-backup-manifest.json"
DRY_RUN     = "--run" not in sys.argv

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


def main():
    try:
        import boto3
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

    if not DRY_RUN:
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote manifest: {MANIFEST}")

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

    # Keep local mtimes aligned to R2 upload time so date-based backfills can
    # select screenshots by patch window without relisting R2.
    touched_existing = 0
    for key in existing_keys:
        meta = manifest.get(key)
        if not meta:
            continue
        path = R2_BACKUP / key
        last_modified = by_key[key]["LastModified"].timestamp()
        os.utime(path, (last_modified, last_modified))
        touched_existing += 1
    print(f"Updated mtimes for {touched_existing} existing files from R2 metadata")

    if not to_download:
        print("Already in sync.")
        return

    WORKERS = 32
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
            last_modified = by_key[key]["LastModified"].timestamp()
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
