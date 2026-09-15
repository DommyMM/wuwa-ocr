"""
Delete images marked invalid during benchmark review

Reads:  backend/invalid_images.json   (built by visualize_diffs.py)
Reads:  wuwabuilds/.env               (R2 credentials)

Actions (dry-run by default):
  1. Deletes each image from Cloudflare R2 bucket
  2. Archives then deletes each image from local r2-backup/

Usage:
  py clean_invalid.py          # dry run — shows what would be deleted
  py clean_invalid.py --run    # actually deletes
  py clean_invalid.py --run --r2-only  # delete from R2 but keep local files
  py clean_invalid.py --run --no-archive  # delete local files without archiving them
"""
import sys
import json
import shutil
from pathlib import Path

BACKEND_DIR  = Path(__file__).parent
INVALID_FILE = BACKEND_DIR / "invalid_images.json"
R2_BACKUP    = BACKEND_DIR.parent / "r2-backup"
ENV_FILE     = BACKEND_DIR.parent / "wuwabuilds" / ".env"
ARCHIVE_DIR  = BACKEND_DIR.parent / "forensics" / "deleted_from_r2"

DRY_RUN = "--run" not in sys.argv
R2_ONLY = "--r2-only" in sys.argv
NO_ARCHIVE = "--no-archive" in sys.argv


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main():
    if not INVALID_FILE.exists():
        print("No invalid_images.json found. Run visualize_diffs.py first.")
        sys.exit(0)

    with open(INVALID_FILE) as f:
        images = json.load(f)

    if not images:
        print("invalid_images.json is empty — nothing to delete.")
        sys.exit(0)

    print(f"{'[DRY RUN] ' if DRY_RUN else ''}Deleting {len(images)} invalid images\n")

    env = load_env(ENV_FILE)
    account_id = env.get("CLOUDFLARE_ACCOUNT_ID", "")
    bucket     = env.get("R2_BUCKET_NAME", "wuwabuilds")
    access_key = env.get("R2_ACCESS_KEY_ID", "")
    secret_key = env.get("R2_SECRET_ACCESS_KEY", "")

    if not all([account_id, access_key, secret_key]):
        print("ERROR: Missing R2 credentials in wuwabuilds/.env")
        sys.exit(1)

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    r2_deleted = 0
    r2_missing = 0
    for name in images:
        if DRY_RUN:
            print(f"  [R2]  would delete  {bucket}/{name}")
            r2_deleted += 1
        else:
            try:
                s3.delete_object(Bucket=bucket, Key=name)
                print(f"  [R2]  deleted  {name}")
                r2_deleted += 1
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in ("NoSuchKey", "404"):
                    print(f"  [R2]  not found (skip)  {name}")
                    r2_missing += 1
                else:
                    print(f"  [R2]  ERROR {code}: {name}")

    local_deleted = 0
    local_missing = 0
    local_archived = 0
    for name in images:
        p = R2_BACKUP / name
        archive_path = ARCHIVE_DIR / name
        if p.exists():
            if R2_ONLY:
                print(f"  [local] keeping local copy  {p}")
            elif DRY_RUN:
                action = "delete"
                if not NO_ARCHIVE:
                    action = f"archive to {archive_path} then delete"
                print(f"  [local] would {action}  {p}")
                local_deleted += 1
            else:
                if not NO_ARCHIVE:
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, archive_path)
                    print(f"  [local] archived {name} -> {archive_path}")
                    local_archived += 1
                p.unlink()
                print(f"  [local] deleted  {name}")
                local_deleted += 1
        else:
            print(f"  [local] not found (skip)  {name}")
            local_missing += 1

    print(f"\n{'─'*50}")
    if DRY_RUN:
        print(f"  Dry run complete. Re-run with --run to apply.")
        if R2_ONLY:
            print(f"  Would delete: {r2_deleted} from R2, 0 from local r2-backup/ (--r2-only)")
        else:
            print(f"  Would delete: {r2_deleted} from R2, {local_deleted} from local r2-backup/")
            if not NO_ARCHIVE:
                print(f"  Would archive local copies to: {ARCHIVE_DIR}")
    else:
        print(f"  R2:    deleted={r2_deleted}  not_found={r2_missing}")
        print(f"  Local: archived={local_archived}  deleted={local_deleted}  not_found={local_missing}")
        if R2_ONLY:
            print(f"  Local files were kept because --r2-only was set.")

        # Cleared after any --run, even when some R2 deletes errored
        with open(INVALID_FILE, "w") as f:
            json.dump([], f)
        print(f"  Cleared invalid_images.json")


if __name__ == "__main__":
    main()
