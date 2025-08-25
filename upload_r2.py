import os
import sys
import mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm


# ------------------ Settings ------------------
WORKERS = 16  # parallel uploads; reduce if your network is weak
LOCAL_JSON_DIR = Path(__file__).resolve().parent / "yardi_rag_jsons"
LOCAL_IMG_DIR  = Path(__file__).resolve().parent / "yardi_images"
LOCAL_DOC_IDX  = Path(__file__).resolve().parent / "doc_index.json"
# ---------------------------------------------


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"ERROR: Missing env var {name}", file=sys.stderr)
        sys.exit(1)
    return v


def make_s3():
    load_dotenv()  # read .env
    account_id = require_env("R2_ACCOUNT_ID")
    access_key = require_env("R2_ACCESS_KEY_ID")
    secret_key = require_env("R2_SECRET_ACCESS_KEY")
    bucket     = require_env("R2_BUCKET")

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5})
    )
    return s3, bucket


def iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def object_same_size(s3, bucket: str, key: str, size: int) -> bool:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        return head.get("ContentLength") == size
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        # If any other error, treat as not existing so we try to upload
        return False


def upload_one(s3, bucket: str, local: Path, key: str):
    # Skip if exists with same size (idempotent)
    size = local.stat().st_size
    if object_same_size(s3, bucket, key, size):
        return "skipped"

    extra = {}
    ctype, _ = mimetypes.guess_type(local.name)
    if ctype:
        extra["ContentType"] = ctype

    s3.upload_file(str(local), bucket, key, ExtraArgs=extra)
    return "uploaded"


def bulk_upload_prefix(s3, bucket: str, local_dir: Path, prefix: str, workers: int) -> tuple[int, int, int]:
    if not local_dir.exists():
        return 0, 0, 0
    files = list(iter_files(local_dir))
    up = 0; sk = 0; fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex, tqdm(total=len(files), desc=f"{prefix}") as bar:
        futures = {}
        for f in files:
            rel = f.relative_to(local_dir).as_posix()
            key = f"{prefix}/{rel}"
            futures[ex.submit(upload_one, s3, bucket, f, key)] = f
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res == "uploaded": up += 1
                elif res == "skipped": sk += 1
                else: fail += 1
            except Exception as e:
                fail += 1
                print(f"[FAIL] {futures[fut]} -> {e}", file=sys.stderr)
            finally:
                bar.update(1)
    return up, sk, fail


def count_prefix(s3, bucket: str, prefix: str) -> tuple[int, int]:
    """Return (object_count, total_bytes) under a prefix."""
    token = None
    count = 0
    total = 0
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            count += 1
            total += obj.get("Size", 0)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return count, total


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"


if __name__ == "__main__":
    s3, bucket = make_s3()

    # Upload JSONs
    up1, sk1, f1 = bulk_upload_prefix(s3, bucket, LOCAL_JSON_DIR, "yardi_rag_jsons", WORKERS)
    # Upload images
    up2, sk2, f2 = bulk_upload_prefix(s3, bucket, LOCAL_IMG_DIR, "yardi_images", WORKERS)

    # Upload doc_index.json if present
    if LOCAL_DOC_IDX.exists():
        try:
            s3.upload_file(str(LOCAL_DOC_IDX), bucket, "doc_index.json",
                           ExtraArgs={"ContentType": "application/json"})
            print("doc_index.json uploaded")
        except Exception as e:
            print(f"[FAIL] doc_index.json -> {e}", file=sys.stderr)
    else:
        print("doc_index.json not found locally; skipping")

    # Verify on server
    c1, b1 = count_prefix(s3, bucket, "yardi_rag_jsons/")
    c2, b2 = count_prefix(s3, bucket, "yardi_images/")
    print("\n==== SUMMARY ====")
    print(f"JSONs: uploaded {up1}, skipped {sk1}, failed {f1}; remote has {c1} files, {human_bytes(b1)}")
    print(f"IMGs : uploaded {up2}, skipped {sk2}, failed {f2}; remote has {c2} files, {human_bytes(b2)}")
    print("Done.")
