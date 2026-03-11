from __future__ import annotations
import io, gzip
from urllib.parse import urlparse
import boto3
import pandas as pd


def parse_s3_uri(s3_uri: str):
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 uri must start with s3://")
    u = urlparse(s3_uri)
    bucket = u.netloc
    prefix = u.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def s3_client():
    return boto3.client("s3")


def s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def s3_put_text(s3, bucket: str, key: str, text: str):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))


def s3_get_text(s3, bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


def s3_list_keys(s3, bucket: str, prefix: str):
    """Yield S3 keys under prefix."""
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            yield it["Key"]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")


def s3_download_file(s3, bucket: str, key: str, local_path: str):
    s3.download_file(bucket, key, local_path)


def s3_upload_file(s3, local_path: str, bucket: str, key: str):
    s3.upload_file(local_path, bucket, key)


def s3_read_csv_gz(s3, bucket: str, key: str, usecols=None) -> pd.DataFrame:
    """Read a gzip CSV stored in S3 into a DataFrame."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as f:
        return pd.read_csv(f, low_memory=False, usecols=usecols)