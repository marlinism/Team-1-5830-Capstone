from __future__ import annotations

import os
import gzip
import boto3
import pandas as pd
from typing import Literal

from parking_processing.utils.s3 import parse_s3_uri


Target = Literal["y_15", "y_30", "both"]


def preds_dir(prefix: str, year: int, target: Target) -> str:
    """
    S3 folder for predictions.

    Example:
      prefix="parking/"
      -> "parking/preds/mvstgcn/year=2022/target=y_15/"
    """
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return f"{prefix}preds/mvstgcn/year={year}/target={target}/"


def preds_key(prefix: str, year: int, week: str, target: Target) -> str:
    return f"{preds_dir(prefix, year, target)}week={week}/pred.csv.gz"


def preds_success_key(prefix: str, year: int, week: str, target: Target) -> str:
    return f"{preds_dir(prefix, year, target)}week={week}/_SUCCESS"


def write_pred_csv_gz(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # ensure stable schema
    if "ts15_utc" in df.columns:
        df = df.copy()
        df["ts15_utc"] = pd.to_datetime(df["ts15_utc"], utc=True)

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)


def upload_week_predictions(
    *,
    s3_root: str,
    year: int,
    week: str,
    target: Target,
    local_csv_gz: str,
    success_text: str = "ok\n",
) -> str:
    """
    Upload one week's predictions to S3.

    Writes:
      pred.csv.gz
      _SUCCESS
    """
    s3 = boto3.client("s3")
    bucket, prefix = parse_s3_uri(s3_root)

    key = preds_key(prefix, year, week, target)
    suc = preds_success_key(prefix, year, week, target)

    # Upload file
    s3.upload_file(local_csv_gz, bucket, key)
    # Write success marker
    s3.put_object(Bucket=bucket, Key=suc, Body=success_text.encode("utf-8"))

    return f"s3://{bucket}/{key}"


def week_predictions_exist(*, s3_root: str, year: int, week: str, target: Target) -> bool:
    """
    Check if _SUCCESS exists for this week/target.
    """
    s3 = boto3.client("s3")
    bucket, prefix = parse_s3_uri(s3_root)
    suc = preds_success_key(prefix, year, week, target)
    try:
        s3.head_object(Bucket=bucket, Key=suc)
        return True
    except Exception:
        return False

def preds_dir_model(prefix: str, year: int, target: Target, model_name: str) -> str:
    """
    S3 folder for predictions under a model namespace.

    Example:
      prefix="parking/"
      model_name="fallback"
      -> "parking/preds/fallback/year=2022/target=y_15/"
    """
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return f"{prefix}preds/{model_name}/year={year}/target={target}/"


def preds_key_model(prefix: str, year: int, week: str, target: Target, model_name: str) -> str:
    return f"{preds_dir_model(prefix, year, target, model_name)}week={week}/pred.csv.gz"


def preds_success_key_model(prefix: str, year: int, week: str, target: Target, model_name: str) -> str:
    return f"{preds_dir_model(prefix, year, target, model_name)}week={week}/_SUCCESS"


def upload_week_predictions_model(
    *,
    s3_root: str,
    year: int,
    week: int | str,
    target: Target,
    local_csv_gz: str,
    model_name: str,
    success_text: str = "ok\n",
) -> str:
    """
    Upload one week's predictions to S3 under a model namespace.

    Writes:
      preds/<model_name>/year=YYYY/target=.../week=.../pred.csv.gz
      preds/<model_name>/year=YYYY/target=.../week=.../_SUCCESS
    """
    s3 = boto3.client("s3")
    bucket, prefix = parse_s3_uri(s3_root)

    week = str(week)
    key = preds_key_model(prefix, year, week, target, model_name)
    suc = preds_success_key_model(prefix, year, week, target, model_name)

    s3.upload_file(local_csv_gz, bucket, key)
    s3.put_object(Bucket=bucket, Key=suc, Body=success_text.encode("utf-8"))

    return f"s3://{bucket}/{key}"


def week_predictions_exist_model(
    *,
    s3_root: str,
    year: int,
    week: str,
    target: Target,
    model_name: str,
) -> bool:
    """
    Check if _SUCCESS exists for this week/target under a model namespace.
    """
    s3 = boto3.client("s3")
    bucket, prefix = parse_s3_uri(s3_root)
    suc = preds_success_key_model(prefix, year, week, target, model_name)
    try:
        s3.head_object(Bucket=bucket, Key=suc)
        return True
    except Exception:
        return False        