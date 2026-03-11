import os
import gzip
import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_gz_csv(df: pd.DataFrame, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)