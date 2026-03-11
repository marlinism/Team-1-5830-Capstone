from __future__ import annotations

import argparse
import os
import re

from parking_processing.etl.run_etl import run_etl
from parking_processing.eda.plots import run_eda


def infer_year_from_path(path: str) -> int | None:
    base = os.path.basename(path)
    candidates = re.findall(r"(19\d{2}|20\d{2})", base)
    if candidates:
        return int(candidates[-1])  # take last match (usually most specific)
    candidates = re.findall(r"(19\d{2}|20\d{2})", path)
    if candidates:
        return int(candidates[-1])
    return None


def main():
    p = argparse.ArgumentParser(prog="smart_park_seattle")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_etl = sub.add_parser("etl")
    p_etl.add_argument("--raw", required=True, help="Path to raw yearly CSV")
    p_etl.add_argument("--out", required=True, help="Output folder for cleaned weekly parts")
    p_etl.add_argument("--year", type=int, default=None)
    p_etl.add_argument("--chunksize", type=int, default=2_000_000)
    p_etl.add_argument("--eda", action="store_true")
    p_etl.add_argument("--eda-outdir", default=None)

    p_eda = sub.add_parser("eda")
    p_eda.add_argument("--pattern", required=True)
    p_eda.add_argument("--outdir", required=True)

    args = p.parse_args()

    if args.cmd == "etl":
        year = args.year if args.year is not None else infer_year_from_path(args.raw)
        if year is None:
            raise SystemExit("Could not infer year from --raw path; please pass --year explicitly.")

        run_etl(raw_csv=args.raw, out_dir=args.out, year=year, chunksize=args.chunksize)

        if args.eda:
            pattern = os.path.join(args.out, f"year={year}", "week=*/part_*.csv.gz")
            outdir = args.eda_outdir or os.path.join(args.out, "eda", f"year={year}")
            run_eda(pattern=pattern, outdir=outdir)
        return

    if args.cmd == "eda":
        run_eda(pattern=args.pattern, outdir=args.outdir)
        return


if __name__ == "__main__":
    main()