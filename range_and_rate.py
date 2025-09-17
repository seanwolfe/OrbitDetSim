#!/usr/bin/env python3
"""
Parallel (MPI) master summary builder for per-run CSVs.

Round-robin file distribution across ranks:
  files are sorted; rank r processes files[i] where i % size == r.

Filename format (strict):
  minimoon-OBJECT_ID_sc-NUMBER_OF_SPACECRAFT_index-DETECTED_INDEX_spacecraft_DETECTED_SPACECRAFT_runs_NUMBER_OF_RUNS_run_RUN_NUMBER_part_1.csv

From the first row of each CSV, compute:
  DETECTED_RANGE(km) and DETECTED_RANGE_RATE(km/s)

Usage:
  mpiexec -n 8 python make_master_mpi.py /path/to/folder --out master_summary.csv
"""

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from mpi4py import MPI

# --- Columns we need (exact names) ---
OBJ_R_COLS = ["GEO_X(KM)", "GEO_Y(KM)", "GEO_Z(KM)"]
OBJ_V_COLS = ["GEO_VX(KM/S)", "GEO_VY(KM/S)", "GEO_VZ(KM/S)"]

SC_R_COLS = ["SC_GEO_X(KM)_PHYS", "SC_GEO_Y(KM)_PHYS", "SC_GEO_Z(KM)_PHYS"]
SC_V_COLS = ["SC_GEO_VX(KM/S)_PHYS", "SC_GEO_VY(KM/S)_PHYS", "SC_GEO_VZ(KM/S)_PHYS"]

# --- Filename regex (strict, tailored to your example) ---
FNAME_RE = re.compile(
    r"""
    ^minimoon-
    (?P<object_id>[^_]+)
    _sc-(?P<num_sc>\d+)
    _index-(?P<detected_index>\d+)
    _spacecraft_(?P<detected_spacecraft>\d+)
    _runs_(?P<num_runs>\d+)
    _run_(?P<run_number>\d+)
    _part_\d+\.csv$
    """,
    re.VERBOSE | re.IGNORECASE,
)

def range_and_range_rate(r_obj, v_obj, r_sc, v_sc):
    """Geometric range (km) and range-rate (km/s) in any inertial frame."""
    r_rel = np.array(r_obj, dtype=float) - np.array(r_sc, dtype=float)
    v_rel = np.array(v_obj, dtype=float) - np.array(v_sc, dtype=float)
    rho = np.linalg.norm(r_rel)
    if rho == 0.0:
        rhodot = np.nan
    else:
        rhodot = float(np.dot(r_rel, v_rel) / rho)
    return float(rho), rhodot

def parse_from_name(fname: str):
    """Extract fields from the filename."""
    m = FNAME_RE.match(fname)
    if not m:
        return None
    g = m.groupdict()
    return dict(
        OBJECT_ID=g["object_id"],
        NUMBER_OF_SPACECRAFT=int(g["num_sc"]),
        DETECTED_INDEX=int(g["detected_index"]),
        DETECTED_SPACECRAFT=int(g["detected_spacecraft"]),
        NUMBER_OF_RUNS=int(g["num_runs"]),
        RUN_NUMBER=int(g["run_number"]),
    )

def read_first_row(path: Path):
    """Read only the first data row from CSV (after header)."""
    df = pd.read_csv(path, nrows=1)
    if df.empty:
        raise ValueError("CSV has header but no data rows")
    return df.iloc[0]

def process_file(path: Path):
    meta = parse_from_name(path.name)
    if meta is None:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")

    row = read_first_row(path)

    # Ensure required columns exist
    for col in OBJ_R_COLS + OBJ_V_COLS + SC_R_COLS + SC_V_COLS:
        if col not in row.index:
            raise KeyError(f"Missing required column '{col}' in {path.name}")

    r_obj = [row[c] for c in OBJ_R_COLS]
    v_obj = [row[c] for c in OBJ_V_COLS]
    r_sc  = [row[c] for c in SC_R_COLS]
    v_sc  = [row[c] for c in SC_V_COLS]

    rng, rng_rate = range_and_range_rate(r_obj, v_obj, r_sc, v_sc)

    out = dict(meta)
    out["DETECTED_RANGE(km)"] = rng
    out["DETECTED_RANGE_RATE(km/s)"] = rng_rate
    return out

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ap = argparse.ArgumentParser(description="MPI-parallel summary of range and range-rate from CSV runs.")
    ap.add_argument("folder", type=str, help="Folder containing per-run CSV files")
    ap.add_argument("--out", type=str, default="master_summary.csv", help="Output master CSV (written by rank 0)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if rank == 0 and not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    # Broadcast the file list from rank 0
    if rank == 0:
        files = sorted([p for p in folder.glob("*.csv")])
    else:
        files = None
    files = comm.bcast(files, root=0)

    # Round-robin assignment
    my_files = [p for i, p in enumerate(files) if i % size == rank]

    my_records = []
    my_errors = []
    for path in my_files:
        try:
            rec = process_file(path)
            my_records.append(rec)
        except Exception as e:
            my_errors.append((path.name, str(e)))

    # Gather results and errors at root
    all_records = comm.gather(my_records, root=0)
    all_errors  = comm.gather(my_errors,  root=0)

    if rank == 0:
        # Flatten
        records = [r for sub in all_records for r in sub]
        errors  = [e for sub in all_errors  for e in sub]

        if not records and errors:
            msg = "\n".join([f"{n}: {err}" for n, err in errors])
            raise RuntimeError(f"No files processed successfully.\nErrors:\n{msg}")

        # Sort output deterministically (optional): by OBJECT_ID, RUN_NUMBER
        if records:
            records.sort(key=lambda d: (str(d["OBJECT_ID"]), int(d["RUN_NUMBER"])))

        out_df = pd.DataFrame(records, columns=[
            "OBJECT_ID",
            "NUMBER_OF_SPACECRAFT",
            "DETECTED_INDEX",
            "DETECTED_SPACECRAFT",
            "NUMBER_OF_RUNS",
            "RUN_NUMBER",
            "DETECTED_RANGE(km)",
            "DETECTED_RANGE_RATE(km/s)",
        ])

        out_path = Path(args.out)
        out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

        print(f"[rank 0] Wrote {len(out_df)} rows to {out_path.resolve()}")
        if errors:
            print("\nCompleted with some file-level errors:")
            for name, err in errors[:50]:
                print(f"  - {name}: {err}")
            if len(errors) > 50:
                print(f"  ... and {len(errors)-50} more")
    # Ensure rank 0 finishes printing before others exit
    comm.Barrier()

if __name__ == "__main__":
    main()
