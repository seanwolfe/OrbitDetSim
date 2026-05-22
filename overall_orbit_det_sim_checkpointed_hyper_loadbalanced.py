import yaml
import os
import pandas as pd
from Asteroid import Asteroid
from Formation import Formation
import numpy as np
import mpi4py.rc
from od_spkf_adaptiveR_Pinjection_decay_no_floor_mature_single_nonaug_nodetect_continue import OD_UKF, od_setup_from_iod, process_tracklet_until_update_with_prior_epoch
from time_tracker import SimTime
import copy
mpi4py.rc.threads = False
from mpi4py import MPI
import spiceypy as sp
import utilities as util
import n_body_integrator as nbody
import argparse
import json
import csv
import gc
import glob
import datetime as dt
import matplotlib.pyplot as plt
from od_attcoord_coverage_mode_v2_tracking_anchor_emsfree import AttitudeCoordinator, compute_J_grid_theta_phi_single_free
import math
import time
from datetime import datetime
import traceback
import pickle
import itertools
import hashlib

# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
sp.furnsh("de430.bsp")
sp.furnsh('naif0012.tls')


def run_runs_x_minimoons_MPI(minimoon_master, config):
    """
    Distribute work over the Cartesian product of:
      run_number in [1..number_of_runs]  AND  minimoon_master rows [0..M-1]
    Saves outputs under a directory named `spacecraft_<num_spacecraft>` and
    includes that tag in the filename.

    Now stores BOTH:
      - values:        visible indices with (FOV + Earth/Moon occlusion)
      - values_ems:    visible indices further filtered by EMS exclusion
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ----------------- config -----------------
    N_runs = int(config['number_of_runs'])
    M_mm = int(len(minimoon_master))
    rows_per_part = int(config.get('number_of_rows_per_part', 50000))
    save_format = config.get('save_format', 'csv')  # 'csv' or 'parquet'
    base_out = str(config['output_df_file_name'])
    base_seed = int(config.get('seed', 12345))

    # required: num_spacecraft in config
    if 'num_spacecraft' not in config:
        raise ValueError("config must include 'num_spacecraft'.")
    num_sc = int(config['num_spacecraft'])

    # ------------- output directory -----------
    base_dir = config['top_dir']
    base_name = os.path.basename(base_out)
    out_dir = os.path.join(base_dir, config["visible_files_folder"])

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    comm.Barrier()  # ensure dir exists before others write

    # ------------- task decomposition ----------
    T = N_runs * M_mm
    task_indices = np.array_split(np.arange(T), size)[rank]

    # ------------- buffers / counters ----------
    cols = [
        "run_number", "object_id", "spacecraft_number",
        "values", "values_ems", "total_length", "spacecraft_1_ini_pos"
    ]
    df_buffer = pd.DataFrame(columns=cols)
    part_number = 1

    # ------------- helper: save buffer ----------
    def flush_buffer():
        nonlocal df_buffer, part_number
        if df_buffer.empty:
            return
        df_out = df_buffer.set_index(["run_number", "object_id", "spacecraft_number"])
        base_filename = os.path.join(
            out_dir,
            f"{base_name}_spacecraft_{num_sc}_rank_{rank}_part_{part_number}"
        )
        if save_format == 'csv':
            filename = base_filename + ".csv"
            df_out.to_csv(filename, sep=',', header=True, index=True)
        elif save_format == 'parquet':
            filename = base_filename + ".parquet"
            df_out.to_parquet(filename, index=True)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        print(f"[rank {rank}] saved {len(df_out)} rows to {filename}", flush=True)
        df_buffer = pd.DataFrame(columns=cols)
        part_number += 1

    # ------------- main loop over my tasks -----
    for t in task_indices:
        run_idx, mm_idx = divmod(int(t), M_mm)
        run_no = run_idx + 1

        # Reproducible RNG per (run_no, mm_idx)
        ss = np.random.SeedSequence([base_seed, run_no, mm_idx])
        np.random.seed(ss.generate_state(1)[0] & 0xFFFFFFFF)

        # ======= do the work for ONE (run, minimoon) =======
        master_i = minimoon_master.iloc[mm_idx]
        current_minimoon = Asteroid(
            master_i['Object id'],
            master_i['Capture Index'],
            config
        )
        formation = Formation(config)
        # old
        # asteroid_pos = current_minimoon.orbit.loc[:, ['Synodic x', 'Synodic y', 'Synodic z']].values
        # earth_pos = np.zeros_like(asteroid_pos)
        # moon_pos = current_minimoon.orbit.loc[:, ['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']].values
        # 
        # formation.match_spacecraft_trajectory(len(asteroid_pos[:, 0]), config)
        # 
        
        # new
        ast_traj_au_eclip = np.array(current_minimoon.orbit.loc[:, ["Geo x", "Geo y", "Geo z",
                                                                    "Geo vx", "Geo vy", "Geo vz"]])
        ast_traj_au_eclip[:, :3] *= (config["AU_TO_M"] / config["KM_TO_M"])
        ast_traj_au_eclip[:, 3:] *= (config["AU_TO_M"] / config["KM_TO_M"] / config["SECONDS_PER_DAY"])
        ast_traj = util.geo_eclip_to_geo_eme_generic(ast_traj_au_eclip)
        jdtdb_epochs = current_minimoon.orbit["Julian Date"]
        formation.match_spacecraft_trajectory_full(len(ast_traj[:, 0]), config)

        # set boresights
        earth_helio_ae = np.array(current_minimoon.orbit.loc[:, ["Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)",
                         "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)"]])
        earth_helio_ae[:, :3] *= (config["AU_TO_M"] / config["KM_TO_M"])
        earth_helio_ae[:, 3:] *= (config["AU_TO_M"] / config["KM_TO_M"] / config["SECONDS_PER_DAY"])
        boresight_ini_secr = np.zeros((len(earth_helio_ae[:, 0]), 3))
        boresight_ini_secr[:, 0] = -1
        boresight_ini_eclip = util.geo_secr_to_geo_eclip_generic(boresight_ini_secr, earth_helio_ae,
                                                                 obj_hint=('time', 'position'),
                                                                 earth_hint=('time', 'state'))
        boresight_ini_eme = util.geo_eclip_to_geo_eme_generic(boresight_ini_eclip,
                                                              hint=('time', 'position'))

        
        new_rows = []
        for jdx, spacecraft in enumerate(formation.spacecraft):
        
            # old
            # sc_pos = spacecraft.matched_trajectory
            # 
            # NEW: function now returns (base_result, ems_filtered_result)
            # visible_base, visible_ems = spacecraft.asteroid_in_fov_batch(
            #     asteroid_pos, sc_pos, earth_pos, moon_pos, config
            # )
        
            # new
            # get s/c eme traj at ast epoch
            sc_traj = util.sc_eme_ast_eme(sc_df=spacecraft.matched_trajectory_full, earth_ast_kms_array=earth_helio_ae)

            # evaluate detection
            visible_base, visible_ems = spacecraft.asteroid_in_fov_batch_km_geocentric(ast_traj[:, :3], sc_traj[:, :3],
                                                                                       boresight_ini_eme,
                                                                                       jdtdb_epochs, config)

            visible_base = np.asarray(visible_base)
            visible_ems = np.asarray(visible_ems)

            len_vis = len(visible_base)  # total epochs

            # Extract indices (>=0)
            base_idx = visible_base[visible_base >= 0].astype(int)
            ems_idx = visible_ems[visible_ems >= 0].astype(int)

            new_rows.append({
                "run_number": run_no,
                "object_id": current_minimoon.id,
                "spacecraft_number": jdx + 1,
                "values": tuple(base_idx),
                "values_ems": tuple(ems_idx),
                "total_length": len_vis,
                "spacecraft_1_ini_pos": tuple(formation.spacecraft[0].ini_position)
            })

        # Append and flush if big
        df_buffer = pd.concat([df_buffer, pd.DataFrame(new_rows)], ignore_index=True)
        if len(df_buffer) >= rows_per_part:
            flush_buffer()

    # flush any leftovers and sync
    flush_buffer()
    comm.Barrier()

    return


def run_sim_runnumbers_MPI_getIOD(config):
    """
    MPI stage that reads detection 'visible' files from:
        visible_files_folder/spacecraft_{num_spacecraft}
    and produces IOD files into:
        IOD_folder_path/spacecraft_{num_spacecraft}
    Also appends a single MASTER_IOD.csv (resumable, duplicate-safe).

    UPDATE (EMS-aware selection):
      Visible files now contain:
        - values       : indices visible with (FOV + Earth/Moon occlusion)
        - values_ems   : indices visible with (FOV + Earth/Moon occlusion + EMS exclusion)

      This stage writes BOTH into MASTER:
        - VALUES_IDX
        - VALUES_IDX_EMS

      And chooses INDEX_USED depending on:
        - config["INCLUDE_EMS_EXCLUSION"] (bool, default False)
          * False -> INDEX_USED = min(VALUES_IDX)
          * True  -> INDEX_USED = min(VALUES_IDX_EMS) if non-empty else min(VALUES_IDX)

      Adds:
        - OCCLUDED_BY_EMS (0/1): at INDEX_USED, would EMS exclude it?
          * defined as 1 iff INDEX_USED not in VALUES_IDX_EMS (and INDEX_USED exists)
          * else 0
          * left blank if INDEX_USED missing
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ---------- paths / config ----------
    num_sc = int(config["num_spacecraft"])
    top_dir = os.path.abspath(config['top_dir'])
    vis_dir = os.path.abspath(os.path.join(top_dir, config["visible_files_folder"]))
    iod_dir = os.path.abspath(os.path.join(top_dir, config["IOD_folder_path"]))
    data_done_dir = os.path.join(iod_dir, "data_done")
    save_format = config.get("save_format", "csv")  # 'csv' | 'parquet' | 'both'

    include_ems_exclusion = bool(config.get("INCLUDE_EMS_EXCLUSION", False))

    # Master CSV + per-row done markers (for resumability)
    master_name = "MASTER_IOD.csv"
    master_path = os.path.join(top_dir, master_name)
    master_done_dir = os.path.join(iod_dir, "master_rows_done")

    if rank == 0:
        os.makedirs(iod_dir, exist_ok=True)
        os.makedirs(master_done_dir, exist_ok=True)
        os.makedirs(data_done_dir, exist_ok=True)
        if not os.path.isdir(vis_dir):
            raise FileNotFoundError(f"Visible files dir not found: {vis_dir}")
    comm.Barrier()

    # ---------- MASTER column schema ----------
    helio_sc_cols = [f"HELIO_SC_{i + 1}(kms)" for i in range(num_sc)]
    epoch_sc_cols = [f"EPOCH_SC_{i + 1}(jdtdb)" for i in range(num_sc)]
    boresight_cols = [f"BORESIGHT_SC_{i + 1}_GEO_SECR" for i in range(num_sc)]

    # detection metadata columns (in MASTER)
    det_meta_cols = [
        "VALUES_IDX",  # serialized list/tuple
        "VALUES_IDX_EMS",  # serialized list/tuple (EMS-filtered)
        "TOTAL_LENGTH",
        "SPACECRAFT_1_INI_POS(km)",  # serialized 3-vector
        "INDEX_USED",  # chosen index for IOD (EMS-aware toggle + fallback)
        "OCCLUDED_BY_EMS",  # 0/1 (blank if INDEX_USED missing)
    ]

    master_columns = (
            [
                "ID_AST",
                "EPOCH_AST(jdtdb)",
                "HELIO_AST(kms)",
                "DETECTING_SC_ID",
            ]
            + det_meta_cols
            + epoch_sc_cols
            + helio_sc_cols
            + boresight_cols
            + ["IOD_DATA_SAVED_AS"]
    )

    # ---------- serialization helpers ----------
    def serialize_vec(v):
        """1D array-like -> 'v0,v1,...' """
        return ",".join(f"{float(x):.16g}" for x in np.asarray(v).ravel())

    def serialize_any_listlike(x):
        """
        For values / mixed iterables:
          - if already a string, return as-is
          - if list/tuple/np array -> serialize numbers like v0,v1,...
          - else -> str(x)
        """
        if isinstance(x, str):
            return x
        try:
            arr = np.asarray(x)
            if arr.ndim == 0:
                return str(x)
            return ",".join(f"{float(v):.16g}" for v in arr.ravel())
        except Exception:
            return str(x)

    def _as_int_listlike(x):
        """
        Robustly coerce a 'values' cell into a 1D int numpy array.
        Works if x is already list/tuple/ndarray. If x is a comma-separated string,
        tries to parse it. Empty/invalid -> empty array.
        """
        if x is None:
            return np.array([], dtype=int)
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s == "()" or s == "[]" or s.lower() == "nan":
                return np.array([], dtype=int)
            # util.read_master often parses these already, but handle plain CSV strings too
            parts = [p for p in s.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",") if
                     p.strip() != ""]
            out = []
            for p in parts:
                try:
                    out.append(int(float(p)))
                except Exception:
                    pass
            return np.asarray(out, dtype=int)
        try:
            arr = np.asarray(x).ravel()
            out = []
            for v in arr:
                try:
                    out.append(int(v))
                except Exception:
                    pass
            return np.asarray(out, dtype=int)
        except Exception:
            return np.array([], dtype=int)

    def _min_index_or_nan(values_cell):
        arr = _as_int_listlike(values_cell)
        if arr.size == 0:
            return np.nan
        # stored values should already be indices >= 0, but keep it safe:
        arr = arr[arr >= 0]
        if arr.size == 0:
            return np.nan
        return float(np.min(arr))

    def write_master_header_if_needed():
        if not os.path.exists(master_path):
            pd.DataFrame(columns=master_columns).to_csv(master_path, index=False)

    def master_row_done_path(row_uid):
        return os.path.join(master_done_dir, f"{row_uid}.done")

    def master_row_already_done(row_uid):
        return os.path.exists(master_row_done_path(row_uid))

    def mark_master_row_done(row_uid):
        with open(master_row_done_path(row_uid), "w") as f:
            f.write("ok\n")

    if rank == 0:
        write_master_header_if_needed()
    comm.Barrier()

    # ---------- file listing ----------
    if rank == 0:
        if hasattr(util, "get_all_files"):
            all_files = util.get_all_files(vis_dir, save_format)
        else:
            patterns = []
            if save_format in ("csv", "both"):
                patterns.append(os.path.join(vis_dir, "*.csv"))
            if save_format in ("parquet", "both"):
                patterns.append(os.path.join(vis_dir, "*.parquet"))
            all_files = []
            for pat in patterns:
                all_files.extend(glob.glob(pat))
        all_files = sorted(all_files)
        total_files = len(all_files)
        file_counter = 0
    else:
        all_files = None
        total_files = None
        file_counter = None

    all_files = comm.bcast(all_files, root=0)
    total_files = comm.bcast(total_files, root=0)

    # ---------- helpers ----------
    def base_source_name(path):
        return os.path.splitext(os.path.basename(path))[0]

    def done_marker_path(basename):
        return os.path.join(data_done_dir, f".done_{basename}.json")

    def outputs_for_source_exist(basename):
        counts = {}
        if save_format in ("csv", "both"):
            counts["csv"] = len(glob.glob(os.path.join(data_done_dir, f"*_{basename}.csv")))
        if save_format in ("parquet", "both"):
            counts["parquet"] = len(glob.glob(os.path.join(data_done_dir, f"*_{basename}.parquet")))
        return counts

    def row_outputs_exist(base_path):
        csv_exists = os.path.exists(base_path + ".csv")
        pq_exists = os.path.exists(base_path + ".parquet")
        if save_format == "csv":
            return csv_exists
        elif save_format == "parquet":
            return pq_exists
        else:  # both
            return csv_exists and pq_exists

    def _to_jdtdb(timestr):
        et = sp.str2et(timestr)
        return float(sp.unitim(et, "ET", "JDTDB"))

    def _as_6xN(A):
        """Force states to shape (6,N) when possible (accepts (N,6) or (6,N))."""
        X = np.asarray(A, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for state series, got shape {X.shape}")
        if X.shape[0] == 6:
            return X
        if X.shape[1] == 6:
            return X.T
        raise ValueError(f"Cannot coerce to (6,N): shape {X.shape}")

    # ---------- main loop over source files ----------
    for file_i in all_files:
        src_base = base_source_name(file_i)

        # Rank 0 decides to skip or process this file
        if rank == 0:
            file_counter += 1
            marker = done_marker_path(src_base)
            if os.path.exists(marker):
                skip_file = True
                print(f"[IOD] {file_counter}/{total_files} SKIP (DONE): {os.path.basename(file_i)}", flush=True)
            else:
                skip_file = False
                print(f"[IOD] {file_counter}/{total_files} RUN : {os.path.basename(file_i)}", flush=True)
        else:
            skip_file = None

        skip_file = comm.bcast(skip_file, root=0)
        if skip_file:
            comm.Barrier()
            continue

        # ---------- rank 0 reads & prepares detected rows, then splits ----------
        if rank == 0:
            print(file_i)
            run_data = util.read_master(file_i, config)

            # Expecting new columns from visible files:
            #   values      (base visibility)
            #   values_ems  (EMS-filtered visibility)
            if "values" not in run_data.columns:
                raise KeyError(f"'values' column not found in visible file: {file_i}")
            if "values_ems" not in run_data.columns:
                raise KeyError(
                    f"'values_ems' column not found in visible file: {file_i}. "
                    f"Update the visible-file generation to include values_ems."
                )

            # compute base/ems mins
            run_data["min_nonnegative_base"] = run_data["values"].apply(_min_index_or_nan)
            run_data["min_nonnegative_ems"] = run_data["values_ems"].apply(_min_index_or_nan)

            # choose INDEX_USED based on toggle + fallback
            def _choose_index_used(row):
                b = row["min_nonnegative_base"]
                e = row["min_nonnegative_ems"]
                if include_ems_exclusion:
                    if np.isfinite(e):
                        return e
                    return b
                else:
                    return b

            run_data["index_used"] = run_data.apply(_choose_index_used, axis=1)

            # OCCLUDED_BY_EMS: at index_used, would EMS exclude it?
            # definition: 1 iff index_used is finite AND not present in values_ems
            def _occluded_by_ems_flag(row):
                idx = row["index_used"]
                if not np.isfinite(idx):
                    return np.nan
                idx = int(idx)
                ems_list = _as_int_listlike(row["values_ems"])
                return 0 if np.any(ems_list == idx) else 1

            run_data["occluded_by_ems"] = run_data.apply(_occluded_by_ems_flag, axis=1)

            # "detected" population for IOD = rows where INDEX_USED is finite
            detected_pop = run_data[np.isfinite(run_data["index_used"])]

            if len(detected_pop) == 0:
                with open(done_marker_path(src_base), "w") as f:
                    json.dump(
                        {
                            "source_file": os.path.basename(file_i),
                            "time_utc": dt.datetime.utcnow().isoformat() + "Z",
                            "num_rows_expected": 0,
                            "num_rows_csv": 0,
                            "num_rows_parquet": 0,
                        },
                        f,
                        indent=2,
                    )
                print(f"[IOD] No detections. Marked DONE for {os.path.basename(file_i)}", flush=True)
                chunks = [detected_pop] * size
                expected_rows = 0
            else:
                idx_splits = np.array_split(np.arange(len(detected_pop)), size)
                chunks = [
                    detected_pop.iloc[idxs] if len(idxs) else detected_pop.iloc[0:0]
                    for idxs in idx_splits
                ]
                expected_rows = len(detected_pop)
        else:
            chunks = None
            expected_rows = None

        expected_rows = comm.bcast(expected_rows, root=0)

        # ---------- scatter chunks ----------
        if rank == 0:
            for dest in range(1, size):
                comm.send(chunks[dest], dest=dest, tag=77)
            my_chunk = chunks[0]
        else:
            my_chunk = comm.recv(source=0, tag=77)

        print(f"[rank {rank}] {src_base}: chunk size = {len(my_chunk)}", flush=True)
        comm.Barrier()

        # -------- per-rank buffer of master rows --------
        master_rows_buffer = []

        if len(my_chunk) > 0:
            # new augmented DF
            detected_appended_pop_chunk = util.get_scs_initial_states_new(my_chunk, config)

            for _, detected_minimoon in detected_appended_pop_chunk.iterrows():
                mm_id = detected_minimoon.name[1]
                sc_id = detected_minimoon.name[2]

                idx0 = int(detected_minimoon["index_used"])

                file_name = f"minimoon-{mm_id}_sc-{int(sc_id)}_index-{idx0}_{src_base}"
                base_path = os.path.join(data_done_dir, file_name)
                row_uid = file_name

                if master_row_already_done(row_uid) and row_outputs_exist(base_path):
                    continue

                # ---- asteroid orbit at detection ----
                orbit_path = os.path.join(config["minimoon_files_folder"], f"{mm_id}.csv")
                orbit = util.read_csv_comma_or_space(orbit_path, header=0)

                asteroid_state_helio = orbit.loc[
                    idx0, ["Helio x", "Helio y", "Helio z", "Helio vx", "Helio vy", "Helio vz"]
                ].values.astype(float)

                asteroid_state_helio[:3] *= (config["AU_TO_M"] / config["KM_TO_M"])
                asteroid_state_helio[3:] *= (config["AU_TO_M"] / config["KM_TO_M"] / config["SECONDS_PER_DAY"])

                asteroid_epoch = float(orbit.loc[idx0, "Julian Date"])  # assumed jdtdb

                # ---- measurement sequence ----
                num_frames = int(config["number_of_frames"])
                step_days = config["time_between_frames"] / config["SECONDS_PER_DAY"]
                epochs = asteroid_epoch + step_days * np.arange(num_frames)
                total_window_s = num_frames * config["time_between_frames"]

                # ---- integrate asteroid (needed for IOD) ----
                asteroid_integrated_states, asteroid_earth_states = nbody.integrate_n_body(
                    asteroid_state_helio, asteroid_epoch, total_window_s,
                    config["time_between_frames"], type="ASTEROID"
                )

                asteroid_state_secr = util.helio_eclip_to_geo_secr_generic(
                    asteroid_integrated_states, asteroid_earth_states, layout="time"
                )

                # ---- detecting spacecraft initial state/epoch from DF ----
                did = int(detected_minimoon["detecting_id"])
                sc_epoch_str = detected_minimoon[f"SC{did}_epoch"]
                sc_geo_ini = np.asarray(detected_minimoon[f"SC{did}_GEO_ECLIP_state"], dtype=float)

                sun_geo_state = np.asarray(
                    sp.spkgeo(10, sp.str2et(sc_epoch_str), "ECLIPJ2000", 399)[0],
                    dtype=float
                )
                sc_helio_ini = sc_geo_ini - sun_geo_state

                sc_int_states, earth_states = nbody.integrate_n_body(
                    sc_helio_ini, sc_epoch_str, total_window_s,
                    config["time_between_frames"], type="SPACECRAFT"
                )

                sc_secr = util.helio_eclip_to_geo_secr_generic(sc_int_states, earth_states, layout="time")

                # ---- transport to asteroid-time ----
                sc_geo = util.geo_secr_to_geo_eclip_generic(sc_secr, asteroid_earth_states, layout="time")
                ast_geo = util.geo_secr_to_geo_eclip_generic(asteroid_state_secr, asteroid_earth_states, layout="time")

                sc_geo_eme = util.geo_eclip_to_geo_eme_generic(sc_geo, layout="time")
                ast_geo_eme = util.geo_eclip_to_geo_eme_generic(ast_geo, layout="time")

                sc_geo_eme = _as_6xN(sc_geo_eme)
                ast_geo_eme = _as_6xN(ast_geo_eme)

                # ---- geometrically sound meas ----
                x_rel = ast_geo_eme[0, :] - sc_geo_eme[0, :]
                y_rel = ast_geo_eme[1, :] - sc_geo_eme[1, :]
                z_rel = ast_geo_eme[2, :] - sc_geo_eme[2, :]
                r_xy = np.hypot(x_rel, y_rel)
                r = np.sqrt(r_xy ** 2 + z_rel ** 2)
                eps = 1e-12
                sin_ra = y_rel / np.maximum(r_xy, eps)
                cos_ra = x_rel / np.maximum(r_xy, eps)
                sin_dec = z_rel / np.maximum(r, eps)

                # ---- physically sound meas ----
                sc_secr_ini = _as_6xN(sc_secr)[:, 0]
                earth_helio_ini = _as_6xN(asteroid_earth_states)[:, 0]
                sc_helio_ini_state = util.geo_secr_to_helio_eclip_generic(sc_secr_ini, earth_helio_ini)

                sc_helio_states, asteroid_earth_states_2 = nbody.integrate_n_body(
                    sc_helio_ini_state, asteroid_epoch, total_window_s,
                    config["time_between_frames"], type="SPACECRAFT-ASTEROIDTIME"
                )

                sc_eme_states = util.helio_eclip_to_geo_eme_generic(
                    sc_helio_states, asteroid_earth_states, layout="time"
                )
                sc_eme_states = _as_6xN(sc_eme_states)

                x_rel_p = ast_geo_eme[0, :] - sc_eme_states[0, :]
                y_rel_p = ast_geo_eme[1, :] - sc_eme_states[1, :]
                z_rel_p = ast_geo_eme[2, :] - sc_eme_states[2, :]
                r_xy_p = np.hypot(x_rel_p, y_rel_p)
                r_p = np.sqrt(r_xy_p ** 2 + z_rel_p ** 2)
                sin_ra_p = y_rel_p / np.maximum(r_xy_p, eps)
                cos_ra_p = x_rel_p / np.maximum(r_xy_p, eps)
                sin_dec_p = z_rel_p / np.maximum(r_p, eps)

                # ---- IOD df ----
                data = np.array(
                    [
                        epochs,
                        ast_geo_eme[0, :], ast_geo_eme[1, :], ast_geo_eme[2, :],
                        ast_geo_eme[3, :], ast_geo_eme[4, :], ast_geo_eme[5, :],
                        sc_geo_eme[0, :], sc_geo_eme[1, :], sc_geo_eme[2, :],
                        sc_geo_eme[3, :], sc_geo_eme[4, :], sc_geo_eme[5, :],
                        sin_ra, cos_ra, sin_dec,
                        sc_eme_states[0, :], sc_eme_states[1, :], sc_eme_states[2, :],
                        sc_eme_states[3, :], sc_eme_states[4, :], sc_eme_states[5, :],
                        sin_ra_p, cos_ra_p, sin_dec_p,
                    ]
                ).T

                df_iod = pd.DataFrame(data, columns=config["IOD_data_columns_geo_and_phys"])

                # ---- write outputs ----
                if save_format in ("csv", "both") and not os.path.exists(base_path + ".csv"):
                    df_iod.to_csv(base_path + ".csv", index=False)
                if save_format in ("parquet", "both") and not os.path.exists(base_path + ".parquet"):
                    df_iod.to_parquet(base_path + ".parquet", index=False)

                saved_files = []
                if os.path.exists(base_path + ".parquet"):
                    saved_files.append(os.path.basename(base_path) + ".parquet")
                if os.path.exists(base_path + ".csv"):
                    saved_files.append(os.path.basename(base_path) + ".csv")
                saved_as_str = ";".join(saved_files) if saved_files else ""

                # ==========================
                # MASTER row
                # ==========================
                if not master_row_already_done(row_uid):
                    did = int(detected_minimoon["detecting_id"])

                    # detecting SC epoch (jdtdb)
                    det_epoch_jdtdb = _to_jdtdb(detected_minimoon[f"SC{did}_epoch"])

                    # per-SC epochs + helio initial states computed from DF
                    helio_sc_states = np.zeros((num_sc, 6), dtype=float)
                    epoch_sc_jdtdb = np.zeros((num_sc,), dtype=float)

                    for i in range(1, num_sc + 1):
                        sc_epoch_i = detected_minimoon[f"SC{i}_epoch"]
                        epoch_sc_jdtdb[i - 1] = _to_jdtdb(sc_epoch_i)

                        sc_geo_i = np.asarray(detected_minimoon[f"SC{i}_GEO_ECLIP_state"], dtype=float)
                        sun_geo_i = np.asarray(
                            sp.spkgeo(10, sp.str2et(sc_epoch_i), "ECLIPJ2000", 399)[0],
                            dtype=float
                        )
                        helio_sc_states[i - 1, :] = sc_geo_i - sun_geo_i

                    # boresights
                    boresights = []
                    for i in range(1, num_sc + 1):
                        bs = detected_minimoon.get(f"SC{i}_boresight", None)
                        boresights.append(bs)

                    # requested detection metadata
                    values_ser = serialize_any_listlike(detected_minimoon.get("values", ""))
                    values_ems_ser = serialize_any_listlike(detected_minimoon.get("values_ems", ""))

                    total_length_val = int(detected_minimoon.get("total_length", 0))
                    sc1_ini_pos = detected_minimoon.get("spacecraft_1_ini_pos", None)
                    sc1_ini_pos_ser = serialize_vec(sc1_ini_pos) if sc1_ini_pos is not None else ""

                    index_used_val = int(detected_minimoon.get("index_used", -1))

                    occluded_flag = detected_minimoon.get("occluded_by_ems", np.nan)
                    if np.isfinite(occluded_flag):
                        occluded_flag_ser = str(int(occluded_flag))
                    else:
                        occluded_flag_ser = ""

                    row_dict = {
                        "ID_AST": mm_id,
                        "EPOCH_AST(jdtdb)": f"{asteroid_epoch:.16f}",
                        "HELIO_AST(kms)": serialize_vec(asteroid_state_helio),
                        "DETECTING_SC_ID": did,

                        "VALUES_IDX": values_ser,
                        "VALUES_IDX_EMS": values_ems_ser,
                        "TOTAL_LENGTH": str(total_length_val),
                        "SPACECRAFT_1_INI_POS(km)": sc1_ini_pos_ser,
                        "INDEX_USED": str(index_used_val),
                        "OCCLUDED_BY_EMS": occluded_flag_ser,

                        "IOD_DATA_SAVED_AS": saved_as_str,
                    }

                    # per-SC epochs
                    for i in range(1, num_sc + 1):
                        row_dict[f"EPOCH_SC_{i}(jdtdb)"] = f"{epoch_sc_jdtdb[i - 1]:.16f}"

                    # per-SC helio states
                    for i in range(1, num_sc + 1):
                        row_dict[f"HELIO_SC_{i}(kms)"] = serialize_vec(helio_sc_states[i - 1, :])

                    # per-SC boresights
                    for i in range(1, num_sc + 1):
                        bs = boresights[i - 1]
                        row_dict[f"BORESIGHT_SC_{i}_GEO_SECR"] = serialize_vec(bs) if bs is not None else ""

                    ordered_row = {col: row_dict.get(col, "") for col in master_columns}
                    master_rows_buffer.append((row_uid, ordered_row))

        # -------- gather and append to MASTER on rank 0 --------
        gathered = comm.gather(master_rows_buffer, root=0)

        if rank == 0:
            all_items = [item for chunk in gathered for item in chunk]
            if all_items:
                filtered_items = [(uid, row) for (uid, row) in all_items if not master_row_already_done(uid)]
                if filtered_items:
                    write_master_header_if_needed()
                    rows_only = [row for (_, row) in filtered_items]
                    df_master_append = pd.DataFrame(rows_only, columns=master_columns)
                    df_master_append = df_master_append[master_columns]
                    df_master_append.to_csv(master_path, mode="a", header=False, index=False)
                    for (uid, _) in filtered_items:
                        mark_master_row_done(uid)

        comm.Barrier()

        # Rank 0: completeness marker
        if rank == 0:
            counts = outputs_for_source_exist(src_base)
            ok_csv = (save_format in ("csv", "both")) and (counts.get("csv", 0) >= (expected_rows or 0))
            ok_pq = (save_format in ("parquet", "both")) and (counts.get("parquet", 0) >= (expected_rows or 0))
            complete = (
                    (save_format == "csv" and ok_csv)
                    or (save_format == "parquet" and ok_pq)
                    or (save_format == "both" and ok_csv and ok_pq)
                    or (expected_rows == 0)
            )
            if complete:
                with open(done_marker_path(src_base), "w") as f:
                    json.dump(
                        {
                            "source_file": os.path.basename(file_i),
                            "time_utc": dt.datetime.utcnow().isoformat() + "Z",
                            "num_rows_expected": int(expected_rows or 0),
                            "num_rows_csv": int(counts.get("csv", 0)),
                            "num_rows_parquet": int(counts.get("parquet", 0)),
                        },
                        f,
                        indent=2,
                    )
                print(
                    f"[IOD] DONE: {os.path.basename(file_i)} "
                    f"(expected {expected_rows}, csv={counts.get('csv', 0)}, pq={counts.get('parquet', 0)})",
                    flush=True,
                )
            else:
                print(
                    f"[IOD] PARTIAL (no marker): {os.path.basename(file_i)} "
                    f"(expected {expected_rows}, csv={counts.get('csv', 0)}, pq={counts.get('parquet', 0)})",
                    flush=True,
                )

        comm.Barrier()

    return


def run_IOD(config):
    """
    MPI stage that reads MASTER_IOD.csv, runs the IOD solver per row, and writes results
    back into MASTER_IOD.csv (resumable via per-row .done markers).

    UPDATED for EMS-aware MASTER schema:
      - INDEX_USED replaces MIN_NONNEGATIVE
      - Includes VALUES_IDX_EMS and OCCLUDED_BY_EMS in the detection/context block
      - Robust UID fallback uses INDEX_USED
    """

    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    viz_flag = bool(config.get("visualization_flag", 0))

    # Paths
    iod_dir = util._iod_dir(config)
    top_dir = os.path.abspath(config['top_dir'])
    master_fn = os.path.join(top_dir, "MASTER_IOD.csv")

    if rank == 0 and not os.path.exists(master_fn):
        print(f"[Stage: IOD Solve] No MASTER_IOD.csv at {master_fn} → skip")
    comm.Barrier()
    if not os.path.exists(master_fn):
        return

    num_sc = int(config["num_spacecraft"])

    # Rank 0: inspect master; broadcast row count / columns
    if rank == 0:
        try:
            _head = pd.read_csv(master_fn, nrows=1)
            n_rows = sum(1 for _ in open(master_fn, "r", encoding="utf-8")) - 1
            cols = list(_head.columns)
            print(f"[Stage: IOD Solve] MASTER rows = {n_rows}")
        except Exception as e:
            print(f"[Stage: IOD Solve] Failed to inspect MASTER: {e}")
            n_rows, cols = 0, None
    else:
        n_rows, cols = 0, None

    n_rows = comm.bcast(n_rows, root=0)
    if n_rows <= 0:
        return
    cols = comm.bcast(cols, root=0)

    # Round-robin assignment (one run per row)
    my_indices = list(range(n_rows))[rank::size]

    # Resume markers (per MASTER row)
    stage3_done_dir = os.path.join(iod_dir, "iod_stage3_done")
    if rank == 0:
        os.makedirs(stage3_done_dir, exist_ok=True)
    comm.Barrier()

    def row_uid_from_saved_as(saved_as_str: str):
        """
        Derive a stable UID from IOD_DATA_SAVED_AS if possible (preferred).
        Returns None if empty/unusable.
        """
        s = str(saved_as_str or "")
        if not s.strip():
            return None
        first = s.split(";")[0].strip()
        if not first:
            return None
        return os.path.splitext(os.path.basename(first))[0]

    def fallback_uid_from_row(row, row_index: int):
        """
        Fallback UID if IOD_DATA_SAVED_AS is empty.
        Uses stable fields in MASTER.
        """
        try:
            a = row.get("ID_AST", "")
            did = int(row.get("DETECTING_SC_ID", -1))
            idx0 = int(row.get("INDEX_USED", -1))  # <-- UPDATED
            if str(a).strip() and did >= 0 and idx0 >= 0:
                return f"minimoon-{a}_sc-{did}_index-{idx0}"
        except Exception:
            pass
        return f"rowidx_{row_index}"

    def done_marker_path(uid: str):
        return os.path.join(stage3_done_dir, f"{uid}.done")

    def is_done(uid: str) -> bool:
        return os.path.exists(done_marker_path(uid))

    def serialize_vec(v):
        """1D array-like -> 'v0,v1,...' """
        return ",".join(f"{float(x):.16g}" for x in np.asarray(v).ravel())

    # Load MASTER once for reading (workers)
    df_master = pd.read_csv(master_fn)

    # -----------------------------
    # Column order spec (final write order)
    # -----------------------------
    helio_sc_cols = [f"HELIO_SC_{i}(kms)" for i in range(1, num_sc + 1)]
    epoch_sc_cols = [f"EPOCH_SC_{i}(jdtdb)" for i in range(1, num_sc + 1)]
    boresight_cols = [f"BORESIGHT_SC_{i}_GEO_SECR" for i in range(1, num_sc + 1)]

    detection_block = (
            [
                "ID_AST",
                "EPOCH_AST(jdtdb)",
                "HELIO_AST(kms)",
                "DETECTING_SC_ID",
                "VALUES_IDX",
                "VALUES_IDX_EMS",  # <-- NEW
                "TOTAL_LENGTH",
                "SPACECRAFT_1_INI_POS(km)",
                "INDEX_USED",  # <-- UPDATED (was MIN_NONNEGATIVE)
                "OCCLUDED_BY_EMS",  # <-- NEW
            ]
            + epoch_sc_cols
            + helio_sc_cols
            + boresight_cols
    )

    # Keep IOD_DATA_SAVED_AS immediately after context block
    saved_as_col = ["IOD_DATA_SAVED_AS"]

    # Parameter block (flattened; TIME_DELTA -> TIME_DELTA_DAYS)
    param_block = [
        "NUMBER_OF_OBSERVATIONS",
        "TIME_DELTA_DAYS",
        "TOTAL_POINTS",
        "SAMPLING_METHOD",
        "LAYER_RATIOS",
        "INPUT_RANGE",
        "HIDDEN_DIMENSION",
        "PHYSICS_WEIGHT",
        "LAMBDA_DIST",
        "LAMBDA_DIST",  # (keep if you actually use it twice; otherwise remove)
        "WEIGHT_SCALE_FACTOR",
        "NUMBER_OF_ITERATIONS",
        "TEMPERATURE",
        "X_TOLERANCE",
        "F_TOLERANCE",
        "MAX_FUNCTION_EVAL",
        "MAX_ITERATiONS",
        "G_TOLERANCE",
        "MIN_RHO",
        "MAX_RHO",
        "MIN_RHO_DOT",
        "MAX_RHO_DOT",
        "DELTA_RHO",
        "DELTA_RHO_DOT",
    ]

    # Metrics / outputs block
    metrics_block = [
        "POS_RMSE",
        "VEL_RMSE",
        "COMPUTATION_TIME_SEC",
        "OPTIMAL_BH_ITERATION",
        "IOD_RESULT_SAVED_AS",
        "IOD_FINAL_STATE",
    ]

    # Internal keys we never write to MASTER
    internal_skip_keys = {"_row_index", "MASTER_UID", "MASTER_ROW_INDEX", "FILE_USED"}

    # Each rank collects updates like: {"_row_index": m_idx, ..., "IOD_RESULT_SAVED_AS": ...}
    updates = []
    processed = skipped = errors = 0

    for m_idx in my_indices:
        row = df_master.iloc[m_idx]

        saved_as = str(row.get("IOD_DATA_SAVED_AS", "") or "")
        master_uid = row_uid_from_saved_as(saved_as)
        if not master_uid:
            master_uid = fallback_uid_from_row(row, m_idx)

        # resume-skip (per row)
        if is_done(master_uid):
            skipped += 1
            continue

        try:
            # ------------------- YOUR IOD PIPELINE (single run) -------------------
            dynamics, orbit, observer, optimizer = (
                config["dynamics"],
                config["orbit"],
                config["observer"],
                config["optimizer"],
            )

            if dynamics == "NBD" and observer == "SPACE" and optimizer == "CONSTRAINED_BASIN_HOPPING":
                import PIELM_basinhopping_w_range_nbody as pielm_ctsn
                import astropy.units as u

                # Fixed parameters you provided
                m_2 = config['M2']
                m_12 = (1.0 - m_2) / 2.0

                parameters = {
                    "NUMBER_OF_OBSERVATIONS": config["NUMBER_OF_OBSERVATIONS"],
                    "TIME_DELTA": config['TIME_DELTA'] * u.day,
                    "TOTAL_POINTS": config['TOTAL_POINTS'],
                    "SAMPLING_METHOD": config['SAMPLING_METHOD'],
                    "LAYER_RATIOS": [(0.0, m_12), (m_12, m_12 + m_2), (m_12 + m_2, 1.0)],
                    "INPUT_RANGE": config['INPUT_RANGE'],
                    "HIDDEN_DIMENSION": config['HIDDEN_DIMENSION'],
                    "PHYSICS_WEIGHT": config['PHYSICS_WEIGHT'],
                    "LAMBDA_DIST": config['LAMBDA_DIST'],
                    "WEIGHT_SCALE_FACTOR": config['WEIGHT_SCALE_FACTOR'],
                    "NUMBER_OF_ITERATIONS": config['NUMBER_OF_ITERATIONS'],
                    "TEMPERATURE": config['TEMPERATURE'],
                    "X_TOLERANCE": config['X_TOLERANCE'],
                    "F_TOLERANCE": config['F_TOLERANCE'],
                    "MAX_FUNCTION_EVAL": config['MAX_FUNCTION_EVAL'],
                    "MAX_ITERATiONS": config['MAX_ITERATiONS'],
                    "G_TOLERANCE": config['G_TOLERANCE'],
                    "MIN_RHO": config['MIN_RHO'],
                    "MAX_RHO": config['MAX_RHO'],
                    "MIN_RHO_DOT": config['MIN_RHO_DOT'],
                    "MAX_RHO_DOT": config['MAX_RHO_DOT'],
                    "DELTA_RHO": config['DELTA_RHO'],
                    "DELTA_RHO_DOT": config['DELTA_RHO_DOT'],
                }

                if viz_flag:
                    config["lambda"] = parameters["TIME_DELTA"]

                # Build data from per-row IOD file referenced in MASTER row
                data = pielm_ctsn.generate_data(config, parameters, master_row=row)

                (
                    results,
                    positions,
                    velocities,
                    nlls_start,
                    final_pos,
                    final_vel,
                    true_pos,
                    true_vel,
                    epochs,
                    comp_time,
                    optimal_bh,
                ) = pielm_ctsn.run(data, config, parameters)

            else:
                raise NotImplementedError("Add other solver branches as in your code.")

            # ------- Compute metrics & write unique per-row result file -------
            out_dir = os.path.join(iod_dir, config["error_file_dir"])
            os.makedirs(out_dir, exist_ok=True)

            # Unique filename per MASTER row (avoid overwrite)
            base_name = f"{master_uid}__{config['dynamics']}_{config['orbit']}_{config['observer']}_{config['optimizer']}.csv"
            file_name = base_name
            file_path = os.path.join(out_dir, file_name)
            ctr = 1
            while os.path.exists(file_path):
                file_name = (
                    f"{master_uid}__{config['dynamics']}_{config['orbit']}_{config['observer']}_{config['optimizer']}__{ctr}.csv"
                )
                file_path = os.path.join(out_dir, file_name)
                ctr += 1

            rmse_df = util.generate_iod_file(file_path, final_pos, final_vel, true_pos, true_vel, epochs)

            # Position RMSE (Euclidean)
            pos_err_sq = np.sum(
                (rmse_df[["IOD_X", "IOD_Y", "IOD_Z"]].values - rmse_df[["TRUE_X", "TRUE_Y", "TRUE_Z"]].values) ** 2,
                axis=1,
            )
            pos_rmse = float(np.sqrt(np.mean(pos_err_sq)))

            # Velocity RMSE (Euclidean)
            vel_err_sq = np.sum(
                (rmse_df[["IOD_VX", "IOD_VY", "IOD_VZ"]].values - rmse_df[["TRUE_VX", "TRUE_VY", "TRUE_VZ"]].values)
                ** 2,
                axis=1,
            )
            vel_rmse = float(np.sqrt(np.mean(vel_err_sq)))

            try:
                final_xyz = np.asarray(final_pos[0][-2, :], dtype=float).reshape(3, )
                final_vxyz = np.asarray(final_vel[0][-2, :], dtype=float).reshape(3, )
                final_state = np.concatenate([final_xyz, final_vxyz]).tolist()
            except Exception:
                final_state = None

            # ---- Flatten parameters to columns (TIME_DELTA → TIME_DELTA_DAYS) ----
            def params_to_update(parameters):
                import astropy.units as u

                upd = {}
                for k, v in parameters.items():
                    if isinstance(v, u.Quantity):
                        if k == "TIME_DELTA":
                            upd["TIME_DELTA_DAYS"] = v.to(u.day).value
                        else:
                            upd[k] = v.to_base_units().value
                        continue
                    if isinstance(v, (np.floating, np.integer)):
                        upd[k] = v.item()
                        continue
                    if isinstance(v, (int, float)):
                        upd[k] = v
                        continue
                    if isinstance(v, (list, tuple, np.ndarray)):
                        if k == "LAYER_RATIOS":
                            upd[k] = v[1][1] - v[1][0]
                        else:
                            upd[k] = json.dumps(v if not isinstance(v, np.ndarray) else v.tolist())
                        continue
                    upd[k] = str(v)
                return upd

            ser_final = serialize_vec(final_state) if final_state is not None else ""

            upd = {
                "_row_index": int(m_idx),  # internal key for placement
                "POS_RMSE": float(pos_rmse),
                "VEL_RMSE": float(vel_rmse),
                "COMPUTATION_TIME_SEC": float(comp_time),
                "OPTIMAL_BH_ITERATION": float(optimal_bh),
                "IOD_RESULT_SAVED_AS": str(file_name),
                "IOD_FINAL_STATE": ser_final,
            }
            upd.update(params_to_update(parameters))

            updates.append(upd)
            processed += 1

            del (
                results,
                positions,
                velocities,
                nlls_start,
                final_pos,
                final_vel,
                true_pos,
                true_vel,
                epochs,
                rmse_df,
                data,
            )
            gc.collect()

        except Exception:
            # Do not mark .done here; committing happens only after master write
            errors += 1
            continue

    # ===== Gather updates to rank 0, write MASTER in-place with order, then commit markers =====
    gathered = comm.gather(updates, root=0)
    committed_uids = None  # list to broadcast

    if rank == 0:
        all_updates = [f for chunk in gathered for f in chunk]

        if all_updates:
            # Reload master to minimize race
            df = pd.read_csv(master_fn)

            # Ensure columns exist (avoid adding internal skip keys)
            new_cols = set().union(*(set(d.keys()) for d in all_updates)) - set(internal_skip_keys)
            missing = [c for c in new_cols if c not in df.columns]

            for c in missing:
                # Heuristic: initialize string-ish columns as empty strings, else NaN
                stringish = (
                        ("SAVED_AS" in c)
                        or ("(kms)" in c)
                        or ("BORESIGHT" in c)
                        or ("VALUES" in c)
                        or ("_INI_POS" in c)
                        or (c in ("SAMPLING_METHOD", "INPUT_RANGE", "IOD_RESULT_SAVED_AS"))
                )
                df[c] = "" if stringish else np.nan

            # Apply updates row-by-row
            for upd in all_updates:
                ri = upd["_row_index"]
                for k, v in upd.items():
                    if k in internal_skip_keys:
                        continue
                    df.at[ri, k] = v

            # Column reordering:
            current_cols = list(df.columns)

            ordered = [c for c in detection_block if c in current_cols]
            ordered += [c for c in saved_as_col if c in current_cols]
            ordered += [c for c in param_block if c in current_cols]
            ordered += [c for c in metrics_block if c in current_cols]

            leftovers = [c for c in current_cols if c not in set(ordered)]
            leftovers = [c for c in leftovers if c not in internal_skip_keys]

            final_cols = ordered + leftovers
            df = df[final_cols]

            # Atomic write-back
            tmp = master_fn + ".tmp"
            df.to_csv(tmp, index=False)
            os.replace(tmp, master_fn)

            # After successful write, collect committed UIDs for marker creation
            committed_uids = []
            for upd in all_updates:
                ri = upd["_row_index"]
                row_now = df.iloc[int(ri)]
                saved_as_val = row_now.get("IOD_DATA_SAVED_AS", "")
                uid = row_uid_from_saved_as(saved_as_val)
                if not uid:
                    uid = fallback_uid_from_row(row_now, int(ri))
                committed_uids.append(uid)
        else:
            committed_uids = []

    # Broadcast committed list; workers create .done files now (commit-after-write)
    committed_uids = comm.bcast(committed_uids, root=0)
    for uid in committed_uids:
        try:
            marker_path = done_marker_path(uid)
            with open(marker_path, "w") as f:
                json.dump({"uid": uid, "status": "ok"}, f)
        except Exception:
            pass

    if rank == 0:
        print(
            f"[Stage: IOD Solve] updated_rows={len(committed_uids)}, processed={processed}, skipped={skipped}, errors={errors}"
        )

    comm.Barrier()
    return


def run_IOD_hyperparameter(config):
    """
    MPI hyperparameter sweep for the PIELM IOD stage.

    Tasks are distributed as the Cartesian product of:
        rows in MASTER_IOD_TUNING.csv  x  hyperparameter combinations  x  trials_per_combo

    The tuning/test master files are created/reused according to config["IOD_HYPERPARAMETER"].
    This function writes rank-local metadata CSVs and merges them on rank 0. It does not update
    MASTER_IOD.csv, so tuning/test separation is preserved.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    hp_cfg = config.get("IOD_HYPERPARAMETER", {}) or {}
    subset_cfg = hp_cfg.get("subset", {}) or {}
    output_cfg = hp_cfg.get("output", {}) or {}
    sweep_cfg = hp_cfg.get("sweep", {}) or {}
    metrics_cfg = hp_cfg.get("metrics", {}) or {}

    iod_dir = util._iod_dir(config)
    full_master_name = subset_cfg.get("source_master_file", "MASTER_IOD.csv")
    source_master_path = os.path.join(config['top_dir'], full_master_name)

    tuning_master_name = subset_cfg.get("tuning_master_file", subset_cfg.get("master_subset_file", "MASTER_IOD_TUNING.csv"))
    test_master_name = subset_cfg.get("test_master_file", "MASTER_IOD_TEST.csv")
    tuning_master_path = os.path.join(iod_dir, tuning_master_name)
    test_master_path = os.path.join(iod_dir, test_master_name)

    mode = str(subset_cfg.get("mode", "explicit_master")).lower()
    reuse_existing = bool(subset_cfg.get("reuse_existing_subset_files", True))
    random_seed = int(subset_cfg.get("random_seed", config.get("seed", 12345)))
    n_tuning_cases = subset_cfg.get("n_tuning_cases", None)
    test_mode = str(subset_cfg.get("test_mode", "complement")).lower()
    n_test_cases = subset_cfg.get("n_test_cases", None)

    if rank == 0:
        os.makedirs(iod_dir, exist_ok=True)
        if mode == "sample":
            have_tuning = os.path.exists(tuning_master_path)
            have_test = os.path.exists(test_master_path)
            if reuse_existing and have_tuning and have_test:
                print(f"[IOD Hyperparameter] Reusing existing subset masters: {tuning_master_path}, {test_master_path}", flush=True)
            else:
                if not os.path.exists(source_master_path):
                    raise FileNotFoundError(f"Source MASTER file not found: {source_master_path}")
                df_all = pd.read_csv(source_master_path)
                if len(df_all) == 0:
                    raise ValueError(f"Source MASTER file has no rows: {source_master_path}")

                if n_tuning_cases is None:
                    raise ValueError("IOD_HYPERPARAMETER.subset.n_tuning_cases must be set when mode='sample'.")
                n_tuning = min(int(n_tuning_cases), len(df_all))
                rng = np.random.default_rng(random_seed)
                all_idx = np.arange(len(df_all))
                tuning_idx = np.sort(rng.choice(all_idx, size=n_tuning, replace=False))
                remaining_idx = np.setdiff1d(all_idx, tuning_idx, assume_unique=False)

                if test_mode == "complement":
                    test_idx = remaining_idx
                elif test_mode in ("sample", "number", "n"):
                    if n_test_cases is None:
                        raise ValueError("IOD_HYPERPARAMETER.subset.n_test_cases must be set when test_mode='sample'.")
                    n_test = min(int(n_test_cases), len(remaining_idx))
                    test_idx = np.sort(rng.choice(remaining_idx, size=n_test, replace=False)) if n_test > 0 else np.array([], dtype=int)
                else:
                    raise ValueError(f"Unsupported test_mode: {test_mode}. Use 'complement' or 'sample'.")

                df_all.iloc[tuning_idx].to_csv(tuning_master_path, index=False)
                df_all.iloc[test_idx].to_csv(test_master_path, index=False)
                print(
                    f"[IOD Hyperparameter] Wrote tuning/test masters: "
                    f"tuning={len(tuning_idx)} rows -> {tuning_master_path}; "
                    f"test={len(test_idx)} rows -> {test_master_path}",
                    flush=True,
                )
        elif mode in ("explicit_master", "explicit"):
            if not os.path.exists(tuning_master_path):
                raise FileNotFoundError(
                    f"Tuning master file not found: {tuning_master_path}. "
                    "Use subset.mode='sample' to create it automatically."
                )
            if not os.path.exists(test_master_path):
                print(f"[IOD Hyperparameter] Note: test master not found yet: {test_master_path}", flush=True)
        else:
            raise ValueError(f"Unsupported IOD_HYPERPARAMETER.subset.mode: {mode}")
    comm.Barrier()

    if not os.path.exists(tuning_master_path):
        if rank == 0:
            print(f"[IOD Hyperparameter] No tuning master at {tuning_master_path}; skipping.", flush=True)
        comm.Barrier()
        return

    # -------------------------
    # Build hyperparameter grid
    # -------------------------
    default_sweep_keys = [
        "M2", "TIME_DELTA", "TOTAL_POINTS", "HIDDEN_DIMENSION", "PHYSICS_WEIGHT", "LAMBDA_DIST",
        "WEIGHT_SCALE_FACTOR", "NUMBER_OF_ITERATIONS", "TEMPERATURE", "MIN_RHO", "MAX_RHO",
        "MIN_RHO_DOT", "MAX_RHO_DOT", "DELTA_RHO", "DELTA_RHO_DOT",
    ]
    # Support lowercase legacy spelling while writing uppercase M2 in outputs.
    if "m2" in sweep_cfg and "M2" not in sweep_cfg:
        sweep_cfg["M2"] = sweep_cfg["m2"]

    def _as_list(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return list(v)
        return [v]

    sweep_keys = [k for k in default_sweep_keys if k in sweep_cfg]
    if not sweep_keys:
        # A single combo using config values only.
        sweep_keys = []
        sweep_values = [()]
    else:
        sweep_values = list(itertools.product(*[_as_list(sweep_cfg[k]) for k in sweep_keys]))

    trials_per_combo = int(hp_cfg.get("trials_per_combo", 1))
    if trials_per_combo < 1:
        trials_per_combo = 1

    # -------------------------
    # Output setup
    # -------------------------
    out_dir_cfg = output_cfg.get("output_dir", "iod_hyperparameter_results")
    out_dir = out_dir_cfg if os.path.isabs(str(out_dir_cfg)) else os.path.join(iod_dir, str(out_dir_cfg))
    per_run_dir = os.path.join(out_dir, output_cfg.get("per_run_dir", "per_run_iod_files"))
    metadata_file = output_cfg.get("metadata_file", "IOD_HYPERPARAMETER_RESULTS.csv")
    final_meta_path = os.path.join(out_dir, metadata_file)
    rank_meta_path = os.path.join(out_dir, f"{os.path.splitext(metadata_file)[0]}_rank{rank}.csv")
    save_per_run = bool(output_cfg.get("save_per_run_iod_files", True))
    retry_failed = bool(output_cfg.get("retry_failed", False))

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(per_run_dir, exist_ok=True)
    comm.Barrier()

    # -------------------------
    # Load tuning master on all ranks
    # -------------------------
    df_tuning = pd.read_csv(tuning_master_path)
    n_rows = len(df_tuning)
    if rank == 0:
        print(
            f"[IOD Hyperparameter] tuning_rows={n_rows}, combos={len(sweep_values)}, "
            f"trials_per_combo={trials_per_combo}, mpi_ranks={size}",
            flush=True,
        )
    if n_rows == 0:
        comm.Barrier()
        return

    # -------------------------
    # Helper functions
    # -------------------------
    def row_uid_from_saved_as(saved_as_str: str):
        s = str(saved_as_str or "")
        if not s.strip():
            return None
        first = s.split(";")[0].strip()
        if not first:
            return None
        return os.path.splitext(os.path.basename(first))[0]

    def fallback_uid_from_row(row, row_index: int):
        try:
            a = row.get("ID_AST", "")
            did = int(row.get("DETECTING_SC_ID", -1))
            idx0 = int(row.get("INDEX_USED", -1))
            if str(a).strip() and did >= 0 and idx0 >= 0:
                return f"minimoon-{a}_sc-{did}_index-{idx0}"
        except Exception:
            pass
        return f"tuningrow_{row_index}"

    def combo_dict_from_tuple(combo_tuple):
        d = {}
        for k, v in zip(sweep_keys, combo_tuple):
            if k == "m2":
                d["M2"] = v
            else:
                d[k] = v
        return d

    def hp_hash(combo_dict, trial_index):
        payload = {k: combo_dict[k] for k in sorted(combo_dict.keys())}
        payload["TRIAL_INDEX"] = int(trial_index)
        s = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

    def task_seed(base_seed, master_uid, hyper_hash, trial_index):
        raw = f"{base_seed}|{master_uid}|{hyper_hash}|{trial_index}"
        return int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:8], 16)

    def build_parameters(config_task, combo_dict, trial_index):
        import astropy.units as u
        p = {}
        # Required/fixed defaults come from config, unless swept.
        def get_param(key, default=None):
            return combo_dict.get(key, config_task.get(key, default))

        m2_val = get_param("M2", config_task.get("m2", 0.1))
        m2_val = float(m2_val)
        m12 = (1.0 - m2_val) / 2.0

        p["NUMBER_OF_OBSERVATIONS"] = int(get_param("NUMBER_OF_OBSERVATIONS", config_task.get("NUMBER_OF_OBSERVATIONS", 16)))
        p["TIME_DELTA"] = float(get_param("TIME_DELTA", config_task["TIME_DELTA"])) * u.day
        p["TOTAL_POINTS"] = int(get_param("TOTAL_POINTS", config_task["TOTAL_POINTS"]))
        p["SAMPLING_METHOD"] = get_param("SAMPLING_METHOD", config_task.get("SAMPLING_METHOD", "uniform"))
        p["LAYER_RATIOS"] = [(0.0, m12), (m12, m12 + m2_val), (m12 + m2_val, 1.0)]
        p["INPUT_RANGE"] = get_param("INPUT_RANGE", config_task.get("INPUT_RANGE", (-1, 1)))
        p["HIDDEN_DIMENSION"] = int(get_param("HIDDEN_DIMENSION", config_task["HIDDEN_DIMENSION"]))
        p["PHYSICS_WEIGHT"] = float(get_param("PHYSICS_WEIGHT", config_task["PHYSICS_WEIGHT"]))
        p["LAMBDA_DIST"] = float(get_param("LAMBDA_DIST", config_task["LAMBDA_DIST"]))
        p["WEIGHT_SCALE_FACTOR"] = float(get_param("WEIGHT_SCALE_FACTOR", config_task["WEIGHT_SCALE_FACTOR"]))
        p["NUMBER_OF_ITERATIONS"] = int(get_param("NUMBER_OF_ITERATIONS", config_task["NUMBER_OF_ITERATIONS"]))
        p["TEMPERATURE"] = float(get_param("TEMPERATURE", config_task["TEMPERATURE"]))
        p["X_TOLERANCE"] = float(get_param("X_TOLERANCE", config_task["X_TOLERANCE"]))
        p["F_TOLERANCE"] = float(get_param("F_TOLERANCE", config_task["F_TOLERANCE"]))
        p["MAX_FUNCTION_EVAL"] = int(get_param("MAX_FUNCTION_EVAL", config_task["MAX_FUNCTION_EVAL"]))
        p["MAX_ITERATiONS"] = int(get_param("MAX_ITERATiONS", config_task["MAX_ITERATiONS"]))
        p["G_TOLERANCE"] = float(get_param("G_TOLERANCE", config_task["G_TOLERANCE"]))
        p["MIN_RHO"] = float(get_param("MIN_RHO", config_task["MIN_RHO"]))
        p["MAX_RHO"] = float(get_param("MAX_RHO", config_task["MAX_RHO"]))
        p["MIN_RHO_DOT"] = float(get_param("MIN_RHO_DOT", config_task["MIN_RHO_DOT"]))
        p["MAX_RHO_DOT"] = float(get_param("MAX_RHO_DOT", config_task["MAX_RHO_DOT"]))
        p["DELTA_RHO"] = float(get_param("DELTA_RHO", config_task["DELTA_RHO"]))
        p["DELTA_RHO_DOT"] = float(get_param("DELTA_RHO_DOT", config_task["DELTA_RHO_DOT"]))
        p["RUN_NUMBER"] = int(trial_index)
        p["M2"] = m2_val
        return p

    def params_to_flat_row(parameters):
        import astropy.units as u
        row = {}
        for k, v in parameters.items():
            if k == "TIME_DELTA" and isinstance(v, u.Quantity):
                row["TIME_DELTA_DAYS"] = float(v.to(u.day).value)
            elif isinstance(v, u.Quantity):
                row[k] = float(v.to_base_units().value)
            elif isinstance(v, (np.floating, np.integer)):
                row[k] = v.item()
            elif isinstance(v, (int, float, str)):
                row[k] = v
            elif isinstance(v, (list, tuple, np.ndarray)):
                if k == "LAYER_RATIOS":
                    try:
                        row[k] = float(v[1][1] - v[1][0])
                    except Exception:
                        row[k] = json.dumps(v, default=str)
                else:
                    row[k] = json.dumps(v.tolist() if isinstance(v, np.ndarray) else v, default=str)
            else:
                row[k] = str(v)
        return row

    def serialize_vec(v):
        try:
            return ",".join(f"{float(x):.16g}" for x in np.asarray(v).ravel())
        except Exception:
            return ""

    def wrap_to_pi(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _to_numpy(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _sigma_radec_rad(data_tuple, config_task):
        # data_tuple[7], data_tuple[8] are degrees when ADD_NOISE == 1, otherwise 0.0 in current PIELM.
        sigma_ra_deg = data_tuple[7]
        sigma_dec_deg = data_tuple[8]
        try:
            sigma_ra_deg = float(_to_numpy(sigma_ra_deg))
            sigma_dec_deg = float(_to_numpy(sigma_dec_deg))
        except Exception:
            sigma_ra_deg = 0.0
            sigma_dec_deg = 0.0
        if sigma_ra_deg > 0 and sigma_dec_deg > 0:
            return np.deg2rad(sigma_ra_deg), np.deg2rad(sigma_dec_deg)

        # Fallback to config values in mas, including pointing, matching PIELM generate_data().
        try:
            mas_to_degree = float(config_task["MAS_TO_DEGREE"])
            sigma_ra_mas = math.sqrt(float(config_task.get("sigma_ra", 0.0)) ** 2 + float(config_task.get("sigma_pointing", 0.0)) ** 2)
            sigma_dec_mas = math.sqrt(float(config_task.get("sigma_dec", 0.0)) ** 2 + float(config_task.get("sigma_pointing", 0.0)) ** 2)
            sigma_ra_rad = np.deg2rad(sigma_ra_mas / mas_to_degree)
            sigma_dec_rad = np.deg2rad(sigma_dec_mas / mas_to_degree)
            if sigma_ra_rad > 0 and sigma_dec_rad > 0:
                return sigma_ra_rad, sigma_dec_rad
        except Exception:
            pass

        # Last fallback: explicit metrics config in radians.
        return float(metrics_cfg.get("sigma_ra_rad", np.nan)), float(metrics_cfg.get("sigma_dec_rad", np.nan))

    def compute_extra_metrics(rmse_df, final_pos, final_vel, true_pos, true_vel, data_tuple, config_task):
        metrics = {}
        # Euclidean RMSEs from generated trajectory file.
        try:
            iod_xyz = rmse_df[["IOD_X", "IOD_Y", "IOD_Z"]].values.astype(float)
            tru_xyz = rmse_df[["TRUE_X", "TRUE_Y", "TRUE_Z"]].values.astype(float)
            iod_v = rmse_df[["IOD_VX", "IOD_VY", "IOD_VZ"]].values.astype(float)
            tru_v = rmse_df[["TRUE_VX", "TRUE_VY", "TRUE_VZ"]].values.astype(float)
            metrics["POS_RMSE"] = float(np.sqrt(np.mean(np.sum((iod_xyz - tru_xyz) ** 2, axis=1))))
            metrics["VEL_RMSE"] = float(np.sqrt(np.mean(np.sum((iod_v - tru_v) ** 2, axis=1))))
        except Exception:
            metrics["POS_RMSE"] = np.nan
            metrics["VEL_RMSE"] = np.nan

        final_state_index = int(metrics_cfg.get("final_state_index", -2))
        try:
            r_est_f = np.asarray(final_pos[0][final_state_index, :], dtype=float).reshape(3)
            v_est_f = np.asarray(final_vel[0][final_state_index, :], dtype=float).reshape(3)
            r_true_f = np.asarray(true_pos[-1, :], dtype=float).reshape(3)
            v_true_f = np.asarray(true_vel[-1, :], dtype=float).reshape(3)
            metrics["POS_FINAL_ERROR"] = float(np.linalg.norm(r_est_f - r_true_f))
            metrics["VEL_FINAL_ERROR"] = float(np.linalg.norm(v_est_f - v_true_f))
        except Exception:
            r_est_f = v_est_f = r_true_f = v_true_f = None
            metrics["POS_FINAL_ERROR"] = np.nan
            metrics["VEL_FINAL_ERROR"] = np.nan

        # Range/range-rate final errors, using the physical observer state used by the solver.
        try:
            obs_r = _to_numpy(data_tuple[1]).astype(float)
            obs_v = _to_numpy(data_tuple[-1]).astype(float)
            R_f = obs_r[-1, :].reshape(3)
            V_f = obs_v[-1, :].reshape(3)
            rho_est_vec = r_est_f - R_f
            rho_true_vec = r_true_f - R_f
            rho_est = float(np.linalg.norm(rho_est_vec))
            rho_true = float(np.linalg.norm(rho_true_vec))
            l_est = rho_est_vec / max(rho_est, 1e-15)
            l_true = rho_true_vec / max(rho_true, 1e-15)
            rhodot_est = float(np.dot(l_est, v_est_f - V_f))
            rhodot_true = float(np.dot(l_true, v_true_f - V_f))
            metrics["RANGE_FINAL_ERROR"] = abs(rho_est - rho_true)
            metrics["RANGE_RATE_FINAL_ERROR"] = abs(rhodot_est - rhodot_true)
            metrics["RANGE_FINAL_EST"] = rho_est
            metrics["RANGE_FINAL_TRUE"] = rho_true
            metrics["RANGE_RATE_FINAL_EST"] = rhodot_est
            metrics["RANGE_RATE_FINAL_TRUE"] = rhodot_true
        except Exception:
            metrics["RANGE_FINAL_ERROR"] = np.nan
            metrics["RANGE_RATE_FINAL_ERROR"] = np.nan
            metrics["RANGE_FINAL_EST"] = np.nan
            metrics["RANGE_FINAL_TRUE"] = np.nan
            metrics["RANGE_RATE_FINAL_EST"] = np.nan
            metrics["RANGE_RATE_FINAL_TRUE"] = np.nan

        # RA/Dec NRMS against the actual measurement tensors used by the solver.
        try:
            y_obs = data_tuple[0]
            sin_ra = _to_numpy(y_obs[0]).astype(float)
            cos_ra = _to_numpy(y_obs[1]).astype(float)
            sin_dec = _to_numpy(y_obs[2]).astype(float)
            ra_meas = np.arctan2(sin_ra, cos_ra)
            dec_meas = np.arcsin(np.clip(sin_dec, -1.0, 1.0))

            obs_r = _to_numpy(data_tuple[1]).astype(float)
            obs_epochs = data_tuple[2]
            # Estimated trajectory over observation arc is final_pos[0][1:-1], aligned with returned `epochs`.
            est_arc = np.asarray(final_pos[0][1:-1, :], dtype=float)
            # Interpolate estimated position to the actual observation epochs.
            try:
                est_epoch_time = data_tuple["__epochs_for_est_arc__"]  # not used currently; kept for future compatibility
            except Exception:
                est_epoch_time = None
            # Reconstruct the epoch grid used by run(): returned epochs are uniformly spaced over obs arc.
            n_est = est_arc.shape[0]
            t_obs_sec = np.array([(t - obs_epochs[0]).to_value("s") for t in obs_epochs], dtype=float)
            t_est_sec = np.linspace(0.0, t_obs_sec[-1], n_est)
            est_at_obs = np.column_stack([np.interp(t_obs_sec, t_est_sec, est_arc[:, j]) for j in range(3)])

            rel = est_at_obs - obs_r
            rel_norm = np.linalg.norm(rel, axis=1)
            ra_pred = np.arctan2(rel[:, 1], rel[:, 0])
            dec_pred = np.arcsin(np.clip(rel[:, 2] / np.maximum(rel_norm, 1e-15), -1.0, 1.0))
            dra = wrap_to_pi(ra_pred - ra_meas)
            ddec = dec_pred - dec_meas
            sigma_ra_rad, sigma_dec_rad = _sigma_radec_rad(data_tuple, config_task)
            metrics["SIGMA_RA_RAD"] = sigma_ra_rad
            metrics["SIGMA_DEC_RAD"] = sigma_dec_rad
            if np.isfinite(sigma_ra_rad) and np.isfinite(sigma_dec_rad) and sigma_ra_rad > 0 and sigma_dec_rad > 0:
                metrics["RA_DEC_NRMS"] = float(np.sqrt(np.mean((dra / sigma_ra_rad) ** 2 + (ddec / sigma_dec_rad) ** 2) / 2.0))
            else:
                metrics["RA_DEC_NRMS"] = np.nan
            metrics["RA_RMS_RAD"] = float(np.sqrt(np.mean(dra ** 2)))
            metrics["DEC_RMS_RAD"] = float(np.sqrt(np.mean(ddec ** 2)))
        except Exception:
            metrics["RA_DEC_NRMS"] = np.nan
            metrics["RA_RMS_RAD"] = np.nan
            metrics["DEC_RMS_RAD"] = np.nan
            metrics["SIGMA_RA_RAD"] = np.nan
            metrics["SIGMA_DEC_RAD"] = np.nan

        try:
            metrics["IOD_FINAL_STATE"] = serialize_vec(np.concatenate([r_est_f, v_est_f]))
        except Exception:
            metrics["IOD_FINAL_STATE"] = ""
        return metrics

    # Metadata header: fixed leading identifiers, parameters, metrics, status.
    base_header = [
        "TASK_ID", "MASTER_UID", "MASTER_ROW_INDEX", "TRIAL_INDEX", "HYPERPARAMETER_HASH",
        "ID_AST", "DETECTING_SC_ID", "INDEX_USED", "IOD_DATA_SAVED_AS", "FILE_USED",
    ]
    param_header = [
        "M2", "NUMBER_OF_OBSERVATIONS", "TIME_DELTA_DAYS", "TOTAL_POINTS", "SAMPLING_METHOD",
        "LAYER_RATIOS", "INPUT_RANGE", "HIDDEN_DIMENSION", "PHYSICS_WEIGHT", "LAMBDA_DIST",
        "WEIGHT_SCALE_FACTOR", "NUMBER_OF_ITERATIONS", "TEMPERATURE", "X_TOLERANCE", "F_TOLERANCE",
        "MAX_FUNCTION_EVAL", "MAX_ITERATiONS", "G_TOLERANCE", "MIN_RHO", "MAX_RHO", "MIN_RHO_DOT",
        "MAX_RHO_DOT", "DELTA_RHO", "DELTA_RHO_DOT", "RUN_NUMBER", "TASK_SEED",
    ]
    metric_header = [
        "POS_RMSE", "VEL_RMSE", "POS_FINAL_ERROR", "VEL_FINAL_ERROR", "RANGE_FINAL_ERROR",
        "RANGE_RATE_FINAL_ERROR", "RANGE_FINAL_EST", "RANGE_FINAL_TRUE", "RANGE_RATE_FINAL_EST",
        "RANGE_RATE_FINAL_TRUE", "RA_DEC_NRMS", "RA_RMS_RAD", "DEC_RMS_RAD", "SIGMA_RA_RAD",
        "SIGMA_DEC_RAD", "COMPUTATION_TIME_SEC", "OPTIMAL_BH_ITERATION", "IOD_RESULT_SAVED_AS",
        "IOD_FINAL_STATE", "STATUS", "ERROR_MESSAGE",
    ]
    rank_header = base_header + param_header + metric_header

    def key_of_csv_row(row):
        return str(row.get("TASK_ID", ""))

    completed = set()
    for pth in [final_meta_path, rank_meta_path]:
        if not os.path.exists(pth):
            continue
        try:
            with open(pth, "r", newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    tid = key_of_csv_row(r)
                    if not tid:
                        continue
                    status = str(r.get("STATUS", ""))
                    if (not retry_failed) or status == "OK":
                        completed.add(tid)
        except Exception:
            pass

    need_header = (not os.path.exists(rank_meta_path)) or (os.path.getsize(rank_meta_path) == 0)
    rank_fh = open(rank_meta_path, "a", newline="", encoding="utf-8")
    rank_writer = csv.DictWriter(rank_fh, fieldnames=rank_header, extrasaction="ignore")
    if need_header:
        rank_writer.writeheader()

    def write_row(row):
        full = {k: row.get(k, "") for k in rank_header}
        rank_writer.writerow(full)
        rank_fh.flush()
        os.fsync(rank_fh.fileno())

    # Stream tasks without materializing row x combo x trial globally.
    task_counter = 0
    processed = skipped = errors = 0
    total_tasks = n_rows * len(sweep_values) * trials_per_combo

    for row_idx in range(n_rows):
        row = df_tuning.iloc[row_idx]
        master_uid = row_uid_from_saved_as(row.get("IOD_DATA_SAVED_AS", "")) or fallback_uid_from_row(row, row_idx)
        for combo_tuple in sweep_values:
            combo_dict = combo_dict_from_tuple(combo_tuple)
            for trial_index in range(trials_per_combo):
                hhash = hp_hash(combo_dict, trial_index)
                task_id = f"{master_uid}__hp_{hhash}"
                my_turn = (task_counter % size == rank)
                task_counter += 1
                if not my_turn:
                    continue
                if task_id in completed:
                    skipped += 1
                    continue

                if processed % 5 == 0:
                    print(f"[rank {rank}] HP task {processed + skipped + errors + 1}; global approx {task_counter}/{total_tasks}: {task_id}", flush=True)

                base_row = {
                    "TASK_ID": task_id,
                    "MASTER_UID": master_uid,
                    "MASTER_ROW_INDEX": int(row_idx),
                    "TRIAL_INDEX": int(trial_index),
                    "HYPERPARAMETER_HASH": hhash,
                    "ID_AST": row.get("ID_AST", ""),
                    "DETECTING_SC_ID": row.get("DETECTING_SC_ID", ""),
                    "INDEX_USED": row.get("INDEX_USED", ""),
                    "IOD_DATA_SAVED_AS": row.get("IOD_DATA_SAVED_AS", ""),
                }

                try:
                    import torch
                    import astropy.units as u
                    import PIELM_basinhopping_w_range_nbody as pielm_ctsn

                    seed_i = task_seed(random_seed, master_uid, hhash, trial_index)
                    config_task = copy.deepcopy(config)
                    config_task["TASK_SEED"] = int(seed_i)
                    config_task["seed"] = int(seed_i)
                    np.random.seed(seed_i % (2**32 - 1))
                    torch.manual_seed(seed_i % (2**32 - 1))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_i % (2**32 - 1))

                    dynamics, orbit, observer, optimizer = (
                        config_task["dynamics"], config_task["orbit"], config_task["observer"], config_task["optimizer"]
                    )
                    if not (dynamics == "NBD" and observer == "SPACE" and optimizer == "CONSTRAINED_BASIN_HOPPING"):
                        raise NotImplementedError(
                            "run_IOD_hyperparameter currently supports dynamics='NBD', observer='SPACE', "
                            "optimizer='CONSTRAINED_BASIN_HOPPING'."
                        )

                    parameters = build_parameters(config_task, combo_dict, trial_index)
                    parameters["TASK_SEED"] = int(seed_i)

                    viz_flag = bool(config_task.get("visualization_flag", 0))
                    if viz_flag:
                        config_task["lambda"] = parameters["TIME_DELTA"]
                        config_task["run_idx"] = trial_index

                    data = pielm_ctsn.generate_data(config_task, parameters, master_row=row)
                    (
                        results, positions, velocities, nlls_start, final_pos, final_vel,
                        true_pos, true_vel, epochs, comp_time, optimal_bh,
                    ) = pielm_ctsn.run(data, config_task, parameters)

                    file_used = data[9]
                    if save_per_run:
                        result_name = f"{task_id}.csv"
                        result_path = os.path.join(per_run_dir, result_name)
                    else:
                        result_name = ""
                        result_path = os.path.join(per_run_dir, f"{task_id}.csv")

                    rmse_df = util.generate_iod_file(result_path, final_pos, final_vel, true_pos, true_vel, epochs)
                    extra_metrics = compute_extra_metrics(rmse_df, final_pos, final_vel, true_pos, true_vel, data, config_task)

                    out = dict(base_row)
                    out.update(params_to_flat_row(parameters))
                    out["TASK_SEED"] = int(seed_i)
                    out["FILE_USED"] = str(file_used)
                    out.update(extra_metrics)
                    out["COMPUTATION_TIME_SEC"] = float(comp_time)
                    out["OPTIMAL_BH_ITERATION"] = float(optimal_bh) if optimal_bh is not None else np.nan
                    out["IOD_RESULT_SAVED_AS"] = result_name
                    out["STATUS"] = "OK"
                    out["ERROR_MESSAGE"] = ""
                    write_row(out)
                    processed += 1

                    del results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, rmse_df, data
                    gc.collect()

                except Exception as exc:
                    err_msg = traceback.format_exc(limit=8)
                    out = dict(base_row)
                    try:
                        parameters = build_parameters(config, combo_dict, trial_index)
                        out.update(params_to_flat_row(parameters))
                    except Exception:
                        pass
                    out["STATUS"] = "ERROR"
                    out["ERROR_MESSAGE"] = str(exc).replace("\n", " ")[:2000]
                    write_row(out)
                    errors += 1
                    print(f"[rank {rank}] ERROR in {task_id}: {exc}", flush=True)
                    continue

    rank_fh.close()
    comm.Barrier()

    # Merge rank files on rank 0.
    if rank == 0:
        rank_files = sorted(glob.glob(os.path.join(out_dir, f"{os.path.splitext(metadata_file)[0]}_rank*.csv")))
        rows = []
        seen = set()
        sources = ([final_meta_path] if os.path.exists(final_meta_path) else []) + rank_files
        for src in sources:
            try:
                with open(src, "r", newline="", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    for r in rdr:
                        tid = str(r.get("TASK_ID", ""))
                        if not tid or tid in seen:
                            continue
                        rows.append(r)
                        seen.add(tid)
            except Exception as exc:
                print(f"[IOD Hyperparameter] Could not merge {src}: {exc}", flush=True)
        with open(final_meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rank_header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"[IOD Hyperparameter] MERGED {len(rows)} rows -> {final_meta_path}", flush=True)

    comm.Barrier()
    if rank == 0:
        print(f"[IOD Hyperparameter] done. rank0 processed={processed}, skipped={skipped}, errors={errors}", flush=True)
    return


def run_OD(config_global):
    """
    Stage 4: Orbit Determination (OD)

    Added diagnostics:
      - one outer-loop CSV per run in a common folder
      - optional per-run detailed CSVs:
            * inner_kf_updates.csv
            * attcoord_candidates.csv
            * optimizer_history.csv
      - per-run progress marker for resume
      - final termination row on no-detection / time-limit / step-limit / error

    Notes:
      - Existing visualization flags/blocks are preserved.
      - Outer-loop log is the primary analysis log.
      - Inner KF / att-coord / optimizer logs are optional detailed logs.
      - Resume is append-safe and replays deterministically unless you later add full state snapshots.
    """

    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Paths & MASTER
    iod_dir = util._iod_dir(config_global)
    top_dir = os.path.abspath(config_global['top_dir'])
    od_dir = os.path.join(top_dir, config_global['od_file_dir'])
    master_fn = os.path.join(top_dir, "MASTER_IOD.csv")
    if rank == 0 and not os.path.exists(master_fn):
        print(f"[Stage: OD] No MASTER_IOD.csv at {master_fn} -> skip")
    comm.Barrier()
    if not os.path.exists(master_fn):
        return

    # Rank 0: inspect MASTER and broadcast row count
    if rank == 0:
        try:
            _head = pd.read_csv(master_fn, nrows=1)
            n_rows = sum(1 for _ in open(master_fn, "r", encoding="utf-8")) - 1
            print(f"[Stage: OD] MASTER rows = {n_rows}")
        except Exception as e:
            print(f"[Stage: OD] Failed to inspect MASTER: {e}")
            n_rows = 0
    else:
        n_rows = 0

    n_rows = comm.bcast(n_rows, root=0)
    if n_rows <= 0:
        return

    # Assignment is built after OD resume/done helpers are available.
    # By default, unfinished rows can be redistributed across ranks on each restart
    # instead of preserving the original static round-robin allocation.
    my_indices = []

    # Resume markers (commit-after-write; separate dir from Stage 3)
    od_done_dir = os.path.join(od_dir, "od_stage4_od_done")
    if rank == 0:
        os.makedirs(od_done_dir, exist_ok=True)
    comm.Barrier()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def uid_from_saved_as(saved_as_str: str, fallback_idx: int):
        s = str(saved_as_str or "")
        if s.strip():
            first = s.split(";")[0].strip()
            if first:
                return os.path.splitext(os.path.basename(first))[0]
        return f"rowidx_{fallback_idx}"

    def done_marker_path(uid: str):
        return os.path.join(od_done_dir, f"{uid}.done")

    def is_done(uid: str) -> bool:
        return os.path.exists(done_marker_path(uid))

    def write_done(uid: str):
        with open(done_marker_path(uid), "w", encoding="utf-8") as f:
            f.write("done\n")

    def make_unique_filename(out_dir: str, base: str, ext: str = ".csv"):
        """Return a unique path like '<out_dir>/<base>.csv', or <base>__1.csv, ..."""
        name = f"{base}{ext}"
        path = os.path.join(out_dir, name)
        k = 1
        while os.path.exists(path):
            name = f"{base}__{k}{ext}"
            path = os.path.join(out_dir, name)
            k += 1
        return path, name

    def _fmt_vec(v, n=3, prec=3):
        if v is None:
            return ""
        v = np.asarray(v).reshape(-1)
        n = min(n, v.size)
        return "[" + ", ".join(f"{v[i]:.{prec}g}" for i in range(n)) + (", ..." if v.size > n else "") + "]"

    def _safe_norm(v):
        if v is None:
            return np.nan
        v = np.asarray(v).reshape(-1)
        return float(np.linalg.norm(v)) if v.size else np.nan

    def _safe_diag(P, n=6):
        if P is None:
            return [np.nan] * n
        try:
            P = np.asarray(P)
            if P.ndim != 2:
                return [np.nan] * n
            d = np.diag(P).reshape(-1)
            out = [np.nan] * n
            for i in range(min(n, d.size)):
                out[i] = float(d[i])
            return out
        except Exception:
            return [np.nan] * n

    def _flatten_vec(v, n_expected=None):
        try:
            if v is None:
                return [np.nan] * n_expected if n_expected is not None else []
            arr = np.asarray(v).reshape(-1)
            out = [float(x) for x in arr]
            if n_expected is not None:
                if len(out) < n_expected:
                    out += [np.nan] * (n_expected - len(out))
                else:
                    out = out[:n_expected]
            return out
        except Exception:
            return [np.nan] * n_expected if n_expected is not None else []

    def _ids_to_str(ids):
        if ids is None:
            return ""
        if isinstance(ids, (list, tuple, np.ndarray)):
            return ";".join(str(int(x)) for x in np.asarray(ids).reshape(-1))
        return str(ids)

    def _first_last_count(x):
        try:
            arr = np.asarray(x).reshape(-1)
            if arr.size == 0:
                return np.nan, np.nan, 0
            return float(arr[0]), float(arr[-1]), int(arr.size)
        except Exception:
            return np.nan, np.nan, 0

    def _best_effort_state_error(x_est, x_true):
        try:
            xe = np.asarray(x_est).reshape(-1)
            xt = np.asarray(x_true).reshape(-1)
            if xe.size < 6 or xt.size < 6:
                return np.nan, np.nan
            pos = float(np.linalg.norm(xe[:3] - xt[:3]))
            vel = float(np.linalg.norm(xe[3:6] - xt[3:6]))
            return pos, vel
        except Exception:
            return np.nan, np.nan

    def _safe_scalar(x):
        try:
            if x is None:
                return np.nan
            if isinstance(x, (float, int, np.floating, np.integer)):
                return float(x)
            arr = np.asarray(x).reshape(-1)
            if arr.size == 0:
                return np.nan
            return float(arr[0])
        except Exception:
            return np.nan

    def _append_row(csv_path, row_dict, header):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not exists:
                writer.writeheader()
            writer.writerow(row_dict)

    def _write_progress(progress_path, payload):
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _read_progress(progress_path):
        if not os.path.exists(progress_path):
            return None
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resume_last_outer_idx(outer_csv_path):
        if not os.path.exists(outer_csv_path):
            return None
        try:
            df = pd.read_csv(outer_csv_path)
            if len(df) == 0:
                return None
            return int(df["od_step_idx"].max())
        except Exception:
            return None

    def _atomic_pickle_dump(obj, path):
        """Atomically write a pickle file so a killed job does not leave a half-written checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def _safe_np_random_state():
        try:
            return np.random.get_state()
        except Exception:
            return None

    def _restore_np_random_state(state):
        if state is None:
            return
        try:
            np.random.set_state(state)
        except Exception:
            pass

    def _checkpoint_path_for(run_dir, checkpoint_name="checkpoint_state.pkl"):
        return os.path.join(run_dir, checkpoint_name)

    def _get_sc_epochs(formation):
        try:
            return [getattr(sc, "curr_sc_epoch", None) for sc in formation.spacecraft]
        except Exception:
            return None

    def _set_sc_epochs(formation, epochs):
        if epochs is None:
            return
        try:
            for sc, ep in zip(formation.spacecraft, epochs):
                sc.curr_sc_epoch = ep
        except Exception:
            pass

    def _save_od_checkpoint(
            checkpoint_path, *, uid, m_idx, rank, ukf, timer, minimoon, formation,
            od_convergence_streak, last_od_stop_metrics, last_completed_outer_step,
            outer_csv_path, termination_reason="", progress_status="running", no_detection_state=None,
            tracking_state=None
    ):
        """Save the minimum state needed to resume from the last completed outer OD step."""
        payload = {
            "schema_version": 1,
            "uid": uid,
            "master_row_idx": int(m_idx),
            "rank": int(rank),
            "outer_csv_path": outer_csv_path,
            "last_completed_outer_step": int(last_completed_outer_step),
            "termination_reason": str(termination_reason or ""),
            "progress_status": str(progress_status or "running"),
            "rng_state_numpy": _safe_np_random_state(),
            # Save object dictionaries for filter/timer because these contain small arrays,
            # counters, adaptive-R histories, and mutable timer search grids.
            "ukf_dict": copy.deepcopy(getattr(ukf, "__dict__", {})),
            "timer_dict": copy.deepcopy(getattr(timer, "__dict__", {})),
            # The asteroid epoch is intentionally not authoritative; timer.curr_epoch is.
            "minimoon_curr_state_eme": np.asarray(getattr(minimoon, "curr_state_eme", np.full(6, np.nan)), dtype=float).copy(),
            "formation_states_eme": np.asarray(formation.get_spacecraft_states(), dtype=float).copy(),
            "formation_pointings_eme": np.asarray(formation.get_spacecraft_pointings(), dtype=float).copy(),
            "formation_sc_epochs": _get_sc_epochs(formation),
            "currently_detecting": tuple(int(x) for x in np.asarray(getattr(formation, "currently_detecting", ()), dtype=int).reshape(-1)),
            "od_convergence_streak": int(od_convergence_streak),
            "last_od_stop_metrics": copy.deepcopy(last_od_stop_metrics),
            "no_detection_state": copy.deepcopy(no_detection_state or {}),
            "tracking_state": copy.deepcopy(tracking_state or {}),
        }
        _atomic_pickle_dump(payload, checkpoint_path)

    def _read_od_checkpoint(checkpoint_path):
        if not os.path.exists(checkpoint_path):
            return None
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)

    def _restore_od_checkpoint(chk, *, ukf, timer, minimoon, formation):
        """Restore checkpoint state into freshly constructed objects."""
        if not isinstance(chk, dict):
            raise ValueError("OD checkpoint is not a dictionary")
        if int(chk.get("schema_version", -1)) != 1:
            raise ValueError(f"Unsupported OD checkpoint schema_version={chk.get('schema_version')}")

        ukf.__dict__.update(copy.deepcopy(chk.get("ukf_dict", {})))
        timer.__dict__.update(copy.deepcopy(chk.get("timer_dict", {})))
        minimoon.curr_state_eme = np.asarray(chk["minimoon_curr_state_eme"], dtype=float).reshape(6,)
        formation.set_spacecraft_states(np.asarray(chk["formation_states_eme"], dtype=float).reshape(-1, 6))
        formation.set_spacecraft_pointings(np.asarray(chk["formation_pointings_eme"], dtype=float).reshape(-1, 3))
        _set_sc_epochs(formation, chk.get("formation_sc_epochs", None))
        formation.currently_detecting = tuple(int(x) for x in np.asarray(chk.get("currently_detecting", ()), dtype=int).reshape(-1))
        _restore_np_random_state(chk.get("rng_state_numpy", None))
        return int(chk.get("od_convergence_streak", 0)), copy.deepcopy(chk.get("last_od_stop_metrics", {}))

    def _write_incomplete_progress(progress_path, *, uid, timer, outer_csv_path, termination_reason, checkpoint_path=None):
        _write_progress(progress_path, {
            "uid": uid,
            "last_completed_outer_step": int(getattr(timer, "curr_od_index", -1)),
            "last_epoch_jdtdb": float(getattr(timer, "curr_epoch", np.nan)),
            "completed": False,
            "termination_reason": str(termination_reason),
            "outer_csv_path": outer_csv_path,
            "checkpoint_path": checkpoint_path,
        })

    def _walltime_near_limit(job_start_time, wall_cfg):
        if not bool(wall_cfg.get("enabled", False)):
            return False
        max_walltime_sec = wall_cfg.get("max_walltime_sec", None)
        if max_walltime_sec is None:
            return False
        safety_buffer_sec = float(wall_cfg.get("safety_buffer_sec", 900.0))
        return (time.time() - job_start_time) >= (float(max_walltime_sec) - safety_buffer_sec)

    def _read_last_outer_row(outer_csv_path):
        """Return the last row of an outer-loop CSV as a dict, or an empty dict."""
        if not os.path.exists(outer_csv_path):
            return {}
        try:
            df = pd.read_csv(outer_csv_path)
            if len(df) == 0:
                return {}
            return df.iloc[-1].to_dict()
        except Exception:
            return {}

    def _csv_scalar(v):
        """Make scalar values safe for CSV without turning useful strings into NaN."""
        try:
            if v is None:
                return ""
            if isinstance(v, str):
                return v
            if pd.isna(v):
                return np.nan
            return v
        except Exception:
            return v

    def _serialize_outer_vec(last_outer_row, prefix, n):
        """Serialize fields like x_est_0...x_est_5 from an outer-loop row."""
        vals = []
        for i in range(n):
            v = last_outer_row.get(f"{prefix}_{i}", np.nan)
            try:
                vals.append("" if pd.isna(v) else f"{float(v):.16g}")
            except Exception:
                vals.append(str(v))
        return ";".join(vals)

    def _build_od_master_row_from_outer_last(
            *, uid, m_idx, rank, outer_csv_path, num_sc, termination_reason_fallback="",
            final_epoch_fallback=np.nan, final_steps_fallback=np.nan
    ):
        """Build one compact OD summary row from the final row of a run's outer-loop CSV."""
        last = _read_last_outer_row(outer_csv_path)

        # Prefer the final outer-loop row. Fall back only if the outer log cannot be read.
        final_epoch = last.get("epoch_end_jdtdb", final_epoch_fallback)
        final_steps = last.get("od_step_idx", final_steps_fallback)
        term_reason = last.get("termination_reason", termination_reason_fallback)

        row = {
            "master_row_idx": int(m_idx),
            "run_uid": uid,
            "rank": int(rank),
            "OD_RUN_UID": uid,
            "OD_RANK": int(rank),
            "OD_RESULT_SAVED_AS": os.path.basename(outer_csv_path),
            "OD_OUTER_CSV_PATH": outer_csv_path,
            "OD_EVENT_TYPE": _csv_scalar(last.get("event_type", "")),
            "OD_TERMINATION_REASON": _csv_scalar(term_reason),
            "OD_PROGRESS_STATUS": _csv_scalar(last.get("progress_status", "")),
            "OD_FINAL_TIME_JDTDB": _csv_scalar(final_epoch),
            "OD_N_STEPS": _csv_scalar(final_steps),
            "OD_HAD_DETECTION": _csv_scalar(last.get("had_detection", np.nan)),
            "OD_N_DETECTIONS": _csv_scalar(last.get("n_detections", np.nan)),
            "OD_DETECTING_IDS": _csv_scalar(last.get("detecting_ids", "")),
            "OD_NO_DETECTION_REASON": _csv_scalar(last.get("no_detection_reason", "")),
            "OD_ALL_EMS_OCCLUDED": _csv_scalar(last.get("all_ems_occluded", np.nan)),
            "OD_IN_EMS_BLACKOUT": _csv_scalar(last.get("in_ems_blackout", np.nan)),
            "OD_PENDING_REACQUISITION": _csv_scalar(last.get("pending_reacquisition", np.nan)),
            "OD_REACQUISITION_ATTEMPT_COUNT": _csv_scalar(last.get("reacquisition_attempt_count", np.nan)),
            "OD_TRACKING_ANCHOR_QUEUE": _csv_scalar(last.get("tracking_anchor_queue", "")),
            "OD_TRACKING_ANCHOR_SID": _csv_scalar(last.get("tracking_anchor_sid", np.nan)),
            "OD_TRACKING_ANCHOR_MODE": _csv_scalar(last.get("tracking_anchor_mode", "")),
            "OD_TRACKING_ANCHOR_FEASIBLE": _csv_scalar(last.get("tracking_anchor_feasible", np.nan)),
            "OD_PROCESSED_EPOCH_FIRST_JDTDB": _csv_scalar(last.get("processed_epoch_first_jdtdb", np.nan)),
            "OD_PROCESSED_EPOCH_LAST_JDTDB": _csv_scalar(last.get("processed_epoch_last_jdtdb", np.nan)),
            "OD_PROCESSED_EPOCH_COUNT": _csv_scalar(last.get("processed_epoch_count", np.nan)),
            "OD_TIME_SEC": _csv_scalar(last.get("od_time_sec", np.nan)),
            "OD_ATTCOORD_TIME_SEC": _csv_scalar(last.get("attcoord_time_sec", np.nan)),
            "OD_LAST_POS_RMSE": _csv_scalar(last.get("pos_err_norm", np.nan)),
            "OD_LAST_VEL_RMSE": _csv_scalar(last.get("vel_err_norm", np.nan)),
            "OD_P_POS_TRACE": _csv_scalar(last.get("P_pos_trace", np.nan)),
            "OD_P_VEL_TRACE": _csv_scalar(last.get("P_vel_trace", np.nan)),
            "OD_P_POS_SIGMA_3D": _csv_scalar(last.get("P_pos_sigma_3d", np.nan)),
            "OD_P_VEL_SIGMA_3D": _csv_scalar(last.get("P_vel_sigma_3d", np.nan)),
            "OD_NIS_LAST": _csv_scalar(last.get("NIS_last", np.nan)),
            "OD_NIS_MEAN": _csv_scalar(last.get("NIS_mean", np.nan)),
            "OD_NIS_COUNT": _csv_scalar(last.get("NIS_count", np.nan)),
            "OD_CONVERGENCE_STREAK": _csv_scalar(last.get("OD_convergence_streak", np.nan)),
            "OD_CHOSEN_CANDIDATE_IDX": _csv_scalar(last.get("chosen_candidate_idx", np.nan)),
            "OD_CHOSEN_CANDIDATE_EPOCH_JDTDB": _csv_scalar(last.get("chosen_candidate_epoch_jdtdb", np.nan)),
            "OD_FINAL_STATE": _serialize_outer_vec(last, "x_est", 6),
            "OD_FINAL_TRUE_STATE": _serialize_outer_vec(last, "x_true", 6),
            "OD_FINAL_P_DIAG": _serialize_outer_vec(last, "P_diag", 6),
            "OD_MASTER_WRITE_UTC": dt.datetime.utcnow().isoformat() + "Z",
        }

        for sc_id in range(num_sc):
            row[f"OD_FINAL_SC{sc_id}_STATE"] = _serialize_outer_vec(last, f"sc{sc_id}_state_post", 6)
            row[f"OD_FINAL_SC{sc_id}_POINTING"] = _serialize_outer_vec(last, f"sc{sc_id}_pointing_post", 3)

        return row

    def _append_od_master_row(od_master_rank_path, row_dict, od_master_header, dedup_existing=True):
        """Append one completed OD summary to this rank's OD master CSV.

        This is rank-local, so it needs no MPI/file lock. If a retry tries to append
        the same master_row_idx/run_uid again, dedup_existing prevents duplicates.
        """
        os.makedirs(os.path.dirname(od_master_rank_path), exist_ok=True)
        if dedup_existing and os.path.exists(od_master_rank_path):
            try:
                prev = pd.read_csv(od_master_rank_path, usecols=["master_row_idx", "run_uid"])
                dup = (prev["master_row_idx"].astype(int) == int(row_dict["master_row_idx"])) & (prev["run_uid"].astype(str) == str(row_dict["run_uid"]))
                if bool(dup.any()):
                    return
            except Exception:
                pass
        _append_row(od_master_rank_path, row_dict, od_master_header)

    def _merge_od_master_rows_into_master(master_fn, od_master_dir, od_master_glob, od_master_header):
        """Rank-0 convenience merge: per-rank OD master rows -> MASTER_IOD.csv.

        The per-rank OD master files remain the durable OD-stage record even if this
        final merge is interrupted by walltime.
        """
        paths = sorted(glob.glob(os.path.join(od_master_dir, od_master_glob)))
        if len(paths) == 0:
            return 0

        frames = []
        for path in paths:
            try:
                df = pd.read_csv(path)
                if len(df) > 0:
                    frames.append(df)
            except Exception as e:
                print(f"[Stage: OD] Could not read OD master file {path}: {e}", flush=True)

        if len(frames) == 0:
            return 0

        od_df = pd.concat(frames, ignore_index=True)
        if "master_row_idx" not in od_df.columns:
            return 0

        # Keep the latest appended summary for each MASTER row.
        if "OD_MASTER_WRITE_UTC" in od_df.columns:
            od_df = od_df.sort_values(["master_row_idx", "OD_MASTER_WRITE_UTC"])
        od_df = od_df.drop_duplicates(subset=["master_row_idx"], keep="last")

        dfm = pd.read_csv(master_fn)
        # Keep rank-local bookkeeping columns in OD_MASTER.csv, but do not add
        # unprefixed columns like run_uid/rank to the main MASTER_IOD.csv.
        merge_cols = [c for c in od_df.columns if c not in ("master_row_idx", "run_uid", "rank")]
        for col in merge_cols:
            if col not in dfm.columns:
                dfm[col] = ""

        for _, upd in od_df.iterrows():
            try:
                ri = int(upd["master_row_idx"])
            except Exception:
                continue
            if ri < 0 or ri >= len(dfm):
                continue
            for col in merge_cols:
                dfm.at[ri, col] = upd[col]

        tmp = master_fn + ".tmp"
        dfm.to_csv(tmp, index=False)
        os.replace(tmp, master_fn)
        return int(len(od_df))

    def _summarize_meas_outer(meas):
        try:
            arr = np.asarray(meas)

            if arr.size == 0:
                return 0, np.nan, [np.nan, np.nan]

            # reshape to (num_pairs, 2)
            pairs = arr.reshape(-1, 2)

            # keep only valid RA/Dec pairs
            valid_mask = np.isfinite(pairs).all(axis=1)
            valid_pairs = pairs[valid_mask]

            n_valid = valid_pairs.shape[0]

            if n_valid == 0:
                return 0, np.nan, [np.nan, np.nan]

            norm = float(np.linalg.norm(valid_pairs))

            first_pair = valid_pairs[0]

            return n_valid, norm, [float(first_pair[0]), float(first_pair[1])]

        except Exception:
            return 0, np.nan, [np.nan, np.nan]

    def _extract_inner_obs_state(sc_states_k, update_idx, detecting_ids=None):
        """
        sc_states_k shape: (M, N, 6)
        We map update_idx to time index update_idx, and by default pick first detecting SC if possible.
        """
        try:
            arr = np.asarray(sc_states_k)
            if arr.ndim != 3:
                return [np.nan] * 6

            M, N, D = arr.shape
            if D < 6 or update_idx >= N:
                return [np.nan] * 6

            sc_pick = 0
            if detecting_ids is not None:
                try:
                    det = np.asarray(detecting_ids).reshape(-1)
                    if det.size > 0:
                        sc_pick = int(det[0])
                except Exception:
                    pass

            sc_pick = max(0, min(M - 1, sc_pick))
            return _flatten_vec(arr[sc_pick, update_idx, :], 6)
        except Exception:
            return [np.nan] * 6

    def _extract_inner_meas_pair(perfect_meas, noisy_meas, update_idx, detecting_ids=None):
        """
        perfect_meas, noisy_meas shapes: (M, N, 2)
        For a given inner update index, extract one representative pair:
        first detecting SC if available, otherwise SC 0.
        """
        try:
            p = np.asarray(perfect_meas)
            n = np.asarray(noisy_meas)
            if p.ndim != 3 or n.ndim != 3:
                return [np.nan, np.nan], [np.nan, np.nan]
            M, N, D = p.shape
            if D < 2 or update_idx >= N:
                return [np.nan, np.nan], [np.nan, np.nan]

            sc_pick = 0
            if detecting_ids is not None:
                try:
                    det = np.asarray(detecting_ids).reshape(-1)
                    if det.size > 0:
                        sc_pick = int(det[0])
                except Exception:
                    pass

            sc_pick = max(0, min(M - 1, sc_pick))
            return _flatten_vec(p[sc_pick, update_idx, :], 2), _flatten_vec(n[sc_pick, update_idx, :], 2)
        except Exception:
            return [np.nan, np.nan], [np.nan, np.nan]

    def print_od_status(
            *,
            timer,
            ukf,
            minimoon=None,
            formation=None,
            x_true=None,
            n_detections=None,
            status_every=1,
            prefix="[OD]",
            print_time=True,
    ):
        k = getattr(timer, "curr_od_index", None)
        if k is None:
            return

        if status_every is not None and status_every > 1:
            if (k % status_every) != 0:
                return

        x_est = getattr(ukf, "x", None)
        P_est = getattr(ukf, "P", None)

        if x_true is None and minimoon is not None:
            x_true = getattr(minimoon, "curr_state_eme", None)

        pos_err_norm, vel_err_norm = _best_effort_state_error(x_est, x_true)
        Pdiag = _safe_diag(P_est, n=6)

        det_id = ""
        if formation is not None and hasattr(formation, "currently_detecting"):
            det_id = str(getattr(formation, "currently_detecting"))

        epoch = getattr(timer, "curr_epoch", None)
        endt = getattr(timer, "end_time", None)

        ts = ""
        if print_time:
            ts = datetime.now().strftime("%H:%M:%S") + " "

        line1 = (
            f"{ts}{prefix} k={k} "
            f"epoch_jdtdb={epoch if epoch is not None else ''} "
            f"end_jdtdb={endt if endt is not None else ''} "
            f"det_sc={det_id} "
            f"n_det={'' if n_detections is None else n_detections}"
        )
        line2 = f"{prefix} x_est={_fmt_vec(x_est, n=6, prec=6)}"
        line3 = (
            f"{prefix} pos_err_norm={pos_err_norm if not np.isnan(pos_err_norm) else ''} "
            f"vel_err_norm={vel_err_norm if not np.isnan(vel_err_norm) else ''} "
            f"P_diag={_fmt_vec(Pdiag, n=6, prec=6)}"
        )

        print(line1)
        print(line2)
        print(line3)

    def _cov_trace_metrics(P):
        """Return trace/sigma metrics for the 3x3 position and velocity covariance blocks."""
        try:
            P = np.asarray(P, dtype=float).reshape(6, 6)
            tr_pos = float(np.trace(P[:3, :3]))
            tr_vel = float(np.trace(P[3:, 3:]))
            sig_pos = float(np.sqrt(max(tr_pos, 0.0)))
            sig_vel = float(np.sqrt(max(tr_vel, 0.0)))
            return tr_pos, tr_vel, sig_pos, sig_vel
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

    def _extract_nis_values(update_history, window=None):
        """Extract NIS = innovation.T @ inv(S) @ innovation from nested UKF update history."""
        vals = []
        try:
            for entry in update_history or []:
                for upd in entry.get("updates", []):
                    info = upd.get("update_info", {})
                    innov = info.get("innovation", None)
                    S = info.get("S", None)
                    if innov is None or S is None:
                        continue
                    innov = np.asarray(innov, dtype=float).reshape(-1)
                    S = np.asarray(S, dtype=float)
                    if innov.size == 0 or S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] != innov.size:
                        continue
                    nis = float(innov @ np.linalg.pinv(S) @ innov)
                    if np.isfinite(nis):
                        vals.append(nis)
        except Exception:
            pass

        if window is not None:
            try:
                window = int(window)
                if window > 0:
                    vals = vals[-window:]
            except Exception:
                pass
        return vals

    def _ems_detection_flags(detection_res_k):
        """Return EMS/no-detection flags from Formation.detect() result dictionaries.

        occluded_ems is the pure EMS exclusion flag. It is intentionally used
        instead of visible_ems_filtered, because visible_ems_filtered also folds
        in FOV and Earth/Moon visibility.
        """
        try:
            rows = list(detection_res_k or [])
        except Exception:
            rows = []

        if len(rows) == 0:
            return {
                "any_detected": False,
                "all_ems_occluded": False,
                "any_ems_occluded": False,
                "n_ems_occluded": 0,
            }

        any_detected = any(bool(d.get("detected", False)) for d in rows if isinstance(d, dict))
        ems_flags = [bool(d.get("occluded_ems", False)) for d in rows if isinstance(d, dict)]
        n_ems = int(sum(ems_flags))
        return {
            "any_detected": bool(any_detected),
            "all_ems_occluded": bool(len(ems_flags) == len(rows) and len(rows) > 0 and all(ems_flags)),
            "any_ems_occluded": bool(any(ems_flags)),
            "n_ems_occluded": n_ems,
        }

    def _no_detection_state_dict(in_ems_blackout, pending_reacquisition, reacquisition_attempt_count):
        return {
            "in_ems_blackout": bool(in_ems_blackout),
            "pending_reacquisition": bool(pending_reacquisition),
            "reacquisition_attempt_count": int(reacquisition_attempt_count),
        }

    def _tracking_state_dict(tracking_anchor_queue, tracking_anchor_sid=None):
        q = [int(x) for x in list(tracking_anchor_queue or [])]
        return {
            "tracking_anchor_queue": q,
            "tracking_anchor_sid": (None if tracking_anchor_sid is None else int(tracking_anchor_sid)),
        }

    def _ids_to_int_list(ids):
        if ids is None:
            return []
        try:
            return [int(x) for x in np.asarray(ids).reshape(-1)]
        except Exception:
            try:
                return [int(x) for x in list(ids)]
            except Exception:
                return []

    def _update_tracking_anchor_queue(anchor_queue, current_detecting_ids):
        """Oldest-still-detecting custody queue.

        Existing anchors that are still detecting keep their order; newly
        detecting spacecraft are appended to the back.  The active anchor is
        always queue[0].
        """
        current = _ids_to_int_list(current_detecting_ids)
        current_set = set(current)
        q_old = [int(x) for x in list(anchor_queue or [])]
        q_new = [sid for sid in q_old if sid in current_set]
        for sid in current:
            if sid not in q_new:
                q_new.append(sid)
        return q_new

    def _tracking_queue_to_str(anchor_queue):
        return ";".join(str(int(x)) for x in list(anchor_queue or []))


    def _od_stop_metrics_and_decision(config, P, update_history, convergence_streak):
        """
        Check OD convergence using only covariance traces and NIS.

        Expected config block, for example:
            od_stop:
              enabled: true
              pos_trace_threshold_km2: 1.0e4          # OR pos_sigma_threshold_km
              vel_trace_threshold_km2_s2: 2.5e-5      # OR vel_sigma_threshold_km_s
              nis_mean_max: 10.0
              nis_mean_min: 0.0                       # optional
              nis_window: 20
              nis_min_samples: 3
              required_consecutive_steps: 1
        """
        stop_cfg = config.get("od_stop", {})
        enabled = bool(stop_cfg.get("enabled", False))

        tr_pos, tr_vel, sig_pos, sig_vel = _cov_trace_metrics(P)
        nis_window = stop_cfg.get("nis_window", None)
        nis_vals = _extract_nis_values(update_history, window=nis_window)
        nis_count = int(len(nis_vals))
        nis_last = float(nis_vals[-1]) if nis_count > 0 else np.nan
        nis_mean = float(np.mean(nis_vals)) if nis_count > 0 else np.nan

        metrics = {
            "trace_pos": tr_pos,
            "trace_vel": tr_vel,
            "sigma_pos_3d": sig_pos,
            "sigma_vel_3d": sig_vel,
            "nis_last": nis_last,
            "nis_mean": nis_mean,
            "nis_count": nis_count,
            "convergence_streak": int(convergence_streak),
        }

        if not enabled:
            return False, 0, metrics, "od_stop_disabled"

        # Accept either trace thresholds or sigma thresholds. Sigma thresholds are converted to trace thresholds.
        pos_trace_thr = stop_cfg.get("pos_trace_threshold_km2", None)
        if pos_trace_thr is None and stop_cfg.get("pos_sigma_threshold_km", None) is not None:
            pos_trace_thr = float(stop_cfg["pos_sigma_threshold_km"]) ** 2

        vel_trace_thr = stop_cfg.get("vel_trace_threshold_km2_s2", None)
        if vel_trace_thr is None and stop_cfg.get("vel_sigma_threshold_km_s", None) is not None:
            vel_trace_thr = float(stop_cfg["vel_sigma_threshold_km_s"]) ** 2

        nis_min_samples = int(stop_cfg.get("nis_min_samples", 1))
        nis_mean_max = stop_cfg.get("nis_mean_max", None)
        nis_mean_min = stop_cfg.get("nis_mean_min", None)
        required_consecutive = int(stop_cfg.get("required_consecutive_steps", 1))

        checks = []
        reasons = []

        if pos_trace_thr is not None:
            ok = np.isfinite(tr_pos) and (tr_pos <= float(pos_trace_thr))
            checks.append(ok)
            if not ok:
                reasons.append(f"pos_trace {tr_pos:.6g} > {float(pos_trace_thr):.6g}")

        if vel_trace_thr is not None:
            ok = np.isfinite(tr_vel) and (tr_vel <= float(vel_trace_thr))
            checks.append(ok)
            if not ok:
                reasons.append(f"vel_trace {tr_vel:.6g} > {float(vel_trace_thr):.6g}")

        if nis_mean_max is not None or nis_mean_min is not None:
            ok_samples = nis_count >= nis_min_samples and np.isfinite(nis_mean)
            if nis_mean_max is not None:
                ok = ok_samples and (nis_mean <= float(nis_mean_max))
                checks.append(ok)
                if not ok:
                    reasons.append(f"nis_mean {nis_mean:.6g} > {float(nis_mean_max):.6g} or insufficient NIS samples")
            if nis_mean_min is not None:
                ok = ok_samples and (nis_mean >= float(nis_mean_min))
                checks.append(ok)
                if not ok:
                    reasons.append(f"nis_mean {nis_mean:.6g} < {float(nis_mean_min):.6g} or insufficient NIS samples")

        # If no actual threshold was provided, never stop.
        if len(checks) == 0:
            return False, 0, metrics, "od_stop_enabled_but_no_thresholds"

        instant_ok = all(checks)
        new_streak = convergence_streak + 1 if instant_ok else 0
        metrics["convergence_streak"] = int(new_streak)

        should_stop = new_streak >= required_consecutive
        reason = "convergence_criteria_met" if should_stop else ("; ".join(reasons) if reasons else "waiting_for_consecutive_convergence")
        return should_stop, new_streak, metrics, reason

    # ---------------------------------------------------------
    # Diagnostics config
    # ---------------------------------------------------------
    diag_cfg = config_global.get("od_diagnostics", {})
    log_inner_kf = bool(diag_cfg.get("log_inner_kf", False))
    log_attcoord = bool(diag_cfg.get("log_attcoord_candidates", False))
    log_optimizer = bool(diag_cfg.get("log_optimizer_history", False))

    checkpoint_cfg = config_global.get("od_checkpoint", {})
    checkpoint_enabled_default = True
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", checkpoint_enabled_default))
    checkpoint_name = str(checkpoint_cfg.get("checkpoint_name", "checkpoint_state.pkl"))
    walltime_cfg = checkpoint_cfg.get("walltime_guard", {})
    # Backward-compatible flat config style is also accepted:
    # od_checkpoint:
    #   walltime_guard_enabled: true
    #   max_walltime_sec: 82800
    #   safety_buffer_sec: 900
    if "walltime_guard_enabled" in checkpoint_cfg or "max_walltime_sec" in checkpoint_cfg:
        walltime_cfg = {
            "enabled": bool(checkpoint_cfg.get("walltime_guard_enabled", checkpoint_cfg.get("enabled", False))),
            "max_walltime_sec": checkpoint_cfg.get("max_walltime_sec", None),
            "safety_buffer_sec": checkpoint_cfg.get("safety_buffer_sec", 900.0),
        }

    outer_dir = os.path.join(od_dir, config_global['od_diagnostics']['outer_loop_dir'])
    detail_root = os.path.join(od_dir, config_global['od_diagnostics']['detail_root_dir'])
    os.makedirs(outer_dir, exist_ok=True)
    os.makedirs(detail_root, exist_ok=True)

    # Load MASTER (workers)
    df_master = pd.read_csv(master_fn)

    # ---------------------------------------------------------
    # Startup load balancing for resumed OD jobs
    # ---------------------------------------------------------
    load_balance_cfg = config_global.get("OD_LOAD_BALANCING", {}) or {}
    redistribute_unfinished = bool(load_balance_cfg.get("redistribute_unfinished_on_start", True))
    repair_done_from_progress = bool(load_balance_cfg.get("repair_done_from_completed_progress", True))
    require_nonempty_outer_for_progress_repair = bool(
        load_balance_cfg.get("require_nonempty_outer_for_progress_repair", True)
    )

    def _progress_path_for_uid(uid: str):
        return os.path.join(detail_root, uid, "progress.json")

    def _outer_csv_path_for_uid(uid: str):
        base = f"{uid}__OD_{config_global['dynamics']}_{config_global['orbit']}_{config_global['observer']}_{config_global['optimizer']}"
        return os.path.join(outer_dir, f"{base}__outer.csv")

    def _outer_csv_has_rows(path: str) -> bool:
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
            # More robust than counting all lines because a header-only CSV should not count.
            return sum(1 for _ in open(path, "r", encoding="utf-8")) > 1
        except Exception:
            return False

    def od_row_is_complete(uid: str) -> bool:
        """Return True iff this OD row should be skipped on startup.

        The .done marker is authoritative.  Optionally, a completed progress.json can
        repair a missing .done marker, which is useful if a job died after writing
        progress but before committing the done file.
        """
        if is_done(uid):
            return True

        if not repair_done_from_progress:
            return False

        progress_path = _progress_path_for_uid(uid)
        if not os.path.exists(progress_path):
            return False

        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress = json.load(f)
            if not bool(progress.get("completed", False)):
                return False

            outer_path = str(progress.get("outer_csv_path", "") or _outer_csv_path_for_uid(uid))
            if require_nonempty_outer_for_progress_repair and not _outer_csv_has_rows(outer_path):
                return False

            # Repair missing done marker. Safe if multiple ranks race here; same content.
            try:
                write_done(uid)
            except Exception:
                pass
            return True
        except Exception:
            return False

    if redistribute_unfinished:
        if rank == 0:
            remaining_indices = []
            completed_count = 0
            for m_idx in range(n_rows):
                row = df_master.iloc[m_idx]
                uid = uid_from_saved_as(str(row.get("IOD_DATA_SAVED_AS", "") or ""), m_idx)
                if od_row_is_complete(uid):
                    completed_count += 1
                else:
                    remaining_indices.append(m_idx)

            print(
                f"[Stage: OD] load balancing: redistributing {len(remaining_indices)} unfinished "
                f"rows across {size} ranks ({completed_count}/{n_rows} already complete).",
                flush=True,
            )
        else:
            remaining_indices = None

        remaining_indices = comm.bcast(remaining_indices, root=0)
        my_indices = list(remaining_indices[rank::size])
    else:
        # Original deterministic static allocation.
        my_indices = list(range(n_rows))[rank::size]
        if rank == 0:
            print(
                "[Stage: OD] load balancing disabled: using static round-robin row assignment.",
                flush=True,
            )

    print(f"[rank {rank}] OD assigned rows after startup filtering = {len(my_indices)}", flush=True)
    comm.Barrier()

    # OD config
    od_duration_days = float(config_global.get('od_duration_days', 1.0))
    od_max_steps = config_global.get('od_max_steps', None)
    od_max_steps = int(od_max_steps) if (od_max_steps is not None) else None

    # No-detection / EMS blackout behavior.
    # During EMS blackout, attitude coordination is skipped and the simulation
    # advances by ems_blackout_dt_sec with prediction only. Once EMS blockage
    # clears after a blackout, one or more reacquisition attitude-coordination
    # attempts are allowed before declaring the object lost.
    od_no_det_cfg = config_global.get("od_no_detection", {})
    ems_blackout_dt_sec = float(od_no_det_cfg.get("ems_blackout_dt_sec", 600.0))
    max_reacquisition_attempts_after_blackout = int(
        od_no_det_cfg.get("max_reacquisition_attempts_after_blackout", 1)
    )
    terminate_if_ems_visible_but_not_detected = bool(
        od_no_det_cfg.get("terminate_if_ems_visible_but_not_detected", True)
    )

    # Tracking-anchor behavior for attitude coordination.  This is separate
    # from the fixed-agent optimizer mechanism: run_OD maintains the custody
    # queue and passes one selected anchor to AttitudeCoordinator.step().
    attcoord_tracking_cfg = config_global.get("attcoord_tracking", {})
    preserve_detector_anchor = bool(attcoord_tracking_cfg.get("preserve_detector_anchor", True))
    tracking_anchor_policy = str(attcoord_tracking_cfg.get("anchor_policy", "oldest_still_detecting")).lower().strip()
    tracking_anchor_fixed_mode = str(attcoord_tracking_cfg.get("anchor_fixed_mode", "mean_los_per_epoch")).lower().strip()
    tracking_anchor_fallback = str(attcoord_tracking_cfg.get("anchor_infeasible_fallback", "next_oldest")).lower().strip()

    if tracking_anchor_policy != "oldest_still_detecting":
        raise ValueError("attcoord_tracking.anchor_policy currently supports only 'oldest_still_detecting'")
    if tracking_anchor_fixed_mode not in ("mean_los_per_epoch", "provided"):
        raise ValueError("attcoord_tracking.anchor_fixed_mode must be 'mean_los_per_epoch' or 'provided'")
    if tracking_anchor_fallback not in ("next_oldest", "release"):
        raise ValueError("attcoord_tracking.anchor_infeasible_fallback must be 'next_oldest' or 'release'")

    # Per-rank OD master files: durable, low-memory OD summary output for MPI/HPC.
    od_master_cfg = config_global.get("od_master", {})
    od_master_enabled = bool(od_master_cfg.get("enabled", True))
    od_master_dir = od_master_cfg.get("dir", os.path.join(od_dir, "od_master_rows"))
    od_master_filename_template = str(od_master_cfg.get("rank_filename_template", "od_master_rank_{rank}.csv"))
    od_master_rank_path = os.path.join(od_master_dir, od_master_filename_template.format(rank=rank))
    od_master_merge_at_end = bool(od_master_cfg.get("merge_into_master_at_end", True))
    od_master_glob = str(od_master_cfg.get("merge_glob", "od_master_rank_*.csv"))

    if od_master_enabled:
        os.makedirs(od_master_dir, exist_ok=True)

    processed = skipped = errors = 0

    # ---------------------------------------------------------
    # Diagnostics headers
    # ---------------------------------------------------------
    num_sc = int(config_global["num_spacecraft"])

    outer_header = [
        "run_uid",
        "master_row_idx",
        "rank",
        "od_step_idx",
        "event_type",
        "termination_reason",
        "epoch_start_jdtdb",
        "epoch_end_jdtdb",
        "processed_epoch_first_jdtdb",
        "processed_epoch_last_jdtdb",
        "processed_epoch_count",
        "had_detection",
        "n_detections",
        "detecting_ids",
        "no_detection_reason",
        "all_ems_occluded",
        "in_ems_blackout",
        "pending_reacquisition",
        "reacquisition_attempt_count",
        "tracking_anchor_queue",
        "tracking_anchor_sid",
        "tracking_anchor_mode",
        "tracking_anchor_feasible",
        "od_time_sec",
        "attcoord_time_sec",
        "true_meas_len",
        "true_meas_norm",
        "true_meas_0",
        "true_meas_1",
        "noisy_meas_len",
        "noisy_meas_norm",
        "noisy_meas_0",
        "noisy_meas_1",
    ]
    outer_header += [f"x_est_{i}" for i in range(6)]
    outer_header += [f"x_true_{i}" for i in range(6)]
    outer_header += [f"P_diag_{i}" for i in range(6)]
    outer_header += [
        "P_pos_trace",
        "P_vel_trace",
        "P_pos_sigma_3d",
        "P_vel_sigma_3d",
        "NIS_last",
        "NIS_mean",
        "NIS_count",
        "OD_convergence_streak",
    ]
    outer_header += ["pos_err_norm", "vel_err_norm"]

    for sc_id in range(num_sc):
        outer_header += [f"sc{sc_id}_state_pre_{i}" for i in range(6)]
    for sc_id in range(num_sc):
        outer_header += [f"sc{sc_id}_pointing_pre_{i}" for i in range(3)]
    for sc_id in range(num_sc):
        outer_header += [f"sc{sc_id}_state_post_{i}" for i in range(6)]
    for sc_id in range(num_sc):
        outer_header += [f"sc{sc_id}_pointing_post_{i}" for i in range(3)]

    outer_header += [
        "chosen_candidate_idx",
        "chosen_candidate_epoch_jdtdb",
        "progress_status",
    ]

    inner_header = [
        "run_uid",
        "master_row_idx",
        "rank",
        "od_step_idx",
        "inner_update_idx",
        "epoch_jdtdb",
    ]
    inner_header += [f"x_prior_{i}" for i in range(6)]
    inner_header += [f"x_post_{i}" for i in range(6)]
    inner_header += [f"P_prior_diag_{i}" for i in range(6)]
    inner_header += [f"P_post_diag_{i}" for i in range(6)]
    inner_header += [f"x_true_{i}" for i in range(6)]
    inner_header += [f"observer_state_{i}" for i in range(6)]
    inner_header += [f"true_meas_{i}" for i in range(2)]
    inner_header += [f"noisy_meas_{i}" for i in range(2)]
    inner_header += ["detecting_ids"]

    attcoord_header = [
        "run_uid",
        "master_row_idx",
        "rank",
        "od_step_idx",
        "candidate_idx",
        "candidate_epoch_jdtdb",
        "is_chosen",
    ]
    attcoord_header += [f"x_pred_{i}" for i in range(6)]
    attcoord_header += [f"P_pred_diag_{i}" for i in range(6)]
    for sc_id in range(num_sc):
        attcoord_header += [f"sc{sc_id}_state_{i}" for i in range(6)]
    for sc_id in range(num_sc):
        attcoord_header += [f"u_cmd_sc{sc_id}_{i}" for i in range(3)]
    attcoord_header += ["J", "coverage_score"]

    optimizer_header = [
        "run_uid",
        "master_row_idx",
        "rank",
        "od_step_idx",
        "optimizer_row_idx",
        "restart_idx",
        "use_fixed_agent",
        "idx_fix",
        "J",
    ]
    optimizer_header += [f"x_param_{i}" for i in range(12)]
    optimizer_header += [f"x_free_param_{i}" for i in range(12)]
    for sc_id in range(num_sc):
        optimizer_header += [f"u_sc{sc_id}_{i}" for i in range(3)]
    optimizer_header += [f"slew_sc{i}" for i in range(num_sc)]

    od_master_header = [
        "master_row_idx",
        "run_uid",
        "rank",
        "OD_RUN_UID",
        "OD_RANK",
        "OD_RESULT_SAVED_AS",
        "OD_OUTER_CSV_PATH",
        "OD_EVENT_TYPE",
        "OD_TERMINATION_REASON",
        "OD_PROGRESS_STATUS",
        "OD_FINAL_TIME_JDTDB",
        "OD_N_STEPS",
        "OD_HAD_DETECTION",
        "OD_N_DETECTIONS",
        "OD_DETECTING_IDS",
        "OD_NO_DETECTION_REASON",
        "OD_ALL_EMS_OCCLUDED",
        "OD_IN_EMS_BLACKOUT",
        "OD_PENDING_REACQUISITION",
        "OD_REACQUISITION_ATTEMPT_COUNT",
        "OD_TRACKING_ANCHOR_QUEUE",
        "OD_TRACKING_ANCHOR_SID",
        "OD_TRACKING_ANCHOR_MODE",
        "OD_TRACKING_ANCHOR_FEASIBLE",
        "OD_PROCESSED_EPOCH_FIRST_JDTDB",
        "OD_PROCESSED_EPOCH_LAST_JDTDB",
        "OD_PROCESSED_EPOCH_COUNT",
        "OD_TIME_SEC",
        "OD_ATTCOORD_TIME_SEC",
        "OD_LAST_POS_RMSE",
        "OD_LAST_VEL_RMSE",
        "OD_P_POS_TRACE",
        "OD_P_VEL_TRACE",
        "OD_P_POS_SIGMA_3D",
        "OD_P_VEL_SIGMA_3D",
        "OD_NIS_LAST",
        "OD_NIS_MEAN",
        "OD_NIS_COUNT",
        "OD_CONVERGENCE_STREAK",
        "OD_CHOSEN_CANDIDATE_IDX",
        "OD_CHOSEN_CANDIDATE_EPOCH_JDTDB",
        "OD_FINAL_STATE",
        "OD_FINAL_TRUE_STATE",
        "OD_FINAL_P_DIAG",
        "OD_MASTER_WRITE_UTC",
    ]
    for sc_id in range(num_sc):
        od_master_header += [f"OD_FINAL_SC{sc_id}_STATE", f"OD_FINAL_SC{sc_id}_POINTING"]

    # ---------------------------------------------------------
    # Main row loop
    # ---------------------------------------------------------
    job_start_time = time.time()

    for m_idx in my_indices:
        row = df_master.iloc[m_idx]

        config = copy.deepcopy(config_global)

        occluded = int(row["OCCLUDED_BY_EMS"]) == 1

        if occluded:
            config["ems"]["R_em"] = 0.0
            config["ems"]["alpha_s_deg"] = 0.0
            print("Doing OD without EMS Exclusion...")
        else:
            print("Doing OD with EMS Exclusion")

        saved_as_str = str(row.get("IOD_DATA_SAVED_AS", "") or "")
        uid = uid_from_saved_as(saved_as_str, m_idx)

        if is_done(uid):
            skipped += 1
            continue

        base = f"{uid}__OD_{config['dynamics']}_{config['orbit']}_{config['observer']}_{config['optimizer']}"
        outer_csv_path = os.path.join(outer_dir, f"{base}__outer.csv")
        run_dir = os.path.join(detail_root, uid)
        progress_path = os.path.join(run_dir, "progress.json")
        inner_csv_path = os.path.join(run_dir, "inner_kf_updates.csv")
        attcoord_csv_path = os.path.join(run_dir, "attcoord_candidates.csv")
        optimizer_csv_path = os.path.join(run_dir, "optimizer_history.csv")
        checkpoint_path = _checkpoint_path_for(run_dir, checkpoint_name)
        row_paused_for_walltime = False

        try:
            # ----------------------------------------------
            # Setup from iod
            # ---------------------------------------------
            setup = od_setup_from_iod(config, row, util=util, sp=sp)

            # --------------------------
            # Initialization (first step)
            # --------------------------
            x0_est = setup["iod"]["ast_iod_eme_ae_kms"]
            P0_est = setup["iod"]["cov"]["P_cart_eme"]
            x_true0 = setup["frames"]["ast_eme_ae_kms"]
            t0_jdtdb = setup["epochs"]["ae_jdtdb"]

            # Process noise initialization
            r_noise = config['radial_process_noise']
            t_noise = config['track_process_noise']
            n_noise = config['normal_process_noise']

            # Measurement noise initialization
            ra_noise = config['sigma_ra']
            dec_noise = config['sigma_dec']
            pointing_noise = config['sigma_pointing']
            meas_noise = config['sigma_meas_noise']

            # ukf weight values
            alpha = config['alpha_ukf']
            beta = config['beta_ukf']
            kappa = config['kappa_ukf']
            epsilon = config['epsilon_ukf']
            adaptive_R_config = config['adaptive_R']

            # ---------------------------------------------
            # build objects
            # -------------------------------------------
            n_body_propagator = nbody.NBodyPropagator(spice=sp, config=config)

            ukf = OD_UKF(
                x0=setup["x0_eme_kms"],
                P0=setup["P0_eme"],
                Sa_rtn=(r_noise, t_noise, n_noise),
                meas_units="mas",
                sigma_ra=ra_noise,
                sigma_dec=dec_noise,
                sigma_pointing=pointing_noise,
                sigma_meas=None,  # do not use unless you want to artificially inflate R
                adaptive_R_window=adaptive_R_config['window'],
                adaptive_R_min_samples=adaptive_R_config['min_samples'],
                adaptive_R_psd_floor=adaptive_R_config['psd_floor'],
                sigma_rho_single_obs_km=adaptive_R_config['single_obs_rho'],
                sigma_rhodot_single_obs_km_s=adaptive_R_config['single_obs_rhodot'],
                p_injection_decay_tau=adaptive_R_config.get('p_injection_decay_tau', 1.0),
                multi_observer_maturity_threshold=adaptive_R_config.get('multi_observer_maturity_threshold', 3),
                mature_single_observer_injection=adaptive_R_config.get('mature_single_observer_injection', 'decayed'),
                ukf_alpha=alpha,
                ukf_beta=beta,
                ukf_kappa=kappa,
                eps=epsilon,
            )

            attitude_coordination = AttitudeCoordinator(config)

            minimoon = Asteroid(
                row['ID_AST'],
                row['INDEX_USED'],
                config,
                current_state_eme=setup['frames']['ast_eme_ae_kms'],
                current_epoch=setup['epochs']['ae_jdtdb']
            )

            formation = Formation(config)
            sc1_ini_index = formation.get_index_from_pos(util.parse_vec_cell(row["SPACECRAFT_1_INI_POS(km)"]))
            formation.recall_formation(sc1_ini_index, config)
            formation.match_spacecraft_trajectory_full(int(row['TOTAL_LENGTH']), config)
            formation.set_spacecraft_states(
                setup["frames"]["sc_eme_ae_kms"],
                set_epochs_from_row=row
            )

            formation.set_spacecraft_pointings(setup["frames"]["sc_pointing_eme_cartesian"])
            formation.currently_detecting = tuple([int(setup["sc_detecting_id"])])

            # Tracking-anchor custody queue: oldest still-detecting spacecraft
            # remains anchor; new detectors are appended behind it.
            tracking_anchor_queue = [int(setup["sc_detecting_id"])]
            tracking_anchor_sid = int(tracking_anchor_queue[0])
            tracking_anchor_mode = "initial"
            tracking_anchor_feasible = int(True)

            timer = SimTime(
                config,
                current_od_index=0,
                current_epoch=t0_jdtdb,
                iod_time=row['COMPUTATION_TIME_SEC'],
                current_integration_epoch=t0_jdtdb,
                current_integration_index=row['INDEX_USED']
            )

            att_coord_viz_flag_ini_ini = False
            if att_coord_viz_flag_ini_ini:

                theta_h_rad = util.fov_deg2_to_half_angle_rad(config["fov"])
                ems_center_xy = np.array(config["ems"]["p_em"][:2])
                ems_center_xyz = np.array(config["ems"]["p_em"])
                ems_radius = config["ems"]['R_em']
                ray_length = config['ray_length']

                sc_eme_ae_kms = formation.get_spacecraft_states()[:, :3]
                sc_pointing_eme_cartesian = formation.get_spacecraft_pointings()
                ang_eme = util.proj_angle_xy_from_plus_x_ccw(sc_pointing_eme_cartesian)
                ast_truth_eme = minimoon.curr_state_eme

                ast_iod_eme = ukf.x
                P_cart_eme = ukf.P[:3,:3]

                ast_truth_eme = np.asarray(ast_truth_eme).reshape(-1)

                agents_xyz = sc_eme_ae_kms
                target_cov_xyz = P_cart_eme
                fig, ax = util.plot_od_scenario_3d_old(
                    t_label=None,
                    agents_xyz=agents_xyz,
                    u_opt_agents_xyz=sc_pointing_eme_cartesian,
                    theta_h_rad=theta_h_rad,
                    ray_length=ray_length,
                    u_curr_agents_xyz=sc_pointing_eme_cartesian,
                    boresight_line_len=ray_length * 0.1,
                    u_init_agents_xyz=None,
                    init_boresight_line_len=ray_length * 1.5,
                    xlim=config['three_d_prop']['xlim'], ylim=config['three_d_prop']['ylim'],
                    zlim=config['three_d_prop']['zlim'],
                    Nx=config['three_d_prop']['Nx'], Ny=config['three_d_prop']['Ny'],
                    Nz=config['three_d_prop']['Nz'],
                    max_points_for_scatter=config['three_d_prop']['max_points'],
                    target_mean_xyz=ast_iod_eme[:3],
                    target_cov_xyz=target_cov_xyz,
                    d_mahal=config['d_mahal'],
                    true_target_xyz=ast_truth_eme[:3],
                    target_mean_traj_xyz=None,
                    true_target_traj_xyz=None,
                    true_target_traj_xyz_2=None,
                    ems_center_xyz=ems_center_xyz,
                    ems_radius=ems_radius,
                    show_coverage=True,
                    show_uncertainty=True,
                    show_truth=True,
                    show_ems=True,
                    show_fov_cones=True,
                    title="3D OD Scenario Demo (EME)",
                    slew_history=None,
                    slew_history_line_len=ray_length,
                    agent_orbit_tracks_xyz=None
                )





            status_every = int(config.get("od_status_every", 1))

            last_pos_rmse = np.nan
            last_vel_rmse = np.nan
            termination_reason = ""
            progress_status = "running"

            prog = _read_progress(progress_path)
            last_completed_outer = _resume_last_outer_idx(outer_csv_path)
            if prog is not None and prog.get("completed", False):
                write_done(uid)
                skipped += 1
                continue

            resume_after_step = -1 if last_completed_outer is None else int(last_completed_outer)

            od_convergence_streak = 0
            last_od_stop_metrics = {
                "trace_pos": np.nan,
                "trace_vel": np.nan,
                "sigma_pos_3d": np.nan,
                "sigma_vel_3d": np.nan,
                "nis_last": np.nan,
                "nis_mean": np.nan,
                "nis_count": 0,
                "convergence_streak": 0,
            }

            # EMS-blackout / reacquisition state. These are also checkpointed.
            in_ems_blackout = False
            pending_reacquisition = False
            reacquisition_attempt_count = 0

            # True checkpoint resume: restore the filter/timer/formation state directly
            # instead of replaying from the beginning. The old resume_after_step guard is
            # kept only as a duplicate-write safety net.
            if checkpoint_enabled and os.path.exists(checkpoint_path):
                chk = _read_od_checkpoint(checkpoint_path)
                od_convergence_streak, restored_metrics = _restore_od_checkpoint(
                    chk, ukf=ukf, timer=timer, minimoon=minimoon, formation=formation
                )
                if restored_metrics:
                    last_od_stop_metrics.update(restored_metrics)
                nd_state = chk.get("no_detection_state", {}) if isinstance(chk, dict) else {}
                in_ems_blackout = bool(nd_state.get("in_ems_blackout", in_ems_blackout))
                pending_reacquisition = bool(nd_state.get("pending_reacquisition", pending_reacquisition))
                reacquisition_attempt_count = int(nd_state.get("reacquisition_attempt_count", reacquisition_attempt_count))
                tr_state = chk.get("tracking_state", {}) if isinstance(chk, dict) else {}
                tracking_anchor_queue = [int(x) for x in tr_state.get("tracking_anchor_queue", tracking_anchor_queue)]
                restored_anchor_sid = tr_state.get("tracking_anchor_sid", None)
                tracking_anchor_sid = None if restored_anchor_sid is None else int(restored_anchor_sid)
                resume_after_step = int(chk.get("last_completed_outer_step", resume_after_step))
                print(
                    f"[OD r{rank} uid={uid}] Resumed from checkpoint at "
                    f"od_step_idx={timer.curr_od_index}, epoch_jdtdb={timer.curr_epoch}",
                    flush=True,
                )

            while True:
                if _walltime_near_limit(job_start_time, walltime_cfg):
                    termination_reason = "walltime_checkpoint"
                    progress_status = "paused"
                    if checkpoint_enabled:
                        _save_od_checkpoint(
                            checkpoint_path, uid=uid, m_idx=m_idx, rank=rank, ukf=ukf, timer=timer,
                            minimoon=minimoon, formation=formation,
                            od_convergence_streak=od_convergence_streak,
                            last_od_stop_metrics=last_od_stop_metrics,
                            last_completed_outer_step=int(getattr(timer, "curr_od_index", -1)),
                            outer_csv_path=outer_csv_path,
                            termination_reason=termination_reason,
                            progress_status=progress_status,
                            no_detection_state=_no_detection_state_dict(
                                in_ems_blackout, pending_reacquisition, reacquisition_attempt_count
                            ),
                            tracking_state=_tracking_state_dict(tracking_anchor_queue, tracking_anchor_sid),
                        )
                    _write_incomplete_progress(
                        progress_path, uid=uid, timer=timer, outer_csv_path=outer_csv_path,
                        termination_reason=termination_reason, checkpoint_path=(checkpoint_path if checkpoint_enabled else None)
                    )
                    row_paused_for_walltime = True
                    print(f"[OD r{rank} uid={uid}] Pausing before walltime limit; checkpoint saved.", flush=True)
                    break

                if timer.curr_epoch > timer.end_time or timer.check_search_times():
                    print("Over Time")
                    termination_reason = "time_limit"
                    progress_status = "completed"
                    event_type = "termination_time_limit"

                    sc_states_pre = np.asarray(formation.get_spacecraft_states(), dtype=float)
                    sc_pointings_pre = np.asarray(formation.get_spacecraft_pointings(), dtype=float)
                    sc_states_post = sc_states_pre.copy()
                    sc_pointings_post = sc_pointings_pre.copy()

                    x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                    x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                    P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                    pos_err, vel_err = _best_effort_state_error(x_est, x_true)

                    row_out = {
                        "run_uid": uid,
                        "master_row_idx": m_idx,
                        "rank": rank,
                        "od_step_idx": int(timer.curr_od_index),
                        "event_type": event_type,
                        "termination_reason": termination_reason,
                        "epoch_start_jdtdb": float(timer.curr_epoch),
                        "epoch_end_jdtdb": float(timer.curr_epoch),
                        "processed_epoch_first_jdtdb": np.nan,
                        "processed_epoch_last_jdtdb": np.nan,
                        "processed_epoch_count": 0,
                        "had_detection": False,
                        "n_detections": 0,
                        "detecting_ids": "",
                        "od_time_sec": np.nan,
                        "attcoord_time_sec": np.nan,
                        "true_meas_len": 0,
                        "true_meas_norm": np.nan,
                        "true_meas_0": np.nan,
                        "true_meas_1": np.nan,
                        "noisy_meas_len": 0,
                        "noisy_meas_norm": np.nan,
                        "noisy_meas_0": np.nan,
                        "noisy_meas_1": np.nan,
                        "pos_err_norm": pos_err,
                        "vel_err_norm": vel_err,
                        "chosen_candidate_idx": np.nan,
                        "chosen_candidate_epoch_jdtdb": np.nan,
                        "progress_status": progress_status,
                    }
                    for i in range(6):
                        row_out[f"x_est_{i}"] = x_est[i]
                        row_out[f"x_true_{i}"] = x_true[i]
                        row_out[f"P_diag_{i}"] = P_diag[i]
                    for sc_id in range(num_sc):
                        pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                        pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                        post_s = _flatten_vec(sc_states_post[sc_id], 6)
                        post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                        for i in range(6):
                            row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                            row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                        for i in range(3):
                            row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                            row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                    if timer.curr_od_index > resume_after_step:
                        _append_row(outer_csv_path, row_out, outer_header)
                    break

                if od_max_steps is not None and timer.curr_od_index >= od_max_steps:
                    print("Out of Iterations")
                    termination_reason = "step_limit"
                    progress_status = "completed"
                    event_type = "termination_step_limit"

                    sc_states_pre = np.asarray(formation.get_spacecraft_states(), dtype=float)
                    sc_pointings_pre = np.asarray(formation.get_spacecraft_pointings(), dtype=float)
                    sc_states_post = sc_states_pre.copy()
                    sc_pointings_post = sc_pointings_pre.copy()

                    x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                    x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                    P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                    pos_err, vel_err = _best_effort_state_error(x_est, x_true)

                    row_out = {
                        "run_uid": uid,
                        "master_row_idx": m_idx,
                        "rank": rank,
                        "od_step_idx": int(timer.curr_od_index),
                        "event_type": event_type,
                        "termination_reason": termination_reason,
                        "epoch_start_jdtdb": float(timer.curr_epoch),
                        "epoch_end_jdtdb": float(timer.curr_epoch),
                        "processed_epoch_first_jdtdb": np.nan,
                        "processed_epoch_last_jdtdb": np.nan,
                        "processed_epoch_count": 0,
                        "had_detection": False,
                        "n_detections": 0,
                        "detecting_ids": "",
                        "od_time_sec": np.nan,
                        "attcoord_time_sec": np.nan,
                        "true_meas_len": 0,
                        "true_meas_norm": np.nan,
                        "true_meas_0": np.nan,
                        "true_meas_1": np.nan,
                        "noisy_meas_len": 0,
                        "noisy_meas_norm": np.nan,
                        "noisy_meas_0": np.nan,
                        "noisy_meas_1": np.nan,
                        "pos_err_norm": pos_err,
                        "vel_err_norm": vel_err,
                        "chosen_candidate_idx": np.nan,
                        "chosen_candidate_epoch_jdtdb": np.nan,
                        "progress_status": progress_status,
                    }
                    for i in range(6):
                        row_out[f"x_est_{i}"] = x_est[i]
                        row_out[f"x_true_{i}"] = x_true[i]
                        row_out[f"P_diag_{i}"] = P_diag[i]
                    for sc_id in range(num_sc):
                        pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                        pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                        post_s = _flatten_vec(sc_states_post[sc_id], 6)
                        post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                        for i in range(6):
                            row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                            row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                        for i in range(3):
                            row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                            row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                    if timer.curr_od_index > resume_after_step:
                        _append_row(outer_csv_path, row_out, outer_header)
                    break

                epoch_start = float(timer.curr_epoch)
                sc_states_pre = np.asarray(formation.get_spacecraft_states(), dtype=float)
                sc_pointings_pre = np.asarray(formation.get_spacecraft_pointings(), dtype=float)

                processed_epoch_first = np.nan
                processed_epoch_last = np.nan
                processed_epoch_count = 0
                had_detection = True
                n_detections = np.nan
                detecting_ids_str = ""
                no_detection_reason = ""
                all_ems_occluded = False
                od_time = np.nan
                attcoord_time = np.nan
                chosen_candidate_idx = np.nan
                chosen_candidate_epoch_jdtdb = np.nan
                p_meas_k = None
                n_meas_k = None

                # update according to iod
                if timer.curr_od_index == 0:
                    event_type = "initial_attcoord"

                    # -------------- REGULAR IOD STEP -------------------
                    attcoord_startime = time.time()

                    timer.set_attcoord_searchtimes()

                    x_ts, P_ts = ukf.propagate_priors(
                        timer.curr_epoch,
                        timer.attcoord_searchtimes_jdtdb,
                        n_body_propagator.propagate_multiple_objects
                    )

                    sc_eme_states_kms_piecewise, anchor_info = util.piecewise_anchor_and_propagate_spacecraft_trajs(
                        formation=formation, minimoon=minimoon, timer=timer,
                        t_targets_jdtdb=timer.attcoord_searchtimes_jdtdb,
                        n_body_propagator=n_body_propagator, return_anchor_info=True
                    )

                    ast_eme_state_kms = minimoon.curr_state_eme
                    ast_eme_traj_kms = n_body_propagator.propagate(ast_eme_state_kms, timer.curr_epoch,
                                                                   timer.attcoord_searchtimes_jdtdb)

                    ast_truth_original_eclip = np.array(minimoon.orbit.loc[:, ["Geo x", "Geo y", "Geo z", "Geo vx",
                                                                               "Geo vy", "Geo vz"]])
                    ast_truth_original_eclip[:, :3] *= config['AU_TO_M'] / 1000
                    ast_truth_original_eclip[:, 3:] *= config['AU_TO_M'] / 1000 / config['SECONDS_PER_DAY']
                    ast_truth_original_eme = util.geo_eclip_to_geo_eme_generic(ast_truth_original_eclip,
                                                                               hint=("time", "state"))

                    # to visualize the possible att coord scenarios
                    viz_prop_flag_ini = False
                    if viz_prop_flag_ini:
                        util.plot_priors_positions_and_cov_2d(
                            x_ts,
                            P_ts,
                            sc_trajs_km2=sc_eme_states_kms_piecewise,
                            stride=config['two_d_prop']['stride'],
                            n_std=config['two_d_prop']['stride'],
                            planes=("xy", "xz", "yz"),
                            title_prefix="Asteroid prior + spacecraft"
                        )

                        util.plot_matched_trajectory_full_range(
                            formation=formation,
                            idx_start=0,
                            idx_stop=2400,
                            frame="GEO_EME",
                            plot_3d=True,
                            plot_xy=False
                        )


                    sc_pointings_eme = formation.get_spacecraft_pointings()

                    sc0 = formation.spacecraft[0]
                    theta_h_rad = util.fov_deg2_to_half_angle_rad(sc0.fov)
                    tau_max = sc0.reaction_wheel_torque
                    h_max = sc0.reaction_wheel_momentum
                    m_m = sc0.mass
                    l_m = sc0.length
                    m_t = sc0.telescope_mass
                    d_t = sc0.telescope_diameter
                    l_t = sc0.telescope_length
                    z_0 = sc0.telescope_offset
                    I_max = (1 / 6) * m_m * (l_m) ** 2 + (1 /12) * m_t * (3*(d_t / 2) ** 2 + l_t ** 2) + m_t * z_0 ** 2
                    alpha_max = 1.63 * tau_max / I_max
                    omega_max = 1.63 * h_max / I_max

                    # Initial attitude coordination preserves the initial detector
                    # as a tracking anchor and fixes it to the predicted mean LOS
                    # separately for each candidate epoch.
                    initial_detecting_list = list(formation.currently_detecting)
                    tracking_anchor_queue = _update_tracking_anchor_queue(
                        tracking_anchor_queue, initial_detecting_list
                    )
                    tracking_anchor_sid = int(tracking_anchor_queue[0]) if tracking_anchor_queue else None
                    tracking_anchor_mode = tracking_anchor_fixed_mode if (preserve_detector_anchor and tracking_anchor_sid is not None) else "none"
                    tracking_anchor_feasible = int(tracking_anchor_sid is not None)

                    res_kcoverage, result_mean, result_kcoverage_series, result_mean_series, mean_time = attitude_coordination.step(
                        sc_eme_states_kms_piecewise[:, :, :3],
                        sc_pointings_eme,
                        x_ts[:, :3],
                        P_ts[:, :3, :3],
                        timer.attcoord_searchtimes,
                        theta_h_rad,
                        alpha_max,
                        omega_max,
                        sc_pointings_eme[initial_detecting_list, :] if len(initial_detecting_list) > 0 else np.empty((0, 3)),
                        initial_detecting_list,
                        d_M=config['d_mahal'],
                        use_fixed_agent=True,
                        fixed_agent_idx=int(initial_detecting_list[0]),
                        fixed_agent_u=sc_pointings_eme[int(initial_detecting_list[0]), :],
                        fixed_agent_u_mode="provided",
                        coverage_point=ast_eme_traj_kms[:, :3]
                    )

                    attcoord_endtime = time.time()
                    timer.set_attcoord_time(attcoord_endtime - attcoord_startime - mean_time)
                    attcoord_time = attcoord_endtime - attcoord_startime - mean_time
                    od_time = 0.0

                    timer.set_slew_time(res_kcoverage.chosen_dt)

                    best_epoch = res_kcoverage.chosen_dt
                    best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                    best_epoch_jdtdb = float(timer.attcoord_searchtimes_jdtdb[best_idx])
                    k_anchor_best = int(anchor_info["anchor_k_of_t"][best_idx])
                    t_anchor_best = float(anchor_info["anchor_epoch_of_t"][best_idx])

                    best_attitudes = res_kcoverage.u_cmd
                    formation.set_spacecraft_pointings(best_attitudes)

                    best_positions = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :])
                    formation.set_spacecraft_states(best_positions)

                    ukf.x = np.squeeze(x_ts[best_idx, :])
                    ukf.P = np.squeeze(P_ts[best_idx, :, :])

                    minimoon.set_state(np.squeeze(ast_eme_traj_kms[best_idx, :]))

                    timer.step(best_epoch_jdtdb, k_anchor_best, t_anchor_best)

                    # If this no-detection step was a reacquisition planning attempt
                    # after EMS blackout, count it only after attitude coordination
                    # has successfully produced and applied a new pointing.
                    if (not had_detection) and pending_reacquisition:
                        reacquisition_attempt_count += 1

                    chosen_candidate_idx = best_idx
                    chosen_candidate_epoch_jdtdb = best_epoch_jdtdb
                    had_detection = True
                    n_detections = 1
                    detecting_ids_str = str(int(setup["sc_detecting_id"]))

                    # visualize att_coord result
                    att_coord_viz_flag_ini = False
                    if att_coord_viz_flag_ini:
                        best_epoch = res_kcoverage.chosen_dt
                        best_idx = np.where(timer.attcoord_searchtimes == best_epoch)[0]

                        theta_h_rad = util.fov_deg2_to_half_angle_rad(config["fov"])
                        ems_center_xy = np.array(config["ems"]["p_em"][:2])
                        ems_center_xyz = np.array(config["ems"]["p_em"])
                        ems_radius = config["ems"]['R_em']
                        ray_length = config['ray_length']

                        sc_eme_ae_kms = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :3])
                        sc_pointing_eme_cartesian = res_kcoverage.u_cmd
                        ang_eme = util.proj_angle_xy_from_plus_x_ccw(sc_pointing_eme_cartesian)
                        ast_truth_eme = np.squeeze(ast_eme_traj_kms[best_idx, :3])

                        ast_iod_eme = np.squeeze(x_ts[best_idx, :3])
                        P_cart_eme = np.squeeze(P_ts[best_idx, :3, :3])

                        ast_truth_eme = np.asarray(ast_truth_eme).reshape(-1)

                        best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                        T = len(timer.attcoord_searchtimes)

                        two_d_pot = False
                        if two_d_pot:
                            def is_valid(idx):
                                J = result_kcoverage_series[idx]["J"]
                                return not (J is None or (isinstance(J, float) and math.isnan(J)))

                            cands = [best_idx, max(0, best_idx - 1), min(T - 1, best_idx + 1)]
                            selected = []
                            for c in cands:
                                if c not in selected and is_valid(c):
                                    selected.append(c)
                            if len(selected) < 3:
                                for c in range(T):
                                    if len(selected) >= 3:
                                        break
                                    if c not in selected and is_valid(c):
                                        selected.append(c)
                            if len(selected) < 3:
                                for c in range(T):
                                    if len(selected) >= 3:
                                        break
                                    if c not in selected:
                                        selected.append(c)

                            planes = [((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ")]

                            def cov2_from_cov3(P3, axes):
                                i, j = axes
                                return P3[np.ix_([i, j], [i, j])]

                            def angles_in_plane(u_cmd_xyz, axes):
                                a, b = axes
                                u = np.asarray(u_cmd_xyz, dtype=float)
                                u2 = u[:, [a, b]]
                                return np.arctan2(u2[:, 1], u2[:, 0])

                            for idx in selected:
                                epoch = timer.attcoord_searchtimes[idx]
                                sc_xyz = np.asarray(sc_eme_states_kms_piecewise[idx, :, :3], dtype=float)
                                ast_truth = np.asarray(ast_eme_traj_kms[idx, :3], dtype=float).reshape(3,)
                                ast_mean = np.asarray(x_ts[idx, :3], dtype=float).reshape(3,)
                                P3 = np.asarray(P_ts[idx, :3, :3], dtype=float).reshape(3, 3)

                                u_cmd_xyz = np.asarray(result_kcoverage_series[idx]["u"], dtype=float)
                                u_curr_xyz = np.asarray(sc_pointings_eme, dtype=float)

                                fig, axes = plt.subplots(1, 3, figsize=(8, 14))
                                fig.suptitle(f"2D Projections @ epoch {epoch}", y=0.99)

                                for ax, (axpair, name) in zip(axes, planes):
                                    agents_2d = sc_xyz[:, list(axpair)]
                                    u_curr_2d = u_curr_xyz[:, list(axpair)]
                                    mean_2d = ast_mean[list(axpair)]
                                    truth_2d = ast_truth[list(axpair)]
                                    cov_2d = cov2_from_cov3(P3, axpair)
                                    mean_traj_2d = np.asarray(x_ts[:, :3], dtype=float)[:, list(axpair)]
                                    truth_traj_2d = np.asarray(ast_eme_traj_kms[:, :3], dtype=float)[:, list(axpair)]
                                    ems_center_2d = np.asarray(ems_center_xyz, dtype=float)[list(axpair)]
                                    ang_2d = angles_in_plane(u_cmd_xyz, axpair)

                                    util.plot_od_scenario_2d(
                                        t_label=f"{epoch} ({name})",
                                        agents_xy=agents_2d,
                                        pointing_angles_rad=ang_2d,
                                        theta_h_rad=theta_h_rad,
                                        ray_length=ray_length * 2,
                                        u_curr_agents_xy=u_curr_2d,
                                        boresight_line_len=ray_length * 0.1,
                                        target_mean_xy=mean_2d,
                                        target_mean_xy_traj=mean_traj_2d,
                                        target_cov_xy=cov_2d,
                                        d_mahal=config['d_mahal'],
                                        true_target_xy=truth_2d,
                                        true_target_xy_traj=truth_traj_2d,
                                        ems_center_xy=ems_center_2d,
                                        ems_radius=ems_radius,
                                        xlim=config['two_d_prop']['xlim'], ylim=config['two_d_prop']['ylim'],
                                        agent_orbit_tracks_xy=None,
                                        ax=ax,
                                        title=None
                                    )

                                plt.tight_layout()

                        agents_xyz = sc_eme_ae_kms
                        target_cov_xyz = P_cart_eme

                        def range_sigma_from_cov(P_eci, r_obj, r_obs):
                            """
                            P_eci : 6x6 or 3x3 covariance in inertial frame
                            r_obj : object inertial position, shape (3,)
                            r_obs : observer inertial position, shape (3,)

                            Returns range standard deviation and variance.
                            """
                            P_r = P_eci[:3, :3] if P_eci.shape == (6, 6) else P_eci

                            rho_vec = np.asarray(r_obj) - np.asarray(r_obs)
                            rho_hat = rho_vec / np.linalg.norm(rho_vec)

                            var_rho = rho_hat @ P_r @ rho_hat
                            sigma_rho = np.sqrt(max(var_rho, 0.0))

                            return sigma_rho, var_rho

                        sigma_rho, var_rho = range_sigma_from_cov(P_cart_eme, ast_iod_eme[:3],
                                                                  agents_xyz[int(formation.currently_detecting[0])])
                        print(sigma_rho, var_rho)

                        def build_slew_history_from_opt_series(opt_series, ids, *, key_u="u", normalize=True):
                            ids = [int(i) for i in ids]
                            out = {i: [] for i in ids}
                            for row_idx, rowh in enumerate(opt_series):
                                if key_u not in rowh:
                                    raise KeyError(f"Row {row_idx} missing key '{key_u}'")
                                U = np.asarray(rowh[key_u], dtype=float)
                                if U.ndim != 2 or U.shape[1] != 3:
                                    raise ValueError(f"Row {row_idx} '{key_u}' must be (M,3), got {U.shape}")
                                M = U.shape[0]
                                for i in ids:
                                    if not (0 <= i < M):
                                        raise IndexError(f"Row {row_idx}: id {i} out of range for M={M}")
                                    out[i].append(U[i].copy())
                            for i in ids:
                                Ui = np.asarray(out[i], dtype=float)
                                if Ui.size == 0:
                                    Ui = Ui.reshape(0, 3)
                                if normalize and Ui.shape[0] > 0:
                                    n = np.linalg.norm(Ui, axis=1, keepdims=True)
                                    n = np.maximum(n, 1e-12)
                                    Ui = Ui / n
                                out[i] = Ui
                            return out

                        ids = config['three_d_prop'].get('history_ids')
                        if ids is None:
                            ids = list(range(config['num_spacecraft']))
                        slew_history = build_slew_history_from_opt_series(res_kcoverage.extra["history"], ids)

                        fig, ax = util.plot_od_scenario_3d_old(
                            t_label=best_epoch,
                            agents_xyz=agents_xyz,
                            u_opt_agents_xyz=sc_pointing_eme_cartesian,
                            theta_h_rad=theta_h_rad,
                            ray_length=ray_length,
                            u_curr_agents_xyz=sc_pointings_eme,
                            boresight_line_len=ray_length * 0.1,
                            u_init_agents_xyz=None,
                            init_boresight_line_len=ray_length * 1.5,
                            xlim=config['three_d_prop']['xlim'], ylim=config['three_d_prop']['ylim'],
                            zlim=config['three_d_prop']['zlim'],
                            Nx=config['three_d_prop']['Nx'], Ny=config['three_d_prop']['Ny'],
                            Nz=config['three_d_prop']['Nz'],
                            max_points_for_scatter=config['three_d_prop']['max_points'],
                            target_mean_xyz=ast_iod_eme[:3],
                            target_cov_xyz=target_cov_xyz,
                            d_mahal=config['d_mahal'],
                            true_target_xyz=ast_truth_eme[:3],
                            target_mean_traj_xyz=x_ts[:, :3],
                            true_target_traj_xyz=ast_eme_traj_kms[:, :3],
                            true_target_traj_xyz_2=ast_truth_original_eme[:, :3],
                            ems_center_xyz=ems_center_xyz,
                            ems_radius=ems_radius,
                            show_coverage=True,
                            show_uncertainty=True,
                            show_truth=True,
                            show_ems=True,
                            show_fov_cones=True,
                            title="3D OD Scenario Demo (EME)",
                            slew_history=slew_history,
                            slew_history_line_len=ray_length,
                            agent_orbit_tracks_xyz=[sc_eme_states_kms_piecewise[:, i, :3] for i in range(config['num_spacecraft'])]
                        )

                        plot_cost_diagnosis = True
                        if plot_cost_diagnosis:
                            util.plot_attcoord_costs_from_series(
                                result_kcoverage_series,
                                title="Dual coverage score vs objective (best per epoch)"
                            )

                            thetas, phis = util.plot_theta_phi_over_history(
                                res_kcoverage.extra["history"],
                                int(config["num_spacecraft"]),
                                deg=True
                            )

                        viz_cost_map = False
                        if viz_cost_map:
                            idx_fix = int(formation.currently_detecting[0])
                            idx_free = 1 - idx_fix
                            u_fix = sc_pointings_eme[idx_fix]

                            TH_free_deg, PH_free_deg, J_grid = compute_J_grid_theta_phi_single_free(
                                ast_iod_eme[:3], target_cov_xyz, agents_xyz, sc_pointings_eme, theta_h_rad,
                                idx_fix=idx_fix,
                                u_fix=u_fix,
                                idx_free=idx_free,
                                theta_range_rad=(-0.5 * np.pi, 0.5 * np.pi),
                                phi_range_rad=(0.0, 2 * np.pi),
                                d_M=config['d_mahal'],
                                kappa_sigma=config['optimizer_att_coord']['kappa_sigma'],
                                lambda_k1=config['optimizer_att_coord']['lambda_k1'],
                                n_mc=config['opt_map']['n_mc'],
                                n_grid_theta=config['opt_map']['n_grid_theta'],
                                n_grid_phi=config['opt_map']['n_grid_phi'],
                            )

                            fig3d = plt.figure(figsize=(9, 6))
                            ax3d = fig3d.add_subplot(111, projection="3d")
                            ax3d.plot_surface(
                                TH_free_deg, PH_free_deg, J_grid,
                                rstride=config['opt_map']['rstride'], cstride=config['opt_map']['cstride'],
                                linewidth=config['opt_map']['linewidth_3d'], alpha=config['opt_map']['alpha']
                            )
                            ax3d.set_xlabel(rf'$\theta_{{{idx_free}}}$ (deg)')
                            ax3d.set_ylabel(rf'$\phi_{{{idx_free}}}$ (deg)')
                            ax3d.set_zlabel(r'$J_t$')
                            ax3d.set_title(rf'$J_t(\theta,\phi)$ for free agent {idx_free} (fixed agent {idx_fix})')
                            plt.tight_layout()

                            plt.figure(figsize=(7, 5.5))
                            cs = plt.contourf(TH_free_deg, PH_free_deg, J_grid, levels=config['opt_map']['levels'])
                            plt.colorbar(cs, label=r'$J_t$')
                            plt.xlabel(rf'$\theta_{{{idx_free}}}$ (deg)')
                            plt.ylabel(rf'$\phi_{{{idx_free}}}$ (deg)')
                            plt.title(rf'$J_t(\theta,\phi)$ for free agent {idx_free} (fixed agent {idx_fix})')
                            plt.grid(alpha=0.3)

                            restart_indices = sorted({entry["restart"] for entry in res_kcoverage.extra["history"]})
                            colors = ['white', 'yellow', 'cyan', 'magenta', 'green', 'orange']
                            markers = ['o', 's', '^', 'D', 'x', '+']

                            for k, r in enumerate(restart_indices):
                                path_entries = [entry for entry in res_kcoverage.extra["history"] if entry["restart"] == r]
                                if not path_entries:
                                    continue

                                theta_path_deg = []
                                phi_path_deg = []

                                for entry in path_entries:
                                    if entry.get("use_fixed_agent", False) and ("x_free" in entry) and (
                                            entry.get("idx_fix", None) is not None):
                                        xk = np.asarray(entry["x_free"], dtype=float).ravel()
                                        th = xk[0]
                                        ph = xk[1]
                                    else:
                                        xk = np.asarray(entry["x"], dtype=float).ravel()
                                        th = xk[2 * idx_free]
                                        ph = xk[2 * idx_free + 1]

                                    theta_path_deg.append(np.rad2deg(th))
                                    phi_path_deg.append(np.rad2deg(ph))

                                theta_path_deg = np.asarray(theta_path_deg)
                                phi_path_deg = np.asarray(phi_path_deg)

                                col = colors[k % len(colors)]
                                m = markers[k % len(markers)]

                                label = f"Trial {r}"
                                if r == 0:
                                    label += " (warm start)"

                                plt.plot(
                                    theta_path_deg, phi_path_deg,
                                    linestyle='-',
                                    marker=m,
                                    color=col,
                                    lw=config['opt_map']['linewidth_2d'],
                                    ms=config['opt_map']['marker_size'],
                                    label=label
                                )

                            plt.legend(loc='upper right')

                    view_other_epoch_res_flag = False
                    if view_other_epoch_res_flag:
                        desired_idx = [5, 6, 7, 8, 9]
                        for des_idx in desired_idx:

                            row = result_kcoverage_series[des_idx]

                            best_epoch = row["dt"]
                            best_idx = np.where(timer.attcoord_searchtimes == best_epoch)[0]

                            theta_h_rad = util.fov_deg2_to_half_angle_rad(config["fov"])
                            ems_center_xyz = np.array(config["ems"]["p_em"])
                            ems_radius = config["ems"]['R_em']
                            ray_length = config['ray_length']

                            sc_eme_ae_kms = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :3])
                            sc_pointing_eme_cartesian = row['u']
                            ang_eme = util.proj_angle_xy_from_plus_x_ccw(sc_pointing_eme_cartesian)
                            ast_truth_eme = np.squeeze(ast_eme_traj_kms[best_idx, :3])

                            ast_iod_eme = np.squeeze(x_ts[best_idx, :3])
                            P_cart_eme = np.squeeze(P_ts[best_idx, :3, :3])

                            ast_truth_eme = np.asarray(ast_truth_eme).reshape(-1)

                            best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                            T = len(timer.attcoord_searchtimes)

                            agents_xyz = sc_eme_ae_kms
                            target_cov_xyz = P_cart_eme

                            def build_slew_history_from_opt_series(opt_series, ids, *, key_u="u", normalize=True):
                                ids = [int(i) for i in ids]
                                out = {i: [] for i in ids}

                                for row_idx, rowh in enumerate(opt_series):
                                    if key_u not in rowh:
                                        raise KeyError(f"Row {row_idx} missing key '{key_u}'")
                                    U = np.asarray(rowh[key_u], dtype=float)
                                    if U.ndim != 2 or U.shape[1] != 3:
                                        raise ValueError(f"Row {row_idx} '{key_u}' must be (M,3), got {U.shape}")
                                    M = U.shape[0]
                                    for i in ids:
                                        if not (0 <= i < M):
                                            raise IndexError(f"Row {row_idx}: id {i} out of range for M={M}")
                                        out[i].append(U[i].copy())

                                for i in ids:
                                    Ui = np.asarray(out[i], dtype=float)
                                    if Ui.size == 0:
                                        Ui = Ui.reshape(0, 3)

                                    if normalize and Ui.shape[0] > 0:
                                        n = np.linalg.norm(Ui, axis=1, keepdims=True)
                                        n = np.maximum(n, 1e-12)
                                        Ui = Ui / n

                                    out[i] = Ui

                                return out

                            ids = config['three_d_prop'].get('history_ids')
                            if ids is None:
                                ids = list(range(config['num_spacecraft']))

                            slew_history = build_slew_history_from_opt_series(row["history"], ids)

                            fig, ax = util.plot_od_scenario_3d_new(
                                t_label=best_epoch,
                                agents_xyz=agents_xyz,
                                u_opt_agents_xyz=sc_pointing_eme_cartesian,
                                theta_h_rad=theta_h_rad,
                                ray_length=ray_length,
                                u_curr_agents_xyz=sc_pointings_eme,
                                boresight_line_len=ray_length * 0.1,
                                u_init_agents_xyz=None,
                                init_boresight_line_len=ray_length * 1.5,
                                agent_orbit_tracks_xyz=[sc_eme_states_kms_piecewise[:, i, :3] for i in
                                                        range(config['num_spacecraft'])],
                                spacecraft_orbit_xyz=None,
                                xlim=config['three_d_prop']['xlim'], ylim=config['three_d_prop']['ylim'],
                                zlim=config['three_d_prop']['zlim'],
                                target_mean_xyz=ast_iod_eme[:3],
                                target_cov_xyz=target_cov_xyz,
                                d_mahal=config['d_mahal'],
                                true_target_xyz=ast_truth_eme[:3],
                                target_mean_traj_xyz=x_ts[:, :3],
                                true_target_traj_xyz=ast_eme_traj_kms[:, :3],
                                true_target_traj_xyz_2=ast_truth_original_eme[:, :3],
                                ems_center_xyz=ems_center_xyz,
                                ems_radius=ems_radius,
                                show_uncertainty=True,
                                show_truth=True,  # green circle
                                show_ems=True,
                                show_fov_cones=True,
                                show_legend=True,
                                show_target_mean_traj=True,
                                show_true_target_traj=True,
                                show_true_target_traj_2=False,
                                show_init_boresights=False,
                                show_current_boresights=False,
                                show_slew_angle_annotations=False,
                                show_agent_name_annotations=True,
                                show_agent_orbit_tracks=True,
                                show_spacecraft_orbit=False,
                                show_coverage=True,

                                Nx=60,
                                Ny=60,
                                Nz=40,

                                # coverage display
                                show_pair_coverage=True,
                                show_triple_coverage=True,
                                pair_only_exact=True,

                                pair_coverage_alpha=0.25,
                                triple_coverage_alpha=0.35,

                                # FOV styling
                                fov_style="surface",  # "surface", "wire", "both"
                                fov_surface_alpha=0.12,
                                fov_surface_color="lightskyblue",
                                fov_n_rays=2,
                                fov_n_circle=64,
                                fov_n_len=20,

                                # styling
                                title=None,
                                label_fontsize=9,
                                label_offset_px=10,
                                slew_label_offset_px=16,
                                fill_alpha=0.10,
                                sparse_wire=True,

                                init_boresight_lw=1.5,
                                init_boresight_alpha=0.95,

                                # slew history
                                slew_history=slew_history,
                                slew_history_line_len=ray_length,
                                slew_history_lw=1.8,
                                slew_history_alpha=0.85,
                                slew_history_cmap="viridis",
                                slew_history_every=1,
                                slew_history_colorbar=True,
                                slew_history_colorbar_label="Slew history step",
                                slew_history_norm_mode="per_agent",
                            )



                #-------------------------
                # Regular OD Step
                # -------------------------
                else:
                    event_type = "regular_update"

                    #######################################################
                    # Gather Tracklet data and Detect
                    ######################################################
                    p_meas_k, n_meas_k, sc_states_k, ast_states_k, epochs_k, detection_res_k = formation.detect(
                        minimoon.curr_state_eme,
                        timer.curr_epoch,
                        n_body_propagator,
                        config
                    )

                    # optional visualization block unchanged
                    confirm_meas = False
                    if confirm_meas:
                        ems_center_xyz = np.array(config["ems"]["p_em"])
                        ems_radius = config["ems"]['R_em']
                        util.plot_detection_geometry_3d(
                            perfect_meas=p_meas_k,
                            noisy_meas=n_meas_k,
                            sc_states=sc_states_k,
                            ast_states=ast_states_k,
                            detection_results=detection_res_k,
                            epochs=epochs_k,
                            los_stride=2,
                            use_true_range_for_los=True,
                            # EMS sphere
                            ems_center_xyz=ems_center_xyz,
                            ems_radius=ems_radius,
                            show_ems=True,
                            ems_alpha=0.10,
                            show_ems_wires=True,
                            title="Detection Geometry",
                            save_path=None,
                            show=False,
                        )
                        two_d_too = False
                        if two_d_too:
                            plane = ("x", "y")
                            plane_str = "".join(plane)
                            util.plot_detection_geometry_2d(
                                perfect_meas=p_meas_k,
                                noisy_meas=n_meas_k,
                                sc_states=sc_states_k,
                                ast_states=ast_states_k,
                                detection_results=detection_res_k,
                                plane=plane,
                                los_stride=2,
                                save_path=None,
                                show=False,
                            )

                    # print(detection_res_k)
                    # plt.show()

                    # -------------------------
                    # Prediction + UKF update only
                    # -------------------------
                    od_time_start = time.time()
                    od_meas_result = process_tracklet_until_update_with_prior_epoch(
                        ukf=ukf,
                        prior_epoch_jdtdb=timer.curr_epoch,
                        n_body_propagator=n_body_propagator,
                        epochs_k=epochs_k,
                        noisy_meas_k=n_meas_k,
                        sc_states_k=sc_states_k,
                        detection_res_k=detection_res_k,
                    )
                    od_time = time.time() - od_time_start

                    had_detection = od_meas_result["had_detection"]
                    detecting_ids = od_meas_result["detecting_ids"]
                    n_detections = od_meas_result["n_detections"]
                    processed_epochs = od_meas_result["processed_epochs"]

                    x_post_k = od_meas_result["posterior_x"]
                    P_post_k = od_meas_result["posterior_P"]
                    update_history_k = od_meas_result["update_history"]

                    formation.currently_detecting = detecting_ids

                    # Update tracking-anchor custody only when there is an actual
                    # detection.  During no-detection / EMS blackout intervals,
                    # the queue is retained but not used unless a detector reappears.
                    if bool(had_detection) and int(n_detections) > 0:
                        tracking_anchor_queue = _update_tracking_anchor_queue(
                            tracking_anchor_queue, detecting_ids
                        )
                        tracking_anchor_sid = int(tracking_anchor_queue[0]) if tracking_anchor_queue else None
                    else:
                        tracking_anchor_sid = None

                    tracking_anchor_mode = "none"
                    tracking_anchor_feasible = int(False)

                    ems_flags = _ems_detection_flags(detection_res_k)
                    all_ems_occluded = bool(ems_flags["all_ems_occluded"])
                    no_detection_reason = ""
                    terminate_for_object_lost = False
                    ems_blackout_prediction_only = False

                    if had_detection and int(n_detections) > 0:
                        no_detection_reason = ""
                        all_ems_occluded = False
                        in_ems_blackout = False
                        pending_reacquisition = False
                        reacquisition_attempt_count = 0
                    else:
                        # EMS blackout dominates: if every spacecraft LOS is EMS-occluded,
                        # do not spend compute on attitude coordination. Just advance by the
                        # configured blackout cadence with prediction only.
                        if all_ems_occluded:
                            no_detection_reason = "ems_blackout_all"
                            event_type = "ems_blackout_prediction_only"
                            in_ems_blackout = True
                            pending_reacquisition = False
                            reacquisition_attempt_count = 0
                            ems_blackout_prediction_only = True
                        elif in_ems_blackout:
                            # EMS just became clear after a blackout. The current no-detection
                            # happened with pre-reacquisition pointings, so allow attitude
                            # coordination once before declaring the object lost.
                            no_detection_reason = "ems_visible_after_blackout_reacquisition"
                            event_type = "ems_reacquisition_attcoord"
                            in_ems_blackout = False
                            pending_reacquisition = True
                        elif pending_reacquisition:
                            # A reacquisition attempt has already been performed. If the next
                            # EMS-visible detection check still finds nothing, terminate unless
                            # the config allows more reacquisition attempts.
                            if (
                                terminate_if_ems_visible_but_not_detected
                                and reacquisition_attempt_count >= max_reacquisition_attempts_after_blackout
                            ):
                                no_detection_reason = "object_lost_after_ems_reacquisition"
                                event_type = "termination_object_lost_after_ems_reacquisition"
                                terminate_for_object_lost = True
                            else:
                                no_detection_reason = "ems_reacquisition_retry_no_detection"
                                event_type = "ems_reacquisition_attcoord"
                        else:
                            no_detection_reason = "object_lost_no_detection"
                            event_type = "termination_object_lost_no_detection"
                            terminate_for_object_lost = bool(terminate_if_ems_visible_but_not_detected)

                    ast_eme_state_kms_during_detection = minimoon.curr_state_eme

                    update_history = od_meas_result["update_history"]
                    processed_epochs = od_meas_result["processed_epochs"]

                    x_est_hist = np.array([entry["x_post"] for entry in update_history])  # (K,6)
                    P_est_hist = np.array([entry["P_post"] for entry in update_history])  # (K,6,6)
                    x_pred_hist = np.array([entry["x_prior"] for entry in update_history])
                    P_pred_hist = np.array([entry["P_prior"] for entry in update_history])

                    if len(processed_epochs) > 0:
                        processed_epoch_first, processed_epoch_last, processed_epoch_count = _first_last_count(processed_epochs)

                    ast_eme_traj_kms_during_detection = n_body_propagator.propagate(
                        ast_eme_state_kms_during_detection,
                        timer.curr_epoch,
                        processed_epochs
                    ) if len(processed_epochs) > 0 else np.empty((0, 6))

                    propped_initial_state = n_body_propagator.propagate(
                        x_est_hist[0], timer.curr_epoch, processed_epochs
                    ) if len(update_history) > 0 else np.empty((0, 6))

                    view_update = False
                    if view_update and len(update_history) > 0:
                        fig, ax = util.plot_od_trajectory_with_measurements_3d(
                            x_est_hist=x_est_hist,
                            ast_true_hist=ast_eme_traj_kms_during_detection,
                            P_est_hist=P_est_hist,
                            noisy_meas=n_meas_k,
                            sc_states=sc_states_k,
                            detection_results=detection_res_k,
                            x_pred_hist=propped_initial_state,
                            d_mahal=3.0,
                            los_stride=2,
                            los_scale_before=0.015,
                            los_scale_after=0.025,
                            title="OD estimate vs truth with LOS and 3σ ellipsoid",
                            save_path=None,
                            show=False,
                        )
                        show_two_d = True
                        if show_two_d:
                            fig, ax = util.plot_od_trajectory_with_measurements_2d(
                                x_est_hist=x_est_hist,
                                ast_true_hist=ast_eme_traj_kms_during_detection,
                                P_est_hist=P_est_hist,
                                noisy_meas=n_meas_k,
                                sc_states=sc_states_k,
                                detection_results=detection_res_k,
                                plane=("x", "y"),
                                x_pred_hist=propped_initial_state,
                                d_mahal=3.0,
                                los_stride=2,
                                los_scale_before=0.015,
                                los_scale_after=0.025,
                                title="OD estimate vs truth with LOS and 3σ ellipse",
                                save_path=None,
                                show=False,
                            )

                    # -----------------------------------------------------
                    # Inner KF diagnostics
                    # -----------------------------------------------------
                    detecting_ids_str = _ids_to_str(detecting_ids)

                    if log_inner_kf and timer.curr_od_index > resume_after_step:
                        for j, entry in enumerate(update_history):
                            row_in = {
                                "run_uid": uid,
                                "master_row_idx": m_idx,
                                "rank": rank,
                                "od_step_idx": int(timer.curr_od_index),
                                "inner_update_idx": j,
                                "epoch_jdtdb": float(processed_epochs[j]) if j < len(processed_epochs) else np.nan,
                                "detecting_ids": detecting_ids_str,
                            }

                            xpr = _flatten_vec(entry.get("x_prior", None), 6)
                            xpo = _flatten_vec(entry.get("x_post", None), 6)
                            Ppr = _safe_diag(entry.get("P_prior", None), 6)
                            Ppo = _safe_diag(entry.get("P_post", None), 6)

                            for i in range(6):
                                row_in[f"x_prior_{i}"] = xpr[i]
                                row_in[f"x_post_{i}"] = xpo[i]
                                row_in[f"P_prior_diag_{i}"] = Ppr[i]
                                row_in[f"P_post_diag_{i}"] = Ppo[i]

                            xtrue_i = [np.nan] * 6
                            if len(ast_eme_traj_kms_during_detection) > j:
                                xtrue_i = _flatten_vec(ast_eme_traj_kms_during_detection[j], 6)
                            for i in range(6):
                                row_in[f"x_true_{i}"] = xtrue_i[i]

                            obs_i = _extract_inner_obs_state(sc_states_k, j, detecting_ids)
                            for i in range(6):
                                row_in[f"observer_state_{i}"] = obs_i[i]

                            true_i, noisy_i = _extract_inner_meas_pair(p_meas_k, n_meas_k, j, detecting_ids)
                            for i in range(2):
                                row_in[f"true_meas_{i}"] = true_i[i]
                                row_in[f"noisy_meas_{i}"] = noisy_i[i]

                            _append_row(inner_csv_path, row_in, inner_header)

                    # No detections are no longer terminal. The UKF/SPKF function
                    # now runs a prediction-only pass through the missed tracklet,
                    # and attitude coordination below continues from that propagated prior.
                    if False and ((not had_detection) or (int(n_detections) == 0)):
                        termination_reason = "no_detection"
                        progress_status = "completed"
                        event_type = "termination_no_detection"

                        sc_states_post = np.asarray(formation.get_spacecraft_states(), dtype=float)
                        sc_pointings_post = np.asarray(formation.get_spacecraft_pointings(), dtype=float)

                        x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                        x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                        P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                        pos_err, vel_err = _best_effort_state_error(x_est, x_true)

                        row_out = {
                            "run_uid": uid,
                            "master_row_idx": m_idx,
                            "rank": rank,
                            "od_step_idx": int(timer.curr_od_index),
                            "event_type": event_type,
                            "termination_reason": termination_reason,
                            "epoch_start_jdtdb": epoch_start,
                            "epoch_end_jdtdb": epoch_start,
                            "processed_epoch_first_jdtdb": processed_epoch_first,
                            "processed_epoch_last_jdtdb": processed_epoch_last,
                            "processed_epoch_count": processed_epoch_count,
                            "had_detection": had_detection,
                            "n_detections": 0,
                            "detecting_ids": "",
                            "od_time_sec": od_time,
                            "attcoord_time_sec": np.nan,
                            "true_meas_len": 0,
                            "true_meas_norm": np.nan,
                            "true_meas_0": np.nan,
                            "true_meas_1": np.nan,
                            "noisy_meas_len": 0,
                            "noisy_meas_norm": np.nan,
                            "noisy_meas_0": np.nan,
                            "noisy_meas_1": np.nan,
                            "pos_err_norm": pos_err,
                            "vel_err_norm": vel_err,
                            "chosen_candidate_idx": np.nan,
                            "chosen_candidate_epoch_jdtdb": np.nan,
                            "progress_status": progress_status,
                        }
                        for i in range(6):
                            row_out[f"x_est_{i}"] = x_est[i]
                            row_out[f"x_true_{i}"] = x_true[i]
                            row_out[f"P_diag_{i}"] = P_diag[i]

                        for sc_id in range(num_sc):
                            pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                            pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                            post_s = _flatten_vec(sc_states_post[sc_id], 6)
                            post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                            for i in range(6):
                                row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                                row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                            for i in range(3):
                                row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                                row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                        if timer.curr_od_index > resume_after_step:
                            _append_row(outer_csv_path, row_out, outer_header)

                        _write_progress(progress_path, {
                            "uid": uid,
                            "last_completed_outer_step": int(timer.curr_od_index),
                            "last_epoch_jdtdb": float(epoch_start),
                            "completed": True,
                            "termination_reason": termination_reason,
                            "outer_csv_path": outer_csv_path,
                        })
                        break

                    # -----------------------------------------------------
                    # Optional OD stop criterion: covariance trace + NIS
                    # -----------------------------------------------------
                    should_stop_od, od_convergence_streak, last_od_stop_metrics, od_stop_reason = _od_stop_metrics_and_decision(
                        config,
                        getattr(ukf, "P", None),
                        update_history,
                        od_convergence_streak,
                    )

                    # A prediction-only no-detection step should not by itself satisfy
                    # OD convergence/termination logic. Keep the metrics for logging,
                    # but require at least one actual detection for stop decisions.
                    if (not had_detection) or (int(n_detections) == 0):
                        should_stop_od = False
                        od_convergence_streak = 0
                        last_od_stop_metrics["convergence_streak"] = 0
                        od_stop_reason = "no_detection_continue"

                    if terminate_for_object_lost:
                        termination_reason = no_detection_reason
                        progress_status = "completed"

                        sc_states_post = np.asarray(formation.get_spacecraft_states(), dtype=float)
                        sc_pointings_post = np.asarray(formation.get_spacecraft_pointings(), dtype=float)

                        x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                        x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                        P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                        pos_err, vel_err = _best_effort_state_error(x_est, x_true)
                        true_meas_len, true_meas_norm, true_pair = _summarize_meas_outer(p_meas_k)
                        noisy_meas_len, noisy_meas_norm, noisy_pair = _summarize_meas_outer(n_meas_k)

                        row_out = {
                            "run_uid": uid,
                            "master_row_idx": m_idx,
                            "rank": rank,
                            "od_step_idx": int(timer.curr_od_index),
                            "event_type": event_type,
                            "termination_reason": termination_reason,
                            "epoch_start_jdtdb": epoch_start,
                            "epoch_end_jdtdb": float(processed_epoch_last) if np.isfinite(processed_epoch_last) else epoch_start,
                            "processed_epoch_first_jdtdb": processed_epoch_first,
                            "processed_epoch_last_jdtdb": processed_epoch_last,
                            "processed_epoch_count": processed_epoch_count,
                            "had_detection": had_detection,
                            "n_detections": n_detections,
                            "detecting_ids": detecting_ids_str,
                            "no_detection_reason": no_detection_reason,
                            "all_ems_occluded": int(bool(all_ems_occluded)),
                            "in_ems_blackout": int(bool(in_ems_blackout)),
                            "pending_reacquisition": int(bool(pending_reacquisition)),
                            "reacquisition_attempt_count": int(reacquisition_attempt_count),
                            "tracking_anchor_queue": _tracking_queue_to_str(tracking_anchor_queue),
                            "tracking_anchor_sid": (np.nan if tracking_anchor_sid is None else int(tracking_anchor_sid)),
                            "tracking_anchor_mode": tracking_anchor_mode,
                            "tracking_anchor_feasible": int(bool(tracking_anchor_feasible)),
                            "od_time_sec": od_time,
                            "attcoord_time_sec": np.nan,
                            "true_meas_len": true_meas_len,
                            "true_meas_norm": true_meas_norm,
                            "true_meas_0": true_pair[0],
                            "true_meas_1": true_pair[1],
                            "noisy_meas_len": noisy_meas_len,
                            "noisy_meas_norm": noisy_meas_norm,
                            "noisy_meas_0": noisy_pair[0],
                            "noisy_meas_1": noisy_pair[1],
                            "P_pos_trace": last_od_stop_metrics["trace_pos"],
                            "P_vel_trace": last_od_stop_metrics["trace_vel"],
                            "P_pos_sigma_3d": last_od_stop_metrics["sigma_pos_3d"],
                            "P_vel_sigma_3d": last_od_stop_metrics["sigma_vel_3d"],
                            "NIS_last": last_od_stop_metrics["nis_last"],
                            "NIS_mean": last_od_stop_metrics["nis_mean"],
                            "NIS_count": last_od_stop_metrics["nis_count"],
                            "OD_convergence_streak": last_od_stop_metrics["convergence_streak"],
                            "pos_err_norm": pos_err,
                            "vel_err_norm": vel_err,
                            "chosen_candidate_idx": np.nan,
                            "chosen_candidate_epoch_jdtdb": np.nan,
                            "progress_status": progress_status,
                        }
                        for i in range(6):
                            row_out[f"x_est_{i}"] = x_est[i]
                            row_out[f"x_true_{i}"] = x_true[i]
                            row_out[f"P_diag_{i}"] = P_diag[i]
                        for sc_id in range(num_sc):
                            pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                            pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                            post_s = _flatten_vec(sc_states_post[sc_id], 6)
                            post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                            for i in range(6):
                                row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                                row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                            for i in range(3):
                                row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                                row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                        if timer.curr_od_index > resume_after_step:
                            _append_row(outer_csv_path, row_out, outer_header)

                        _write_progress(progress_path, {
                            "uid": uid,
                            "last_completed_outer_step": int(timer.curr_od_index),
                            "last_epoch_jdtdb": float(row_out["epoch_end_jdtdb"]),
                            "completed": True,
                            "termination_reason": termination_reason,
                            "outer_csv_path": outer_csv_path,
                        })
                        break

                    if ems_blackout_prediction_only:
                        progress_status = "running"
                        termination_reason = ""
                        attcoord_time = 0.0
                        chosen_candidate_idx = np.nan
                        chosen_candidate_epoch_jdtdb = np.nan

                        t_after_tracklet = float(processed_epoch_last) if np.isfinite(processed_epoch_last) else float(timer.curr_epoch)
                        target_epoch = float(timer.curr_epoch) + float(ems_blackout_dt_sec) / 86400.0
                        target_epoch = max(target_epoch, t_after_tracklet)
                        target_epoch = min(target_epoch, float(timer.end_time))

                        # The no-detection UKF routine already predicted through the tracklet.
                        # If the configured blackout step goes beyond that, keep predicting
                        # to the blackout target epoch.
                        if target_epoch > t_after_tracklet + 1e-15:
                            ukf.predict(
                                t_after_tracklet,
                                target_epoch,
                                n_body_propagator.propagate_multiple_objects
                            )
                            processed_epochs = np.append(np.asarray(processed_epochs, dtype=float), target_epoch)
                            processed_epoch_first, processed_epoch_last, processed_epoch_count = _first_last_count(processed_epochs)

                        # Propagate truth/minimoon and spacecraft states to the same blackout target.
                        ast_blackout = n_body_propagator.propagate(
                            minimoon.curr_state_eme,
                            timer.curr_epoch,
                            np.asarray([target_epoch], dtype=float),
                        )
                        ast_blackout = np.asarray(ast_blackout, dtype=float).reshape(-1, 6)

                        sc_blackout, anchor_info_blackout = util.piecewise_anchor_and_propagate_spacecraft_trajs(
                            formation=formation, minimoon=minimoon, timer=timer,
                            t_targets_jdtdb=np.asarray([target_epoch], dtype=float),
                            n_body_propagator=n_body_propagator, return_anchor_info=True
                        )
                        sc_blackout = np.asarray(sc_blackout, dtype=float).reshape(1, num_sc, 6)

                        formation.set_spacecraft_states(sc_blackout[0, :, :])
                        # Keep existing pointings during EMS blackout.
                        minimoon.set_state(ast_blackout[0, :])

                        k_anchor_best = int(anchor_info_blackout["anchor_k_of_t"][0])
                        t_anchor_best = float(anchor_info_blackout["anchor_epoch_of_t"][0])
                        timer.set_attcoord_time(0.0)
                        timer.set_slew_time(0.0)
                        timer.step(target_epoch, k_anchor_best, t_anchor_best)

                        sc_states_post = np.asarray(formation.get_spacecraft_states(), dtype=float)
                        sc_pointings_post = np.asarray(formation.get_spacecraft_pointings(), dtype=float)

                        x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                        x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                        P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                        pos_err, vel_err = _best_effort_state_error(x_est, x_true)
                        true_meas_len, true_meas_norm, true_pair = _summarize_meas_outer(p_meas_k)
                        noisy_meas_len, noisy_meas_norm, noisy_pair = _summarize_meas_outer(n_meas_k)

                        row_out = {
                            "run_uid": uid,
                            "master_row_idx": m_idx,
                            "rank": rank,
                            "od_step_idx": int(timer.curr_od_index),
                            "event_type": event_type,
                            "termination_reason": termination_reason,
                            "epoch_start_jdtdb": epoch_start,
                            "epoch_end_jdtdb": float(timer.curr_epoch),
                            "processed_epoch_first_jdtdb": processed_epoch_first,
                            "processed_epoch_last_jdtdb": processed_epoch_last,
                            "processed_epoch_count": processed_epoch_count,
                            "had_detection": had_detection,
                            "n_detections": n_detections,
                            "detecting_ids": detecting_ids_str,
                            "no_detection_reason": no_detection_reason,
                            "all_ems_occluded": int(bool(all_ems_occluded)),
                            "in_ems_blackout": int(bool(in_ems_blackout)),
                            "pending_reacquisition": int(bool(pending_reacquisition)),
                            "reacquisition_attempt_count": int(reacquisition_attempt_count),
                            "tracking_anchor_queue": _tracking_queue_to_str(tracking_anchor_queue),
                            "tracking_anchor_sid": (np.nan if tracking_anchor_sid is None else int(tracking_anchor_sid)),
                            "tracking_anchor_mode": tracking_anchor_mode,
                            "tracking_anchor_feasible": int(bool(tracking_anchor_feasible)),
                            "od_time_sec": od_time,
                            "attcoord_time_sec": attcoord_time,
                            "true_meas_len": true_meas_len,
                            "true_meas_norm": true_meas_norm,
                            "true_meas_0": true_pair[0],
                            "true_meas_1": true_pair[1],
                            "noisy_meas_len": noisy_meas_len,
                            "noisy_meas_norm": noisy_meas_norm,
                            "noisy_meas_0": noisy_pair[0],
                            "noisy_meas_1": noisy_pair[1],
                            "P_pos_trace": last_od_stop_metrics["trace_pos"],
                            "P_vel_trace": last_od_stop_metrics["trace_vel"],
                            "P_pos_sigma_3d": last_od_stop_metrics["sigma_pos_3d"],
                            "P_vel_sigma_3d": last_od_stop_metrics["sigma_vel_3d"],
                            "NIS_last": last_od_stop_metrics["nis_last"],
                            "NIS_mean": last_od_stop_metrics["nis_mean"],
                            "NIS_count": last_od_stop_metrics["nis_count"],
                            "OD_convergence_streak": last_od_stop_metrics["convergence_streak"],
                            "pos_err_norm": pos_err,
                            "vel_err_norm": vel_err,
                            "chosen_candidate_idx": chosen_candidate_idx,
                            "chosen_candidate_epoch_jdtdb": chosen_candidate_epoch_jdtdb,
                            "progress_status": progress_status,
                        }
                        for i in range(6):
                            row_out[f"x_est_{i}"] = x_est[i]
                            row_out[f"x_true_{i}"] = x_true[i]
                            row_out[f"P_diag_{i}"] = P_diag[i]
                        for sc_id in range(num_sc):
                            pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                            pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                            post_s = _flatten_vec(sc_states_post[sc_id], 6)
                            post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                            for i in range(6):
                                row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                                row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                            for i in range(3):
                                row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                                row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                        if timer.curr_od_index > resume_after_step:
                            _append_row(outer_csv_path, row_out, outer_header)
                            _write_progress(progress_path, {
                                "uid": uid,
                                "last_completed_outer_step": int(timer.curr_od_index),
                                "last_epoch_jdtdb": float(timer.curr_epoch),
                                "completed": False,
                                "termination_reason": "",
                                "outer_csv_path": outer_csv_path,
                                "checkpoint_path": (checkpoint_path if checkpoint_enabled else None),
                            })
                            if checkpoint_enabled:
                                _save_od_checkpoint(
                                    checkpoint_path, uid=uid, m_idx=m_idx, rank=rank, ukf=ukf, timer=timer,
                                    minimoon=minimoon, formation=formation,
                                    od_convergence_streak=od_convergence_streak,
                                    last_od_stop_metrics=last_od_stop_metrics,
                                    last_completed_outer_step=int(timer.curr_od_index),
                                    outer_csv_path=outer_csv_path,
                                    termination_reason="",
                                    progress_status=progress_status,
                                    no_detection_state=_no_detection_state_dict(
                                        in_ems_blackout, pending_reacquisition, reacquisition_attempt_count
                                    ),
                                    tracking_state=_tracking_state_dict(tracking_anchor_queue, tracking_anchor_sid),
                                )

                        print_od_status(
                            timer=timer,
                            ukf=ukf,
                            minimoon=minimoon,
                            formation=formation,
                            x_true=None,
                            n_detections=n_detections,
                            status_every=status_every,
                            prefix=f"[OD r{rank} uid={uid}]",
                        )
                        continue

                    if should_stop_od:
                        termination_reason = od_stop_reason
                        progress_status = "completed"
                        event_type = "termination_convergence"

                        sc_states_post = np.asarray(formation.get_spacecraft_states(), dtype=float)
                        sc_pointings_post = np.asarray(formation.get_spacecraft_pointings(), dtype=float)

                        x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                        x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                        P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                        pos_err, vel_err = _best_effort_state_error(x_est, x_true)
                        true_meas_len, true_meas_norm, true_pair = _summarize_meas_outer(p_meas_k)
                        noisy_meas_len, noisy_meas_norm, noisy_pair = _summarize_meas_outer(n_meas_k)

                        row_out = {
                            "run_uid": uid,
                            "master_row_idx": m_idx,
                            "rank": rank,
                            "od_step_idx": int(timer.curr_od_index),
                            "event_type": event_type,
                            "termination_reason": termination_reason,
                            "epoch_start_jdtdb": epoch_start,
                            "epoch_end_jdtdb": float(processed_epoch_last) if np.isfinite(processed_epoch_last) else epoch_start,
                            "processed_epoch_first_jdtdb": processed_epoch_first,
                            "processed_epoch_last_jdtdb": processed_epoch_last,
                            "processed_epoch_count": processed_epoch_count,
                            "had_detection": had_detection,
                            "n_detections": n_detections,
                            "detecting_ids": detecting_ids_str,
                            "od_time_sec": od_time,
                            "attcoord_time_sec": np.nan,
                            "true_meas_len": true_meas_len,
                            "true_meas_norm": true_meas_norm,
                            "true_meas_0": true_pair[0],
                            "true_meas_1": true_pair[1],
                            "noisy_meas_len": noisy_meas_len,
                            "noisy_meas_norm": noisy_meas_norm,
                            "noisy_meas_0": noisy_pair[0],
                            "noisy_meas_1": noisy_pair[1],
                            "P_pos_trace": last_od_stop_metrics["trace_pos"],
                            "P_vel_trace": last_od_stop_metrics["trace_vel"],
                            "P_pos_sigma_3d": last_od_stop_metrics["sigma_pos_3d"],
                            "P_vel_sigma_3d": last_od_stop_metrics["sigma_vel_3d"],
                            "NIS_last": last_od_stop_metrics["nis_last"],
                            "NIS_mean": last_od_stop_metrics["nis_mean"],
                            "NIS_count": last_od_stop_metrics["nis_count"],
                            "OD_convergence_streak": last_od_stop_metrics["convergence_streak"],
                            "pos_err_norm": pos_err,
                            "vel_err_norm": vel_err,
                            "chosen_candidate_idx": np.nan,
                            "chosen_candidate_epoch_jdtdb": np.nan,
                            "progress_status": progress_status,
                        }
                        for i in range(6):
                            row_out[f"x_est_{i}"] = x_est[i]
                            row_out[f"x_true_{i}"] = x_true[i]
                            row_out[f"P_diag_{i}"] = P_diag[i]

                        for sc_id in range(num_sc):
                            pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                            pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                            post_s = _flatten_vec(sc_states_post[sc_id], 6)
                            post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                            for i in range(6):
                                row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                                row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                            for i in range(3):
                                row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                                row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                        if timer.curr_od_index > resume_after_step:
                            _append_row(outer_csv_path, row_out, outer_header)

                        _write_progress(progress_path, {
                            "uid": uid,
                            "last_completed_outer_step": int(timer.curr_od_index),
                            "last_epoch_jdtdb": float(row_out["epoch_end_jdtdb"]),
                            "completed": True,
                            "termination_reason": termination_reason,
                            "outer_csv_path": outer_csv_path,
                        })
                        break

                    # -----------------------------------------------------
                    # Perform Attitude Coordination - with uncertainty growth
                    # -------------------------------------------------------
                    attcoord_startime = time.time()

                    timer.set_attcoord_searchtimes(od_time=od_time)

                    x_ts, P_ts = ukf.propagate_priors(
                        epochs_k[-1],
                        timer.attcoord_searchtimes_jdtdb,
                        n_body_propagator.propagate_multiple_objects
                    )

                    sc_eme_states_kms_piecewise, anchor_info = util.piecewise_anchor_and_propagate_spacecraft_trajs(
                        formation=formation, minimoon=minimoon, timer=timer,
                        t_targets_jdtdb=timer.attcoord_searchtimes_jdtdb,
                        n_body_propagator=n_body_propagator, return_anchor_info=True
                    )

                    ast_eme_state_kms = minimoon.curr_state_eme
                    ast_eme_traj_kms = n_body_propagator.propagate(ast_eme_state_kms, timer.curr_epoch,
                                                                   timer.attcoord_searchtimes_jdtdb)

                    ast_truth_original_eclip = np.array(minimoon.orbit.loc[:, ["Geo x", "Geo y", "Geo z", "Geo vx",
                                                                               "Geo vy", "Geo vz"]])
                    ast_truth_original_eclip[:, :3] *= config['AU_TO_M'] / 1000
                    ast_truth_original_eclip[:, 3:] *= config['AU_TO_M'] / 1000 / config['SECONDS_PER_DAY']
                    ast_truth_original_eme = util.geo_eclip_to_geo_eme_generic(ast_truth_original_eclip,
                                                                               hint=("time", "state"))

                    viz_prop_flag = False
                    if viz_prop_flag:
                        util.plot_priors_positions_and_cov_2d(
                            x_ts,
                            P_ts,
                            sc_trajs_km2=sc_eme_states_kms_piecewise,
                            stride=config['two_d_prop']['stride'],
                            n_std=config['two_d_prop']['stride'],
                            planes=("xy", "xz", "yz"),
                            title_prefix="Asteroid prior + spacecraft"
                        )

                        plot_matched_flag = False
                        if plot_matched_flag:
                            util.plot_matched_trajectory_full_range(
                                formation=formation,
                                idx_start=0,
                                idx_stop=2400,
                                frame="GEO_EME",
                                plot_3d=True,
                                plot_xy=False
                            )

                    sc_pointings_eme = formation.get_spacecraft_pointings()

                    sc0 = formation.spacecraft[0]
                    theta_h_rad = util.fov_deg2_to_half_angle_rad(sc0.fov)
                    tau_max = sc0.reaction_wheel_torque
                    h_max = sc0.reaction_wheel_momentum
                    m_m = sc0.mass
                    l_m = sc0.length
                    m_t = sc0.telescope_mass
                    d_t = sc0.telescope_diameter
                    l_t = sc0.telescope_length
                    z_0 = sc0.telescope_offset
                    I_max = (1 / 6) * m_m * (l_m) ** 2 + (1 / 12) * m_t * (
                                3 * (d_t / 2) ** 2 + l_t ** 2) + m_t * z_0 ** 2
                    alpha_max = 1.63 * tau_max / I_max
                    omega_max = 1.63 * h_max / I_max

                    use_tracking = bool(config_global.get('use_tracking', False))
                    detecting_list = list(formation.currently_detecting)
                    detecting_u = sc_pointings_eme[detecting_list, :] if len(detecting_list) > 0 else np.empty((0, 3))

                    # Tracking-anchor policy: keep the oldest still-detecting spacecraft
                    # as custody anchor.  The attitude coordinator recomputes that
                    # anchor's fixed boresight as LOS-to-mean for each candidate epoch.
                    anchor_candidates = list(tracking_anchor_queue) if (preserve_detector_anchor and len(detecting_list) > 0) else []
                    if len(anchor_candidates) == 0 and (not use_tracking):
                        # Backward-compatible fallback for old non-tracking behavior.
                        anchor_candidates = detecting_list[:1] if len(detecting_list) == 1 else []

                    res_kcoverage = result_mean = result_kcoverage_series = result_mean_series = None
                    mean_time = 0.0
                    chosen_anchor_sid = None
                    chosen_anchor_mode = "none"
                    chosen_anchor_feasible = False

                    def _call_attcoord_with_anchor(anchor_sid):
                        use_anchor = anchor_sid is not None
                        fixed_u = None
                        fixed_mode = tracking_anchor_fixed_mode
                        if use_anchor and fixed_mode != "mean_los_per_epoch":
                            fixed_u = sc_pointings_eme[int(anchor_sid), :]
                        return attitude_coordination.step(
                            sc_eme_states_kms_piecewise[:, :, :3],
                            sc_pointings_eme,
                            x_ts[:, :3],
                            P_ts[:, :3, :3],
                            timer.attcoord_searchtimes,
                            theta_h_rad,
                            alpha_max,
                            omega_max,
                            detecting_u,
                            detecting_list,
                            d_M=config['d_mahal'],
                            use_fixed_agent=bool(use_anchor),
                            fixed_agent_idx=(None if not use_anchor else int(anchor_sid)),
                            fixed_agent_u=fixed_u,
                            fixed_agent_u_mode=fixed_mode,
                            coverage_point=ast_eme_traj_kms[:, :3],
                        )

                    # Try the oldest still-detecting anchor first.  If the entire
                    # candidate horizon is infeasible for that anchor and config allows
                    # it, promote to the next-oldest detector.
                    if preserve_detector_anchor and len(anchor_candidates) > 0:
                        for anchor_sid_try in anchor_candidates:
                            tmp = _call_attcoord_with_anchor(int(anchor_sid_try))
                            res_try = tmp[0]
                            if getattr(res_try, "u_cmd", None) is not None and np.isfinite(float(getattr(res_try, "chosen_dt", np.nan))):
                                res_kcoverage, result_mean, result_kcoverage_series, result_mean_series, mean_time = tmp
                                chosen_anchor_sid = int(anchor_sid_try)
                                chosen_anchor_mode = tracking_anchor_fixed_mode
                                chosen_anchor_feasible = True
                                break
                            if tracking_anchor_fallback != "next_oldest":
                                res_kcoverage, result_mean, result_kcoverage_series, result_mean_series, mean_time = tmp
                                chosen_anchor_sid = int(anchor_sid_try)
                                chosen_anchor_mode = tracking_anchor_fixed_mode
                                chosen_anchor_feasible = False
                                break

                    if res_kcoverage is None:
                        # No usable anchor or anchor preservation disabled: optimize freely.
                        res_kcoverage, result_mean, result_kcoverage_series, result_mean_series, mean_time = attitude_coordination.step(
                            sc_eme_states_kms_piecewise[:, :, :3],
                            sc_pointings_eme,
                            x_ts[:, :3],
                            P_ts[:, :3, :3],
                            timer.attcoord_searchtimes,
                            theta_h_rad,
                            alpha_max,
                            omega_max,
                            detecting_u,
                            detecting_list,
                            d_M=config['d_mahal'],
                            use_fixed_agent=False,
                            fixed_agent_idx=None,
                            fixed_agent_u=None,
                            fixed_agent_u_mode="provided",
                            coverage_point=ast_eme_traj_kms[:, :3],
                        )
                        chosen_anchor_sid = None
                        chosen_anchor_mode = "free"
                        chosen_anchor_feasible = False

                    tracking_anchor_sid = chosen_anchor_sid
                    tracking_anchor_mode = chosen_anchor_mode
                    tracking_anchor_feasible = int(bool(chosen_anchor_feasible))

                    attcoord_endtime = time.time()
                    timer.set_attcoord_time(attcoord_endtime - attcoord_startime - mean_time)
                    attcoord_time = attcoord_endtime - attcoord_startime - mean_time

                    timer.set_slew_time(res_kcoverage.chosen_dt)

                    best_epoch = res_kcoverage.chosen_dt
                    best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                    best_epoch_jdtdb = float(timer.attcoord_searchtimes_jdtdb[best_idx])
                    k_anchor_best = int(anchor_info["anchor_k_of_t"][best_idx])
                    t_anchor_best = float(anchor_info["anchor_epoch_of_t"][best_idx])

                    best_attitudes = res_kcoverage.u_cmd
                    formation.set_spacecraft_pointings(best_attitudes)

                    best_positions = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :])
                    formation.set_spacecraft_states(best_positions)

                    ukf.x = np.squeeze(x_ts[best_idx, :])
                    ukf.P = np.squeeze(P_ts[best_idx, :, :])

                    minimoon.set_state(np.squeeze(ast_eme_traj_kms[best_idx, :]))

                    timer.step(best_epoch_jdtdb, k_anchor_best, t_anchor_best)

                    chosen_candidate_idx = best_idx
                    chosen_candidate_epoch_jdtdb = best_epoch_jdtdb

                    # detailed attcoord logging
                    if log_attcoord and timer.curr_od_index > resume_after_step:
                        for cand_idx in range(len(timer.attcoord_searchtimes_jdtdb)):
                            row_att = {
                                "run_uid": uid,
                                "master_row_idx": m_idx,
                                "rank": rank,
                                "od_step_idx": int(timer.curr_od_index),
                                "candidate_idx": cand_idx,
                                "candidate_epoch_jdtdb": float(timer.attcoord_searchtimes_jdtdb[cand_idx]),
                                "is_chosen": int(cand_idx == best_idx),
                            }
                            xpred = _flatten_vec(x_ts[cand_idx], 6)
                            Ppred = _safe_diag(P_ts[cand_idx], 6)
                            for i in range(6):
                                row_att[f"x_pred_{i}"] = xpred[i]
                                row_att[f"P_pred_diag_{i}"] = Ppred[i]

                            sc_cand = np.asarray(sc_eme_states_kms_piecewise[cand_idx], dtype=float)
                            for sc_id in range(num_sc):
                                vals = _flatten_vec(sc_cand[sc_id], 6)
                                for i in range(6):
                                    row_att[f"sc{sc_id}_state_{i}"] = vals[i]

                            u_cand = None
                            J_cand = np.nan
                            coverage_cand = np.nan
                            if cand_idx < len(result_kcoverage_series):
                                if isinstance(result_kcoverage_series[cand_idx], dict):
                                    u_cand = result_kcoverage_series[cand_idx].get("u", None)
                                    J_cand = _safe_scalar(result_kcoverage_series[cand_idx].get("J", np.nan))
                                    coverage_cand = _safe_scalar(result_kcoverage_series[cand_idx].get("kcoverage", np.nan))
                            if u_cand is None:
                                u_cand = best_attitudes if cand_idx == best_idx else np.full((num_sc, 3), np.nan)
                            u_cand = np.asarray(u_cand, dtype=float)
                            for sc_id in range(num_sc):
                                vals = _flatten_vec(u_cand[sc_id], 3)
                                for i in range(3):
                                    row_att[f"u_cmd_sc{sc_id}_{i}"] = vals[i]

                            row_att["J"] = J_cand
                            row_att["coverage_score"] = coverage_cand
                            _append_row(attcoord_csv_path, row_att, attcoord_header)

                    # optimizer logging based on actual history structure
                    if log_optimizer and hasattr(res_kcoverage, "extra") and isinstance(res_kcoverage.extra, dict):
                        hist = res_kcoverage.extra.get("history", [])
                        if timer.curr_od_index > resume_after_step:
                            for h_idx, entry in enumerate(hist):
                                row_opt = {
                                    "run_uid": uid,
                                    "master_row_idx": m_idx,
                                    "rank": rank,
                                    "od_step_idx": int(timer.curr_od_index),
                                    "optimizer_row_idx": h_idx,
                                    "restart_idx": entry.get("restart", np.nan),
                                    "use_fixed_agent": entry.get("use_fixed_agent", np.nan),
                                    "idx_fix": entry.get("idx_fix", np.nan),
                                    "J": _safe_scalar(entry.get("J", np.nan)),
                                }

                                x_full = _flatten_vec(entry.get("x", None), 12)
                                x_free = _flatten_vec(entry.get("x_free", None), 12)
                                for i in range(12):
                                    row_opt[f"x_param_{i}"] = x_full[i]
                                    row_opt[f"x_free_param_{i}"] = x_free[i]

                                u_hist = np.asarray(entry.get("u", np.full((num_sc, 3), np.nan)), dtype=float)
                                if u_hist.ndim == 1:
                                    u_hist = u_hist.reshape(-1, 3)
                                for sc_id in range(min(num_sc, u_hist.shape[0])):
                                    vals = _flatten_vec(u_hist[sc_id], 3)
                                    for i in range(3):
                                        row_opt[f"u_sc{sc_id}_{i}"] = vals[i]

                                slews_hist = np.asarray(entry.get("slew", np.full((num_sc,), np.nan))).reshape(-1)
                                for sc_id in range(num_sc):
                                    row_opt[f"slew_sc{sc_id}"] = float(slews_hist[sc_id]) if sc_id < slews_hist.size else np.nan

                                _append_row(optimizer_csv_path, row_opt, optimizer_header)

                    # visualize att_coord result
                    # The visualization block assumes at least one detecting spacecraft
                    # in a few diagnostic calculations, so skip it for prediction-only
                    # no-detection steps.
                    att_coord_viz_flag = False
                    if att_coord_viz_flag and had_detection:
                        best_epoch = res_kcoverage.chosen_dt
                        best_idx = np.where(timer.attcoord_searchtimes == best_epoch)[0]

                        theta_h_rad = util.fov_deg2_to_half_angle_rad(config["fov"])
                        ems_center_xy = np.array(config["ems"]["p_em"][:2])
                        ems_center_xyz = np.array(config["ems"]["p_em"])
                        ems_radius = config["ems"]['R_em']
                        ray_length = config['ray_length']

                        sc_eme_ae_kms = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :3])
                        sc_pointing_eme_cartesian = res_kcoverage.u_cmd
                        ang_eme = util.proj_angle_xy_from_plus_x_ccw(sc_pointing_eme_cartesian)
                        ast_truth_eme = np.squeeze(ast_eme_traj_kms[best_idx, :3])

                        ast_iod_eme = np.squeeze(x_ts[best_idx, :3])
                        P_cart_eme = np.squeeze(P_ts[best_idx, :3, :3])

                        ast_truth_eme = np.asarray(ast_truth_eme).reshape(-1)

                        best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                        T = len(timer.attcoord_searchtimes)

                        two_d_pot = False
                        if two_d_pot:
                            def is_valid(idx):
                                J = result_kcoverage_series[idx]["J"]
                                return not (J is None or (isinstance(J, float) and math.isnan(J)))

                            cands = [best_idx, max(0, best_idx - 1), min(T - 1, best_idx + 1)]
                            selected = []

                            for c in cands:
                                if c not in selected and is_valid(c):
                                    selected.append(c)

                            if len(selected) < 3:
                                for c in range(T):
                                    if len(selected) >= 3:
                                        break
                                    if c not in selected and is_valid(c):
                                        selected.append(c)

                            if len(selected) < 3:
                                for c in range(T):
                                    if len(selected) >= 3:
                                        break
                                    if c not in selected:
                                        selected.append(c)

                            planes = [((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ")]

                            def cov2_from_cov3(P3, axes):
                                i, j = axes
                                return P3[np.ix_([i, j], [i, j])]

                            def angles_in_plane(u_cmd_xyz, axes):
                                a, b = axes
                                u = np.asarray(u_cmd_xyz, dtype=float)
                                u2 = u[:, [a, b]]
                                return np.arctan2(u2[:, 1], u2[:, 0])

                            for idx in selected:
                                epoch = timer.attcoord_searchtimes[idx]
                                sc_xyz = np.asarray(sc_eme_states_kms_piecewise[idx, :, :3], dtype=float)
                                ast_truth = np.asarray(ast_eme_traj_kms[idx, :3], dtype=float).reshape(3,)
                                ast_mean = np.asarray(x_ts[idx, :3], dtype=float).reshape(3,)
                                P3 = np.asarray(P_ts[idx, :3, :3], dtype=float).reshape(3, 3)
                                u_cmd_xyz = np.asarray(result_kcoverage_series[idx]["u"], dtype=float)
                                u_curr_xyz = np.asarray(sc_pointings_eme, dtype=float)

                                fig, axes = plt.subplots(1, 3, figsize=(8, 14))
                                fig.suptitle(f"2D Projections @ epoch {epoch}", y=0.99)

                                for ax, (axpair, name) in zip(axes, planes):
                                    agents_2d = sc_xyz[:, list(axpair)]
                                    u_curr_2d = u_curr_xyz[:, list(axpair)]
                                    mean_2d = ast_mean[list(axpair)]
                                    truth_2d = ast_truth[list(axpair)]
                                    cov_2d = cov2_from_cov3(P3, axpair)
                                    mean_traj_2d = np.asarray(x_ts[:, :3], dtype=float)[:, list(axpair)]
                                    truth_traj_2d = np.asarray(ast_eme_traj_kms[:, :3], dtype=float)[:, list(axpair)]
                                    ems_center_2d = np.asarray(ems_center_xyz, dtype=float)[list(axpair)]
                                    ang_2d = angles_in_plane(u_cmd_xyz, axpair)

                                    util.plot_od_scenario_2d(
                                        t_label=f"{epoch} ({name})",
                                        agents_xy=agents_2d,
                                        pointing_angles_rad=ang_2d,
                                        theta_h_rad=theta_h_rad,
                                        ray_length=ray_length * 2,
                                        u_curr_agents_xy=u_curr_2d,
                                        boresight_line_len=ray_length * 0.1,
                                        target_mean_xy=mean_2d,
                                        target_mean_xy_traj=mean_traj_2d,
                                        target_cov_xy=cov_2d,
                                        d_mahal=config['d_mahal'],
                                        true_target_xy=truth_2d,
                                        true_target_xy_traj=truth_traj_2d,
                                        ems_center_xy=ems_center_2d,
                                        ems_radius=ems_radius,
                                        xlim=config['two_d_prop']['xlim'], ylim=config['two_d_prop']['ylim'],
                                        agent_orbit_tracks_xy=None,
                                        ax=ax,
                                        title=None
                                    )

                                plt.tight_layout()



                        agents_xyz = sc_eme_ae_kms
                        target_cov_xyz = P_cart_eme

                        def range_sigma_from_cov(P_eci, r_obj, r_obs):
                            """
                            P_eci : 6x6 or 3x3 covariance in inertial frame
                            r_obj : object inertial position, shape (3,)
                            r_obs : observer inertial position, shape (3,)

                            Returns range standard deviation and variance.
                            """
                            P_r = P_eci[:3, :3] if P_eci.shape == (6, 6) else P_eci

                            rho_vec = np.asarray(r_obj) - np.asarray(r_obs)
                            rho_hat = rho_vec / np.linalg.norm(rho_vec)

                            var_rho = rho_hat @ P_r @ rho_hat
                            sigma_rho = np.sqrt(max(var_rho, 0.0))

                            return sigma_rho, var_rho

                        sigma_rho, var_rho = range_sigma_from_cov(P_cart_eme, ast_iod_eme[:3],
                                                                  agents_xyz[formation.currently_detecting[0]])
                        print(sigma_rho, var_rho)

                        def build_slew_history_from_opt_series(opt_series, ids, *, key_u="u", normalize=True):
                            ids = [int(i) for i in ids]
                            out = {i: [] for i in ids}

                            for row_idx, rowh in enumerate(opt_series):
                                if key_u not in rowh:
                                    raise KeyError(f"Row {row_idx} missing key '{key_u}'")
                                U = np.asarray(rowh[key_u], dtype=float)
                                if U.ndim != 2 or U.shape[1] != 3:
                                    raise ValueError(f"Row {row_idx} '{key_u}' must be (M,3), got {U.shape}")
                                M = U.shape[0]
                                for i in ids:
                                    if not (0 <= i < M):
                                        raise IndexError(f"Row {row_idx}: id {i} out of range for M={M}")
                                    out[i].append(U[i].copy())

                            for i in ids:
                                Ui = np.asarray(out[i], dtype=float)
                                if Ui.size == 0:
                                    Ui = Ui.reshape(0, 3)

                                if normalize and Ui.shape[0] > 0:
                                    n = np.linalg.norm(Ui, axis=1, keepdims=True)
                                    n = np.maximum(n, 1e-12)
                                    Ui = Ui / n

                                out[i] = Ui

                            return out

                        ids = config['three_d_prop'].get('history_ids')
                        if ids is None:
                            ids = list(range(config['num_spacecraft']))

                        slew_history = build_slew_history_from_opt_series(res_kcoverage.extra["history"], ids)

                        fig, ax = util.plot_od_scenario_3d_new(
                            t_label=best_epoch,
                            agents_xyz=agents_xyz,
                            u_opt_agents_xyz=sc_pointing_eme_cartesian,
                            theta_h_rad=theta_h_rad,
                            ray_length=ray_length,
                            u_curr_agents_xyz=sc_pointings_eme,
                            boresight_line_len=ray_length * 0.1,
                            u_init_agents_xyz=None,
                            init_boresight_line_len=ray_length * 1.5,
                            agent_orbit_tracks_xyz=[sc_eme_states_kms_piecewise[:, i, :3] for i in
                                                    range(config['num_spacecraft'])],
                            spacecraft_orbit_xyz=None,
                            xlim=config['three_d_prop']['xlim'], ylim=config['three_d_prop']['ylim'],
                            zlim=config['three_d_prop']['zlim'],
                            target_mean_xyz=ast_iod_eme[:3],
                            target_cov_xyz=target_cov_xyz,
                            d_mahal=config['d_mahal'],
                            true_target_xyz=ast_truth_eme[:3],
                            target_mean_traj_xyz=x_ts[:, :3],
                            true_target_traj_xyz=ast_eme_traj_kms[:, :3],
                            true_target_traj_xyz_2=ast_truth_original_eme[:, :3],
                            ems_center_xyz=ems_center_xyz,
                            ems_radius=ems_radius,
                            show_uncertainty=True,
                            show_truth=True,  # green circle
                            show_ems=True,
                            show_fov_cones=True,
                            show_legend=True,
                            show_target_mean_traj=True,
                            show_true_target_traj=True,
                            show_true_target_traj_2=False,
                            show_init_boresights=False,
                            show_current_boresights=False,
                            show_slew_angle_annotations=False,
                            show_agent_name_annotations=True,
                            show_agent_orbit_tracks=True,
                            show_spacecraft_orbit=False,
                            show_coverage=False,

                            Nx=60,
                            Ny=60,
                            Nz=40,

                            # coverage display
                            show_pair_coverage=True,
                            show_triple_coverage=True,
                            pair_only_exact=True,

                            pair_coverage_alpha=0.25,
                            triple_coverage_alpha=0.35,

                            # FOV styling
                            fov_style="surface",  # "surface", "wire", "both"
                            fov_surface_alpha=0.12,
                            fov_surface_color="lightskyblue",
                            fov_n_rays=2,
                            fov_n_circle=64,
                            fov_n_len=20,

                            # styling
                            title=None,
                            label_fontsize=9,
                            label_offset_px=10,
                            slew_label_offset_px=16,
                            fill_alpha=0.10,
                            sparse_wire=True,

                            init_boresight_lw=1.5,
                            init_boresight_alpha=0.95,

                            # slew history
                            slew_history=None,
                            slew_history_line_len=ray_length,
                            slew_history_lw=1.8,
                            slew_history_alpha=0.85,
                            slew_history_cmap="viridis",
                            slew_history_every=1,
                            slew_history_colorbar=True,
                            slew_history_colorbar_label="Slew history step",
                            slew_history_norm_mode="per_agent",
                        )

                        cost_func_plots = True
                        if cost_func_plots:
                            util.plot_attcoord_costs_from_series(
                                result_kcoverage_series,
                                title="Dual coverage score vs objective (best per epoch)"
                            )

                            thetas, phis = util.plot_theta_phi_over_history(
                                res_kcoverage.extra["history"],
                                int(config["num_spacecraft"]),
                                deg=True
                            )

                        viz_cost_map = False
                        if viz_cost_map:
                            idx_fix = formation.currently_detecting
                            idx_free = 1 - idx_fix
                            u_fix = sc_pointings_eme[idx_fix]

                            TH_free_deg, PH_free_deg, J_grid = compute_J_grid_theta_phi_single_free(
                                ast_iod_eme[:3], target_cov_xyz, agents_xyz, sc_pointings_eme, theta_h_rad,
                                idx_fix=idx_fix,
                                u_fix=u_fix,
                                idx_free=idx_free,
                                theta_range_rad=(-0.5 * np.pi, 0.5 * np.pi),
                                phi_range_rad=(0.0, 2 * np.pi),
                                d_M=config['d_mahal'], kappa_sigma=config['optimizer_att_coord']['kappa_sigma'],
                                lambda_k1=config['optimizer_att_coord']['lambda_k1'],
                                n_mc=config['opt_map']['n_mc'],
                                n_grid_theta=config['opt_map']['n_grid_theta'],
                                n_grid_phi=config['opt_map']['n_grid_phi'],
                            )

                            fig3d = plt.figure(figsize=(9, 6))
                            ax3d = fig3d.add_subplot(111, projection="3d")
                            ax3d.plot_surface(
                                TH_free_deg, PH_free_deg, J_grid,
                                rstride=config['opt_map']['rstride'], cstride=config['opt_map']['cstride'],
                                linewidth=config['opt_map']['linewidth_3d'], alpha=config['opt_map']['alpha']
                            )
                            ax3d.set_xlabel(rf'$\theta_{{{idx_free}}}$ (deg)')
                            ax3d.set_ylabel(rf'$\phi_{{{idx_free}}}$ (deg)')
                            ax3d.set_zlabel(r'$J_t$')
                            ax3d.set_title(rf'$J_t(\theta,\phi)$ for free agent {idx_free} (fixed agent {idx_fix})')
                            plt.tight_layout()

                            plt.figure(figsize=(7, 5.5))
                            cs = plt.contourf(TH_free_deg, PH_free_deg, J_grid, levels=config['opt_map']['levels'])
                            plt.colorbar(cs, label=r'$J_t$')
                            plt.xlabel(rf'$\theta_{{{idx_free}}}$ (deg)')
                            plt.ylabel(rf'$\phi_{{{idx_free}}}$ (deg)')
                            plt.title(rf'$J_t(\theta,\phi)$ for free agent {idx_free} (fixed agent {idx_fix})')
                            plt.grid(alpha=0.3)

                            restart_indices = sorted({entry["restart"] for entry in res_kcoverage.extra["history"]})
                            colors = ['white', 'yellow', 'cyan', 'magenta', 'green', 'orange']
                            markers = ['o', 's', '^', 'D', 'x', '+']

                            for k, r in enumerate(restart_indices):
                                path_entries = [entry for entry in res_kcoverage.extra["history"] if entry["restart"] == r]
                                if not path_entries:
                                    continue

                                theta_path_deg = []
                                phi_path_deg = []

                                for entry in path_entries:
                                    if entry.get("use_fixed_agent", False) and ("x_free" in entry) and (
                                            entry.get("idx_fix", None) is not None):
                                        xk = np.asarray(entry["x_free"], dtype=float).ravel()
                                        th = xk[0]
                                        ph = xk[1]
                                    else:
                                        xk = np.asarray(entry["x"], dtype=float).ravel()
                                        th = xk[2 * idx_free]
                                        ph = xk[2 * idx_free + 1]

                                    theta_path_deg.append(np.rad2deg(th))
                                    phi_path_deg.append(np.rad2deg(ph))

                                theta_path_deg = np.asarray(theta_path_deg)
                                phi_path_deg = np.asarray(phi_path_deg)

                                col = colors[k % len(colors)]
                                m = markers[k % len(markers)]

                                label = f"Trial {r}"
                                if r == 0:
                                    label += " (warm start)"

                                plt.plot(
                                    theta_path_deg, phi_path_deg,
                                    linestyle='-',
                                    marker=m,
                                    color=col,
                                    lw=config['opt_map']['linewidth_2d'],
                                    ms=config['opt_map']['marker_size'],
                                    label=label
                                )

                            plt.legend(loc='upper right')

                    view_other_epoch_res_flag = False
                    if view_other_epoch_res_flag:
                        desired_idx = [15, 16, 17, 18, 19, 20, 21, 22]
                        for des_idx in desired_idx:

                            row = result_kcoverage_series[des_idx]

                            best_epoch = row["dt"]
                            best_idx = np.where(timer.attcoord_searchtimes == best_epoch)[0]

                            theta_h_rad = util.fov_deg2_to_half_angle_rad(config["fov"])
                            ems_center_xyz = np.array(config["ems"]["p_em"])
                            ems_radius = config["ems"]['R_em']
                            ray_length = config['ray_length']

                            sc_eme_ae_kms = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :3])
                            sc_pointing_eme_cartesian = row['u']
                            ang_eme = util.proj_angle_xy_from_plus_x_ccw(sc_pointing_eme_cartesian)
                            ast_truth_eme = np.squeeze(ast_eme_traj_kms[best_idx, :3])

                            ast_iod_eme = np.squeeze(x_ts[best_idx, :3])
                            P_cart_eme = np.squeeze(P_ts[best_idx, :3, :3])

                            ast_truth_eme = np.asarray(ast_truth_eme).reshape(-1)

                            best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                            T = len(timer.attcoord_searchtimes)

                            agents_xyz = sc_eme_ae_kms
                            target_cov_xyz = P_cart_eme


                            def build_slew_history_from_opt_series(opt_series, ids, *, key_u="u", normalize=True):
                                ids = [int(i) for i in ids]
                                out = {i: [] for i in ids}

                                for row_idx, rowh in enumerate(opt_series):
                                    if key_u not in rowh:
                                        raise KeyError(f"Row {row_idx} missing key '{key_u}'")
                                    U = np.asarray(rowh[key_u], dtype=float)
                                    if U.ndim != 2 or U.shape[1] != 3:
                                        raise ValueError(f"Row {row_idx} '{key_u}' must be (M,3), got {U.shape}")
                                    M = U.shape[0]
                                    for i in ids:
                                        if not (0 <= i < M):
                                            raise IndexError(f"Row {row_idx}: id {i} out of range for M={M}")
                                        out[i].append(U[i].copy())

                                for i in ids:
                                    Ui = np.asarray(out[i], dtype=float)
                                    if Ui.size == 0:
                                        Ui = Ui.reshape(0, 3)

                                    if normalize and Ui.shape[0] > 0:
                                        n = np.linalg.norm(Ui, axis=1, keepdims=True)
                                        n = np.maximum(n, 1e-12)
                                        Ui = Ui / n

                                    out[i] = Ui

                                return out

                            ids = config['three_d_prop'].get('history_ids')
                            if ids is None:
                                ids = list(range(config['num_spacecraft']))

                            slew_history = build_slew_history_from_opt_series(row["history"], ids)

                            fig, ax = util.plot_od_scenario_3d_new(
                                t_label=best_epoch,
                                agents_xyz=agents_xyz,
                                u_opt_agents_xyz=sc_pointing_eme_cartesian,
                                theta_h_rad=theta_h_rad,
                                ray_length=ray_length,
                                u_curr_agents_xyz=sc_pointings_eme,
                                boresight_line_len=ray_length * 0.1,
                                u_init_agents_xyz=None,
                                init_boresight_line_len=ray_length * 1.5,
                                agent_orbit_tracks_xyz=[sc_eme_states_kms_piecewise[:, i, :3] for i in
                                                        range(config['num_spacecraft'])],
                                spacecraft_orbit_xyz=None,
                                xlim=config['three_d_prop']['xlim'], ylim=config['three_d_prop']['ylim'],
                                zlim=config['three_d_prop']['zlim'],
                                target_mean_xyz=ast_iod_eme[:3],
                                target_cov_xyz=target_cov_xyz,
                                d_mahal=config['d_mahal'],
                                true_target_xyz=ast_truth_eme[:3],
                                target_mean_traj_xyz=x_ts[:, :3],
                                true_target_traj_xyz=ast_eme_traj_kms[:, :3],
                                true_target_traj_xyz_2=ast_truth_original_eme[:, :3],
                                ems_center_xyz=ems_center_xyz,
                                ems_radius=ems_radius,
                                show_uncertainty=True,
                                show_truth=True,  # green circle
                                show_ems=True,
                                show_fov_cones=True,
                                show_legend=True,
                                show_target_mean_traj=True,
                                show_true_target_traj=True,
                                show_true_target_traj_2=False,
                                show_init_boresights=False,
                                show_current_boresights=False,
                                show_slew_angle_annotations=False,
                                show_agent_name_annotations=True,
                                show_agent_orbit_tracks=True,
                                show_spacecraft_orbit=False,
                                show_coverage=True,

                                Nx=60,
                                Ny=60,
                                Nz=40,

                                # coverage display
                                show_pair_coverage=True,
                                show_triple_coverage=True,
                                pair_only_exact=True,

                                pair_coverage_alpha=0.25,
                                triple_coverage_alpha=0.35,

                                # FOV styling
                                fov_style="surface",  # "surface", "wire", "both"
                                fov_surface_alpha=0.12,
                                fov_surface_color="lightskyblue",
                                fov_n_rays=2,
                                fov_n_circle=64,
                                fov_n_len=20,

                                # styling
                                title=None,
                                label_fontsize=9,
                                label_offset_px=10,
                                slew_label_offset_px=16,
                                fill_alpha=0.10,
                                sparse_wire=True,

                                init_boresight_lw=1.5,
                                init_boresight_alpha=0.95,

                                # slew history
                                slew_history=slew_history,
                                slew_history_line_len=ray_length,
                                slew_history_lw=1.8,
                                slew_history_alpha=0.85,
                                slew_history_cmap="viridis",
                                slew_history_every=1,
                                slew_history_colorbar=True,
                                slew_history_colorbar_label="Slew history step",
                                slew_history_norm_mode="per_agent",
                            )


                    # plt.show()

                # ---------------------------------------------------------
                # Outer-loop diagnostics row
                # ---------------------------------------------------------
                sc_states_post = np.asarray(formation.get_spacecraft_states(), dtype=float)
                sc_pointings_post = np.asarray(formation.get_spacecraft_pointings(), dtype=float)

                x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                pos_err, vel_err = _best_effort_state_error(x_est, x_true)

                true_meas_len, true_meas_norm, true_pair = _summarize_meas_outer(p_meas_k)
                noisy_meas_len, noisy_meas_norm, noisy_pair = _summarize_meas_outer(n_meas_k)

                row_out = {
                    "run_uid": uid,
                    "master_row_idx": m_idx,
                    "rank": rank,
                    "od_step_idx": int(timer.curr_od_index),
                    "event_type": event_type,
                    "termination_reason": termination_reason,
                    "epoch_start_jdtdb": epoch_start,
                    "epoch_end_jdtdb": float(timer.curr_epoch),
                    "processed_epoch_first_jdtdb": processed_epoch_first,
                    "processed_epoch_last_jdtdb": processed_epoch_last,
                    "processed_epoch_count": processed_epoch_count,
                    "had_detection": had_detection,
                    "n_detections": n_detections,
                    "detecting_ids": detecting_ids_str,
                    "no_detection_reason": no_detection_reason,
                    "all_ems_occluded": int(bool(all_ems_occluded)),
                    "in_ems_blackout": int(bool(in_ems_blackout)),
                    "pending_reacquisition": int(bool(pending_reacquisition)),
                    "reacquisition_attempt_count": int(reacquisition_attempt_count),
                    "tracking_anchor_queue": _tracking_queue_to_str(tracking_anchor_queue),
                    "tracking_anchor_sid": (np.nan if tracking_anchor_sid is None else int(tracking_anchor_sid)),
                    "tracking_anchor_mode": tracking_anchor_mode,
                    "tracking_anchor_feasible": int(bool(tracking_anchor_feasible)),
                    "od_time_sec": od_time,
                    "attcoord_time_sec": attcoord_time,
                    "true_meas_len": true_meas_len,
                    "true_meas_norm": true_meas_norm,
                    "true_meas_0": true_pair[0],
                    "true_meas_1": true_pair[1],
                    "noisy_meas_len": noisy_meas_len,
                    "noisy_meas_norm": noisy_meas_norm,
                    "noisy_meas_0": noisy_pair[0],
                    "noisy_meas_1": noisy_pair[1],
                    "P_pos_trace": last_od_stop_metrics["trace_pos"],
                    "P_vel_trace": last_od_stop_metrics["trace_vel"],
                    "P_pos_sigma_3d": last_od_stop_metrics["sigma_pos_3d"],
                    "P_vel_sigma_3d": last_od_stop_metrics["sigma_vel_3d"],
                    "NIS_last": last_od_stop_metrics["nis_last"],
                    "NIS_mean": last_od_stop_metrics["nis_mean"],
                    "NIS_count": last_od_stop_metrics["nis_count"],
                    "OD_convergence_streak": last_od_stop_metrics["convergence_streak"],
                    "pos_err_norm": pos_err,
                    "vel_err_norm": vel_err,
                    "chosen_candidate_idx": chosen_candidate_idx,
                    "chosen_candidate_epoch_jdtdb": chosen_candidate_epoch_jdtdb,
                    "progress_status": progress_status,
                }

                for i in range(6):
                    row_out[f"x_est_{i}"] = x_est[i]
                    row_out[f"x_true_{i}"] = x_true[i]
                    row_out[f"P_diag_{i}"] = P_diag[i]

                for sc_id in range(num_sc):
                    pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                    pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                    post_s = _flatten_vec(sc_states_post[sc_id], 6)
                    post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                    for i in range(6):
                        row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                        row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                    for i in range(3):
                        row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                        row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                if timer.curr_od_index > resume_after_step:
                    _append_row(outer_csv_path, row_out, outer_header)
                    _write_progress(progress_path, {
                        "uid": uid,
                        "last_completed_outer_step": int(timer.curr_od_index),
                        "last_epoch_jdtdb": float(timer.curr_epoch),
                        "completed": False,
                        "termination_reason": "",
                        "outer_csv_path": outer_csv_path,
                        "checkpoint_path": (checkpoint_path if checkpoint_enabled else None),
                    })
                    if checkpoint_enabled:
                        _save_od_checkpoint(
                            checkpoint_path, uid=uid, m_idx=m_idx, rank=rank, ukf=ukf, timer=timer,
                            minimoon=minimoon, formation=formation,
                            od_convergence_streak=od_convergence_streak,
                            last_od_stop_metrics=last_od_stop_metrics,
                            last_completed_outer_step=int(timer.curr_od_index),
                            outer_csv_path=outer_csv_path,
                            termination_reason="",
                            progress_status=progress_status,
                            no_detection_state=_no_detection_state_dict(
                                in_ems_blackout, pending_reacquisition, reacquisition_attempt_count
                            ),
                            tracking_state=_tracking_state_dict(tracking_anchor_queue, tracking_anchor_sid),
                        )

                print_od_status(
                    timer=timer,
                    ukf=ukf,
                    minimoon=minimoon,
                    formation=formation,
                    x_true=None,
                    n_detections=n_detections,
                    status_every=status_every,
                    prefix=f"[OD r{rank} uid={uid}]",
                )

            # after loop
            final_epoch = float(getattr(timer, "curr_epoch", np.nan))
            final_steps = int(getattr(timer, "curr_od_index", 0))

            if row_paused_for_walltime:
                # Incomplete by design: leave no .done marker so the next job resumes this row.
                # Stop this rank's row loop too; near walltime, starting another row is unsafe.
                break

            if od_master_enabled:
                od_summary = _build_od_master_row_from_outer_last(
                    uid=uid, m_idx=m_idx, rank=rank, outer_csv_path=outer_csv_path, num_sc=num_sc,
                    termination_reason_fallback=termination_reason,
                    final_epoch_fallback=final_epoch,
                    final_steps_fallback=final_steps,
                )
                _append_od_master_row(od_master_rank_path, od_summary, od_master_header)

            _write_progress(progress_path, {
                "uid": uid,
                "last_completed_outer_step": final_steps,
                "last_epoch_jdtdb": final_epoch,
                "completed": True,
                "termination_reason": termination_reason,
                "outer_csv_path": outer_csv_path,
                "checkpoint_path": (checkpoint_path if checkpoint_enabled else None),
            })
            write_done(uid)
            processed += 1

        except Exception as e:
            errors += 1
            traceback.print_exc()

            try:
                row_out = {
                    "run_uid": uid,
                    "master_row_idx": m_idx,
                    "rank": rank,
                    "od_step_idx": int(getattr(timer, "curr_od_index", -1)) if "timer" in locals() else -1,
                    "event_type": "termination_error",
                    "termination_reason": f"{type(e).__name__}: {e}",
                    "epoch_start_jdtdb": float(getattr(timer, "curr_epoch", np.nan)) if "timer" in locals() else np.nan,
                    "epoch_end_jdtdb": float(getattr(timer, "curr_epoch", np.nan)) if "timer" in locals() else np.nan,
                    "processed_epoch_first_jdtdb": np.nan,
                    "processed_epoch_last_jdtdb": np.nan,
                    "processed_epoch_count": 0,
                    "had_detection": False,
                    "n_detections": 0,
                    "detecting_ids": "",
                    "od_time_sec": np.nan,
                    "attcoord_time_sec": np.nan,
                    "true_meas_len": 0,
                    "true_meas_norm": np.nan,
                    "true_meas_0": np.nan,
                    "true_meas_1": np.nan,
                    "noisy_meas_len": 0,
                    "noisy_meas_norm": np.nan,
                    "noisy_meas_0": np.nan,
                    "noisy_meas_1": np.nan,
                    "pos_err_norm": np.nan,
                    "vel_err_norm": np.nan,
                    "chosen_candidate_idx": np.nan,
                    "chosen_candidate_epoch_jdtdb": np.nan,
                    "progress_status": "error",
                }
                for i in range(6):
                    row_out[f"x_est_{i}"] = np.nan
                    row_out[f"x_true_{i}"] = np.nan
                    row_out[f"P_diag_{i}"] = np.nan
                for sc_id in range(num_sc):
                    for i in range(6):
                        row_out[f"sc{sc_id}_state_pre_{i}"] = np.nan
                        row_out[f"sc{sc_id}_state_post_{i}"] = np.nan
                    for i in range(3):
                        row_out[f"sc{sc_id}_pointing_pre_{i}"] = np.nan
                        row_out[f"sc{sc_id}_pointing_post_{i}"] = np.nan

                _append_row(outer_csv_path, row_out, outer_header)
                err_final_epoch = float(getattr(timer, "curr_epoch", np.nan)) if "timer" in locals() else np.nan
                err_final_step = int(getattr(timer, "curr_od_index", -1)) if "timer" in locals() else -1
                _write_progress(progress_path, {
                    "uid": uid,
                    "last_completed_outer_step": err_final_step,
                    "last_epoch_jdtdb": err_final_epoch,
                    "completed": True,
                    "termination_reason": f"ERROR_TERMINAL: {type(e).__name__}: {e}",
                    "outer_csv_path": outer_csv_path,
                    "checkpoint_path": (checkpoint_path if checkpoint_enabled else None),
                })
                if od_master_enabled:
                    od_summary = _build_od_master_row_from_outer_last(
                        uid=uid, m_idx=m_idx, rank=rank, outer_csv_path=outer_csv_path, num_sc=num_sc,
                        termination_reason_fallback=f"ERROR_TERMINAL: {type(e).__name__}: {e}",
                        final_epoch_fallback=err_final_epoch,
                        final_steps_fallback=err_final_step,
                    )
                    _append_od_master_row(od_master_rank_path, od_summary, od_master_header)
                write_done(uid)
            except Exception:
                try:
                    write_done(uid)
                except Exception:
                    pass

    # ---------------------------------------------------------
    # Optional rank-0 merge: per-rank OD master rows -> MASTER_IOD.csv
    # ---------------------------------------------------------
    comm.Barrier()

    if rank == 0:
        merged_rows = 0
        if od_master_enabled and od_master_merge_at_end:
            try:
                merged_rows = _merge_od_master_rows_into_master(
                    master_fn, od_master_dir, od_master_glob, od_master_header
                )
                print(f"[Stage: OD] merged {merged_rows} OD summary rows into {master_fn}", flush=True)
            except Exception as e:
                print(f"[Stage: OD] WARNING: could not merge OD master rows into MASTER_IOD.csv: {e}", flush=True)
                traceback.print_exc()

        print(f"[Stage: OD] rank0 summary: processed={processed}, skipped={skipped}, errors={errors}")


def run_OD_legacy(config_global):
    """
    Stage 4: Orbit Determination (OD)

    Added diagnostics:
      - one outer-loop CSV per run in a common folder
      - optional per-run detailed CSVs:
            * inner_kf_updates.csv
            * attcoord_candidates.csv
            * optimizer_history.csv
      - per-run progress marker for resume
      - final termination row on no-detection / time-limit / step-limit / error

    Notes:
      - Existing visualization flags/blocks are preserved.
      - Outer-loop log is the primary analysis log.
      - Inner KF / att-coord / optimizer logs are optional detailed logs.
      - Resume is append-safe and replays deterministically unless you later add full state snapshots.
    """

    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Paths & MASTER
    iod_dir = util._iod_dir(config_global)
    master_fn = os.path.join(iod_dir, "MASTER_IOD.csv")
    if rank == 0 and not os.path.exists(master_fn):
        print(f"[Stage: OD] No MASTER_IOD.csv at {master_fn} -> skip")
    comm.Barrier()
    if not os.path.exists(master_fn):
        return

    # Rank 0: inspect MASTER and broadcast row count
    if rank == 0:
        try:
            _head = pd.read_csv(master_fn, nrows=1)
            n_rows = sum(1 for _ in open(master_fn, "r", encoding="utf-8")) - 1
            print(f"[Stage: OD] MASTER rows = {n_rows}")
        except Exception as e:
            print(f"[Stage: OD] Failed to inspect MASTER: {e}")
            n_rows = 0
    else:
        n_rows = 0

    n_rows = comm.bcast(n_rows, root=0)
    if n_rows <= 0:
        return

    # Round-robin assignment
    my_indices = list(range(n_rows))[rank::size]

    # Resume markers (commit-after-write; separate dir from Stage 3)
    od_done_dir = os.path.join(iod_dir, "iod_stage4_od_done")
    if rank == 0:
        os.makedirs(od_done_dir, exist_ok=True)
    comm.Barrier()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def uid_from_saved_as(saved_as_str: str, fallback_idx: int):
        s = str(saved_as_str or "")
        if s.strip():
            first = s.split(";")[0].strip()
            if first:
                return os.path.splitext(os.path.basename(first))[0]
        return f"rowidx_{fallback_idx}"

    def done_marker_path(uid: str):
        return os.path.join(od_done_dir, f"{uid}.done")

    def is_done(uid: str) -> bool:
        return os.path.exists(done_marker_path(uid))

    def write_done(uid: str):
        with open(done_marker_path(uid), "w", encoding="utf-8") as f:
            f.write("done\n")

    def make_unique_filename(out_dir: str, base: str, ext: str = ".csv"):
        """Return a unique path like '<out_dir>/<base>.csv', or <base>__1.csv, ..."""
        name = f"{base}{ext}"
        path = os.path.join(out_dir, name)
        k = 1
        while os.path.exists(path):
            name = f"{base}__{k}{ext}"
            path = os.path.join(out_dir, name)
            k += 1
        return path, name

    def _fmt_vec(v, n=3, prec=3):
        if v is None:
            return ""
        v = np.asarray(v).reshape(-1)
        n = min(n, v.size)
        return "[" + ", ".join(f"{v[i]:.{prec}g}" for i in range(n)) + (", ..." if v.size > n else "") + "]"

    def _safe_norm(v):
        if v is None:
            return np.nan
        v = np.asarray(v).reshape(-1)
        return float(np.linalg.norm(v)) if v.size else np.nan

    def _safe_diag(P, n=6):
        if P is None:
            return [np.nan] * n
        try:
            P = np.asarray(P)
            if P.ndim != 2:
                return [np.nan] * n
            d = np.diag(P).reshape(-1)
            out = [np.nan] * n
            for i in range(min(n, d.size)):
                out[i] = float(d[i])
            return out
        except Exception:
            return [np.nan] * n

    def _flatten_vec(v, n_expected=None):
        try:
            if v is None:
                return [np.nan] * n_expected if n_expected is not None else []
            arr = np.asarray(v).reshape(-1)
            out = [float(x) for x in arr]
            if n_expected is not None:
                if len(out) < n_expected:
                    out += [np.nan] * (n_expected - len(out))
                else:
                    out = out[:n_expected]
            return out
        except Exception:
            return [np.nan] * n_expected if n_expected is not None else []

    def _ids_to_str(ids):
        if ids is None:
            return ""
        if isinstance(ids, (list, tuple, np.ndarray)):
            return ";".join(str(int(x)) for x in np.asarray(ids).reshape(-1))
        return str(ids)

    def _first_last_count(x):
        try:
            arr = np.asarray(x).reshape(-1)
            if arr.size == 0:
                return np.nan, np.nan, 0
            return float(arr[0]), float(arr[-1]), int(arr.size)
        except Exception:
            return np.nan, np.nan, 0

    def _best_effort_state_error(x_est, x_true):
        try:
            xe = np.asarray(x_est).reshape(-1)
            xt = np.asarray(x_true).reshape(-1)
            if xe.size < 6 or xt.size < 6:
                return np.nan, np.nan
            pos = float(np.linalg.norm(xe[:3] - xt[:3]))
            vel = float(np.linalg.norm(xe[3:6] - xt[3:6]))
            return pos, vel
        except Exception:
            return np.nan, np.nan

    def _safe_scalar(x):
        try:
            if x is None:
                return np.nan
            if isinstance(x, (float, int, np.floating, np.integer)):
                return float(x)
            arr = np.asarray(x).reshape(-1)
            if arr.size == 0:
                return np.nan
            return float(arr[0])
        except Exception:
            return np.nan

    def _append_row(csv_path, row_dict, header):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not exists:
                writer.writeheader()
            writer.writerow(row_dict)

    def _write_progress(progress_path, payload):
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _read_progress(progress_path):
        if not os.path.exists(progress_path):
            return None
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resume_last_outer_idx(outer_csv_path):
        if not os.path.exists(outer_csv_path):
            return None
        try:
            df = pd.read_csv(outer_csv_path)
            if len(df) == 0:
                return None
            return int(df["od_step_idx"].max())
        except Exception:
            return None

    def _atomic_pickle_dump(obj, path):
        """Atomically write a pickle file so a killed job does not leave a half-written checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def _safe_np_random_state():
        try:
            return np.random.get_state()
        except Exception:
            return None

    def _restore_np_random_state(state):
        if state is None:
            return
        try:
            np.random.set_state(state)
        except Exception:
            pass

    def _checkpoint_path_for(run_dir, checkpoint_name="checkpoint_state.pkl"):
        return os.path.join(run_dir, checkpoint_name)

    def _get_sc_epochs(formation):
        try:
            return [getattr(sc, "curr_sc_epoch", None) for sc in formation.spacecraft]
        except Exception:
            return None

    def _set_sc_epochs(formation, epochs):
        if epochs is None:
            return
        try:
            for sc, ep in zip(formation.spacecraft, epochs):
                sc.curr_sc_epoch = ep
        except Exception:
            pass

    def _save_od_checkpoint(
            checkpoint_path, *, uid, m_idx, rank, ukf, timer, minimoon, formation,
            od_convergence_streak, last_od_stop_metrics, last_completed_outer_step,
            outer_csv_path, termination_reason="", progress_status="running"
    ):
        """Save the minimum state needed to resume from the last completed outer OD step."""
        payload = {
            "schema_version": 1,
            "uid": uid,
            "master_row_idx": int(m_idx),
            "rank": int(rank),
            "outer_csv_path": outer_csv_path,
            "last_completed_outer_step": int(last_completed_outer_step),
            "termination_reason": str(termination_reason or ""),
            "progress_status": str(progress_status or "running"),
            "rng_state_numpy": _safe_np_random_state(),
            # Save object dictionaries for filter/timer because these contain small arrays,
            # counters, adaptive-R histories, and mutable timer search grids.
            "ukf_dict": copy.deepcopy(getattr(ukf, "__dict__", {})),
            "timer_dict": copy.deepcopy(getattr(timer, "__dict__", {})),
            # The asteroid epoch is intentionally not authoritative; timer.curr_epoch is.
            "minimoon_curr_state_eme": np.asarray(getattr(minimoon, "curr_state_eme", np.full(6, np.nan)), dtype=float).copy(),
            "formation_states_eme": np.asarray(formation.get_spacecraft_states(), dtype=float).copy(),
            "formation_pointings_eme": np.asarray(formation.get_spacecraft_pointings(), dtype=float).copy(),
            "formation_sc_epochs": _get_sc_epochs(formation),
            "currently_detecting": tuple(int(x) for x in np.asarray(getattr(formation, "currently_detecting", ()), dtype=int).reshape(-1)),
            "od_convergence_streak": int(od_convergence_streak),
            "last_od_stop_metrics": copy.deepcopy(last_od_stop_metrics),
        }
        _atomic_pickle_dump(payload, checkpoint_path)

    def _read_od_checkpoint(checkpoint_path):
        if not os.path.exists(checkpoint_path):
            return None
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)

    def _restore_od_checkpoint(chk, *, ukf, timer, minimoon, formation):
        """Restore checkpoint state into freshly constructed objects."""
        if not isinstance(chk, dict):
            raise ValueError("OD checkpoint is not a dictionary")
        if int(chk.get("schema_version", -1)) != 1:
            raise ValueError(f"Unsupported OD checkpoint schema_version={chk.get('schema_version')}")

        ukf.__dict__.update(copy.deepcopy(chk.get("ukf_dict", {})))
        timer.__dict__.update(copy.deepcopy(chk.get("timer_dict", {})))
        minimoon.curr_state_eme = np.asarray(chk["minimoon_curr_state_eme"], dtype=float).reshape(6,)
        formation.set_spacecraft_states(np.asarray(chk["formation_states_eme"], dtype=float).reshape(-1, 6))
        formation.set_spacecraft_pointings(np.asarray(chk["formation_pointings_eme"], dtype=float).reshape(-1, 3))
        _set_sc_epochs(formation, chk.get("formation_sc_epochs", None))
        formation.currently_detecting = tuple(int(x) for x in np.asarray(chk.get("currently_detecting", ()), dtype=int).reshape(-1))
        _restore_np_random_state(chk.get("rng_state_numpy", None))
        return int(chk.get("od_convergence_streak", 0)), copy.deepcopy(chk.get("last_od_stop_metrics", {}))

    def _write_incomplete_progress(progress_path, *, uid, timer, outer_csv_path, termination_reason, checkpoint_path=None):
        _write_progress(progress_path, {
            "uid": uid,
            "last_completed_outer_step": int(getattr(timer, "curr_od_index", -1)),
            "last_epoch_jdtdb": float(getattr(timer, "curr_epoch", np.nan)),
            "completed": False,
            "termination_reason": str(termination_reason),
            "outer_csv_path": outer_csv_path,
            "checkpoint_path": checkpoint_path,
        })

    def _walltime_near_limit(job_start_time, wall_cfg):
        if not bool(wall_cfg.get("enabled", False)):
            return False
        max_walltime_sec = wall_cfg.get("max_walltime_sec", None)
        if max_walltime_sec is None:
            return False
        safety_buffer_sec = float(wall_cfg.get("safety_buffer_sec", 900.0))
        return (time.time() - job_start_time) >= (float(max_walltime_sec) - safety_buffer_sec)

    def _read_last_outer_row(outer_csv_path):
        """Return the last row of an outer-loop CSV as a dict, or an empty dict."""
        if not os.path.exists(outer_csv_path):
            return {}
        try:
            df = pd.read_csv(outer_csv_path)
            if len(df) == 0:
                return {}
            return df.iloc[-1].to_dict()
        except Exception:
            return {}

    def _csv_scalar(v):
        """Make scalar values safe for CSV without turning useful strings into NaN."""
        try:
            if v is None:
                return ""
            if isinstance(v, str):
                return v
            if pd.isna(v):
                return np.nan
            return v
        except Exception:
            return v

    def _serialize_outer_vec(last_outer_row, prefix, n):
        """Serialize fields like x_est_0...x_est_5 from an outer-loop row."""
        vals = []
        for i in range(n):
            v = last_outer_row.get(f"{prefix}_{i}", np.nan)
            try:
                vals.append("" if pd.isna(v) else f"{float(v):.16g}")
            except Exception:
                vals.append(str(v))
        return ";".join(vals)

    def _build_od_master_row_from_outer_last(
            *, uid, m_idx, rank, outer_csv_path, num_sc, termination_reason_fallback="",
            final_epoch_fallback=np.nan, final_steps_fallback=np.nan
    ):
        """Build one compact OD summary row from the final row of a run's outer-loop CSV."""
        last = _read_last_outer_row(outer_csv_path)

        # Prefer the final outer-loop row. Fall back only if the outer log cannot be read.
        final_epoch = last.get("epoch_end_jdtdb", final_epoch_fallback)
        final_steps = last.get("od_step_idx", final_steps_fallback)
        term_reason = last.get("termination_reason", termination_reason_fallback)

        row = {
            "master_row_idx": int(m_idx),
            "run_uid": uid,
            "rank": int(rank),
            "OD_RUN_UID": uid,
            "OD_RANK": int(rank),
            "OD_RESULT_SAVED_AS": os.path.basename(outer_csv_path),
            "OD_OUTER_CSV_PATH": outer_csv_path,
            "OD_EVENT_TYPE": _csv_scalar(last.get("event_type", "")),
            "OD_TERMINATION_REASON": _csv_scalar(term_reason),
            "OD_PROGRESS_STATUS": _csv_scalar(last.get("progress_status", "")),
            "OD_FINAL_TIME_JDTDB": _csv_scalar(final_epoch),
            "OD_N_STEPS": _csv_scalar(final_steps),
            "OD_HAD_DETECTION": _csv_scalar(last.get("had_detection", np.nan)),
            "OD_N_DETECTIONS": _csv_scalar(last.get("n_detections", np.nan)),
            "OD_DETECTING_IDS": _csv_scalar(last.get("detecting_ids", "")),
            "OD_PROCESSED_EPOCH_FIRST_JDTDB": _csv_scalar(last.get("processed_epoch_first_jdtdb", np.nan)),
            "OD_PROCESSED_EPOCH_LAST_JDTDB": _csv_scalar(last.get("processed_epoch_last_jdtdb", np.nan)),
            "OD_PROCESSED_EPOCH_COUNT": _csv_scalar(last.get("processed_epoch_count", np.nan)),
            "OD_TIME_SEC": _csv_scalar(last.get("od_time_sec", np.nan)),
            "OD_ATTCOORD_TIME_SEC": _csv_scalar(last.get("attcoord_time_sec", np.nan)),
            "OD_LAST_POS_RMSE": _csv_scalar(last.get("pos_err_norm", np.nan)),
            "OD_LAST_VEL_RMSE": _csv_scalar(last.get("vel_err_norm", np.nan)),
            "OD_P_POS_TRACE": _csv_scalar(last.get("P_pos_trace", np.nan)),
            "OD_P_VEL_TRACE": _csv_scalar(last.get("P_vel_trace", np.nan)),
            "OD_P_POS_SIGMA_3D": _csv_scalar(last.get("P_pos_sigma_3d", np.nan)),
            "OD_P_VEL_SIGMA_3D": _csv_scalar(last.get("P_vel_sigma_3d", np.nan)),
            "OD_NIS_LAST": _csv_scalar(last.get("NIS_last", np.nan)),
            "OD_NIS_MEAN": _csv_scalar(last.get("NIS_mean", np.nan)),
            "OD_NIS_COUNT": _csv_scalar(last.get("NIS_count", np.nan)),
            "OD_CONVERGENCE_STREAK": _csv_scalar(last.get("OD_convergence_streak", np.nan)),
            "OD_CHOSEN_CANDIDATE_IDX": _csv_scalar(last.get("chosen_candidate_idx", np.nan)),
            "OD_CHOSEN_CANDIDATE_EPOCH_JDTDB": _csv_scalar(last.get("chosen_candidate_epoch_jdtdb", np.nan)),
            "OD_FINAL_STATE": _serialize_outer_vec(last, "x_est", 6),
            "OD_FINAL_TRUE_STATE": _serialize_outer_vec(last, "x_true", 6),
            "OD_FINAL_P_DIAG": _serialize_outer_vec(last, "P_diag", 6),
            "OD_MASTER_WRITE_UTC": dt.datetime.utcnow().isoformat() + "Z",
        }

        for sc_id in range(num_sc):
            row[f"OD_FINAL_SC{sc_id}_STATE"] = _serialize_outer_vec(last, f"sc{sc_id}_state_post", 6)
            row[f"OD_FINAL_SC{sc_id}_POINTING"] = _serialize_outer_vec(last, f"sc{sc_id}_pointing_post", 3)

        return row

    def _append_od_master_row(od_master_rank_path, row_dict, od_master_header, dedup_existing=True):
        """Append one completed OD summary to this rank's OD master CSV.

        This is rank-local, so it needs no MPI/file lock. If a retry tries to append
        the same master_row_idx/run_uid again, dedup_existing prevents duplicates.
        """
        os.makedirs(os.path.dirname(od_master_rank_path), exist_ok=True)
        if dedup_existing and os.path.exists(od_master_rank_path):
            try:
                prev = pd.read_csv(od_master_rank_path, usecols=["master_row_idx", "run_uid"])
                dup = (prev["master_row_idx"].astype(int) == int(row_dict["master_row_idx"])) & (prev["run_uid"].astype(str) == str(row_dict["run_uid"]))
                if bool(dup.any()):
                    return
            except Exception:
                pass
        _append_row(od_master_rank_path, row_dict, od_master_header)

    def _merge_od_master_rows_into_master(master_fn, od_master_dir, od_master_glob, od_master_header):
        """Rank-0 convenience merge: per-rank OD master rows -> MASTER_IOD.csv.

        The per-rank OD master files remain the durable OD-stage record even if this
        final merge is interrupted by walltime.
        """
        paths = sorted(glob.glob(os.path.join(od_master_dir, od_master_glob)))
        if len(paths) == 0:
            return 0

        frames = []
        for path in paths:
            try:
                df = pd.read_csv(path)
                if len(df) > 0:
                    frames.append(df)
            except Exception as e:
                print(f"[Stage: OD] Could not read OD master file {path}: {e}", flush=True)

        if len(frames) == 0:
            return 0

        od_df = pd.concat(frames, ignore_index=True)
        if "master_row_idx" not in od_df.columns:
            return 0

        # Keep the latest appended summary for each MASTER row.
        if "OD_MASTER_WRITE_UTC" in od_df.columns:
            od_df = od_df.sort_values(["master_row_idx", "OD_MASTER_WRITE_UTC"])
        od_df = od_df.drop_duplicates(subset=["master_row_idx"], keep="last")

        dfm = pd.read_csv(master_fn)
        # Keep rank-local bookkeeping columns in OD_MASTER.csv, but do not add
        # unprefixed columns like run_uid/rank to the main MASTER_IOD.csv.
        merge_cols = [c for c in od_df.columns if c not in ("master_row_idx", "run_uid", "rank")]
        for col in merge_cols:
            if col not in dfm.columns:
                dfm[col] = ""

        for _, upd in od_df.iterrows():
            try:
                ri = int(upd["master_row_idx"])
            except Exception:
                continue
            if ri < 0 or ri >= len(dfm):
                continue
            for col in merge_cols:
                dfm.at[ri, col] = upd[col]

        tmp = master_fn + ".tmp"
        dfm.to_csv(tmp, index=False)
        os.replace(tmp, master_fn)
        return int(len(od_df))

    def _summarize_meas_outer(meas):
        try:
            arr = np.asarray(meas)

            if arr.size == 0:
                return 0, np.nan, [np.nan, np.nan]

            # reshape to (num_pairs, 2)
            pairs = arr.reshape(-1, 2)

            # keep only valid RA/Dec pairs
            valid_mask = np.isfinite(pairs).all(axis=1)
            valid_pairs = pairs[valid_mask]

            n_valid = valid_pairs.shape[0]

            if n_valid == 0:
                return 0, np.nan, [np.nan, np.nan]

            norm = float(np.linalg.norm(valid_pairs))

            first_pair = valid_pairs[0]

            return n_valid, norm, [float(first_pair[0]), float(first_pair[1])]

        except Exception:
            return 0, np.nan, [np.nan, np.nan]

    def _extract_inner_obs_state(sc_states_k, update_idx, detecting_ids=None):
        """
        sc_states_k shape: (M, N, 6)
        We map update_idx to time index update_idx, and by default pick first detecting SC if possible.
        """
        try:
            arr = np.asarray(sc_states_k)
            if arr.ndim != 3:
                return [np.nan] * 6

            M, N, D = arr.shape
            if D < 6 or update_idx >= N:
                return [np.nan] * 6

            sc_pick = 0
            if detecting_ids is not None:
                try:
                    det = np.asarray(detecting_ids).reshape(-1)
                    if det.size > 0:
                        sc_pick = int(det[0])
                except Exception:
                    pass

            sc_pick = max(0, min(M - 1, sc_pick))
            return _flatten_vec(arr[sc_pick, update_idx, :], 6)
        except Exception:
            return [np.nan] * 6

    def _extract_inner_meas_pair(perfect_meas, noisy_meas, update_idx, detecting_ids=None):
        """
        perfect_meas, noisy_meas shapes: (M, N, 2)
        For a given inner update index, extract one representative pair:
        first detecting SC if available, otherwise SC 0.
        """
        try:
            p = np.asarray(perfect_meas)
            n = np.asarray(noisy_meas)
            if p.ndim != 3 or n.ndim != 3:
                return [np.nan, np.nan], [np.nan, np.nan]
            M, N, D = p.shape
            if D < 2 or update_idx >= N:
                return [np.nan, np.nan], [np.nan, np.nan]

            sc_pick = 0
            if detecting_ids is not None:
                try:
                    det = np.asarray(detecting_ids).reshape(-1)
                    if det.size > 0:
                        sc_pick = int(det[0])
                except Exception:
                    pass

            sc_pick = max(0, min(M - 1, sc_pick))
            return _flatten_vec(p[sc_pick, update_idx, :], 2), _flatten_vec(n[sc_pick, update_idx, :], 2)
        except Exception:
            return [np.nan, np.nan], [np.nan, np.nan]

    def print_od_status(
            *,
            timer,
            ukf,
            minimoon=None,
            formation=None,
            x_true=None,
            n_detections=None,
            status_every=1,
            prefix="[OD]",
            print_time=True,
    ):
        k = getattr(timer, "curr_od_index", None)
        if k is None:
            return

        if status_every is not None and status_every > 1:
            if (k % status_every) != 0:
                return

        x_est = getattr(ukf, "x", None)
        P_est = getattr(ukf, "P", None)

        if x_true is None and minimoon is not None:
            x_true = getattr(minimoon, "curr_state_eme", None)

        pos_err_norm, vel_err_norm = _best_effort_state_error(x_est, x_true)
        Pdiag = _safe_diag(P_est, n=6)

        det_id = ""
        if formation is not None and hasattr(formation, "currently_detecting"):
            det_id = str(getattr(formation, "currently_detecting"))

        epoch = getattr(timer, "curr_epoch", None)
        endt = getattr(timer, "end_time", None)

        ts = ""
        if print_time:
            ts = datetime.now().strftime("%H:%M:%S") + " "

        line1 = (
            f"{ts}{prefix} k={k} "
            f"epoch_jdtdb={epoch if epoch is not None else ''} "
            f"end_jdtdb={endt if endt is not None else ''} "
            f"det_sc={det_id} "
            f"n_det={'' if n_detections is None else n_detections}"
        )
        line2 = f"{prefix} x_est={_fmt_vec(x_est, n=6, prec=6)}"
        line3 = (
            f"{prefix} pos_err_norm={pos_err_norm if not np.isnan(pos_err_norm) else ''} "
            f"vel_err_norm={vel_err_norm if not np.isnan(vel_err_norm) else ''} "
            f"P_diag={_fmt_vec(Pdiag, n=6, prec=6)}"
        )

        print(line1)
        print(line2)
        print(line3)

    def _cov_trace_metrics(P):
        """Return trace/sigma metrics for the 3x3 position and velocity covariance blocks."""
        try:
            P = np.asarray(P, dtype=float).reshape(6, 6)
            tr_pos = float(np.trace(P[:3, :3]))
            tr_vel = float(np.trace(P[3:, 3:]))
            sig_pos = float(np.sqrt(max(tr_pos, 0.0)))
            sig_vel = float(np.sqrt(max(tr_vel, 0.0)))
            return tr_pos, tr_vel, sig_pos, sig_vel
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

    def _extract_nis_values(update_history, window=None):
        """Extract NIS = innovation.T @ inv(S) @ innovation from nested UKF update history."""
        vals = []
        try:
            for entry in update_history or []:
                for upd in entry.get("updates", []):
                    info = upd.get("update_info", {})
                    innov = info.get("innovation", None)
                    S = info.get("S", None)
                    if innov is None or S is None:
                        continue
                    innov = np.asarray(innov, dtype=float).reshape(-1)
                    S = np.asarray(S, dtype=float)
                    if innov.size == 0 or S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] != innov.size:
                        continue
                    nis = float(innov @ np.linalg.pinv(S) @ innov)
                    if np.isfinite(nis):
                        vals.append(nis)
        except Exception:
            pass

        if window is not None:
            try:
                window = int(window)
                if window > 0:
                    vals = vals[-window:]
            except Exception:
                pass
        return vals

    def _od_stop_metrics_and_decision(config, P, update_history, convergence_streak):
        """
        Check OD convergence using only covariance traces and NIS.

        Expected config block, for example:
            od_stop:
              enabled: true
              pos_trace_threshold_km2: 1.0e4          # OR pos_sigma_threshold_km
              vel_trace_threshold_km2_s2: 2.5e-5      # OR vel_sigma_threshold_km_s
              nis_mean_max: 10.0
              nis_mean_min: 0.0                       # optional
              nis_window: 20
              nis_min_samples: 3
              required_consecutive_steps: 1
        """
        stop_cfg = config.get("od_stop", {})
        enabled = bool(stop_cfg.get("enabled", False))

        tr_pos, tr_vel, sig_pos, sig_vel = _cov_trace_metrics(P)
        nis_window = stop_cfg.get("nis_window", None)
        nis_vals = _extract_nis_values(update_history, window=nis_window)
        nis_count = int(len(nis_vals))
        nis_last = float(nis_vals[-1]) if nis_count > 0 else np.nan
        nis_mean = float(np.mean(nis_vals)) if nis_count > 0 else np.nan

        metrics = {
            "trace_pos": tr_pos,
            "trace_vel": tr_vel,
            "sigma_pos_3d": sig_pos,
            "sigma_vel_3d": sig_vel,
            "nis_last": nis_last,
            "nis_mean": nis_mean,
            "nis_count": nis_count,
            "convergence_streak": int(convergence_streak),
        }

        if not enabled:
            return False, 0, metrics, "od_stop_disabled"

        # Accept either trace thresholds or sigma thresholds. Sigma thresholds are converted to trace thresholds.
        pos_trace_thr = stop_cfg.get("pos_trace_threshold_km2", None)
        if pos_trace_thr is None and stop_cfg.get("pos_sigma_threshold_km", None) is not None:
            pos_trace_thr = float(stop_cfg["pos_sigma_threshold_km"]) ** 2

        vel_trace_thr = stop_cfg.get("vel_trace_threshold_km2_s2", None)
        if vel_trace_thr is None and stop_cfg.get("vel_sigma_threshold_km_s", None) is not None:
            vel_trace_thr = float(stop_cfg["vel_sigma_threshold_km_s"]) ** 2

        nis_min_samples = int(stop_cfg.get("nis_min_samples", 1))
        nis_mean_max = stop_cfg.get("nis_mean_max", None)
        nis_mean_min = stop_cfg.get("nis_mean_min", None)
        required_consecutive = int(stop_cfg.get("required_consecutive_steps", 1))

        checks = []
        reasons = []

        if pos_trace_thr is not None:
            ok = np.isfinite(tr_pos) and (tr_pos <= float(pos_trace_thr))
            checks.append(ok)
            if not ok:
                reasons.append(f"pos_trace {tr_pos:.6g} > {float(pos_trace_thr):.6g}")

        if vel_trace_thr is not None:
            ok = np.isfinite(tr_vel) and (tr_vel <= float(vel_trace_thr))
            checks.append(ok)
            if not ok:
                reasons.append(f"vel_trace {tr_vel:.6g} > {float(vel_trace_thr):.6g}")

        if nis_mean_max is not None or nis_mean_min is not None:
            ok_samples = nis_count >= nis_min_samples and np.isfinite(nis_mean)
            if nis_mean_max is not None:
                ok = ok_samples and (nis_mean <= float(nis_mean_max))
                checks.append(ok)
                if not ok:
                    reasons.append(f"nis_mean {nis_mean:.6g} > {float(nis_mean_max):.6g} or insufficient NIS samples")
            if nis_mean_min is not None:
                ok = ok_samples and (nis_mean >= float(nis_mean_min))
                checks.append(ok)
                if not ok:
                    reasons.append(f"nis_mean {nis_mean:.6g} < {float(nis_mean_min):.6g} or insufficient NIS samples")

        # If no actual threshold was provided, never stop.
        if len(checks) == 0:
            return False, 0, metrics, "od_stop_enabled_but_no_thresholds"

        instant_ok = all(checks)
        new_streak = convergence_streak + 1 if instant_ok else 0
        metrics["convergence_streak"] = int(new_streak)

        should_stop = new_streak >= required_consecutive
        reason = "convergence_criteria_met" if should_stop else ("; ".join(reasons) if reasons else "waiting_for_consecutive_convergence")
        return should_stop, new_streak, metrics, reason

    # ---------------------------------------------------------
    # Diagnostics config
    # ---------------------------------------------------------
    diag_cfg = config_global.get("od_diagnostics", {})
    log_inner_kf = bool(diag_cfg.get("log_inner_kf", False))
    log_attcoord = bool(diag_cfg.get("log_attcoord_candidates", False))
    log_optimizer = bool(diag_cfg.get("log_optimizer_history", False))

    checkpoint_cfg = config_global.get("od_checkpoint", {})
    checkpoint_enabled_default = True
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", checkpoint_enabled_default))
    checkpoint_name = str(checkpoint_cfg.get("checkpoint_name", "checkpoint_state.pkl"))
    walltime_cfg = checkpoint_cfg.get("walltime_guard", {})
    # Backward-compatible flat config style is also accepted:
    # od_checkpoint:
    #   walltime_guard_enabled: true
    #   max_walltime_sec: 82800
    #   safety_buffer_sec: 900
    if "walltime_guard_enabled" in checkpoint_cfg or "max_walltime_sec" in checkpoint_cfg:
        walltime_cfg = {
            "enabled": bool(checkpoint_cfg.get("walltime_guard_enabled", checkpoint_cfg.get("enabled", False))),
            "max_walltime_sec": checkpoint_cfg.get("max_walltime_sec", None),
            "safety_buffer_sec": checkpoint_cfg.get("safety_buffer_sec", 900.0),
        }

    outer_dir = diag_cfg.get("outer_loop_dir", os.path.join(iod_dir, "od_outer_logs"))
    detail_root = diag_cfg.get("detail_root_dir", os.path.join(iod_dir, "od_run_details"))
    os.makedirs(outer_dir, exist_ok=True)
    os.makedirs(detail_root, exist_ok=True)

    # Load MASTER (workers)
    df_master = pd.read_csv(master_fn)

    # OD config
    od_duration_days = float(config_global.get('od_duration_days', 1.0))
    od_max_steps = config_global.get('od_max_steps', None)
    od_max_steps = int(od_max_steps) if (od_max_steps is not None) else None

    # Per-rank OD master files: durable, low-memory OD summary output for MPI/HPC.
    od_master_cfg = config_global.get("od_master", {})
    od_master_enabled = bool(od_master_cfg.get("enabled", True))
    od_master_dir = od_master_cfg.get("dir", os.path.join(iod_dir, "od_master_rows"))
    od_master_filename_template = str(od_master_cfg.get("rank_filename_template", "od_master_rank_{rank}.csv"))
    od_master_rank_path = os.path.join(od_master_dir, od_master_filename_template.format(rank=rank))
    od_master_merge_at_end = bool(od_master_cfg.get("merge_into_master_at_end", True))
    od_master_glob = str(od_master_cfg.get("merge_glob", "od_master_rank_*.csv"))

    if od_master_enabled:
        os.makedirs(od_master_dir, exist_ok=True)

    processed = skipped = errors = 0

    # ---------------------------------------------------------
    # Diagnostics headers
    # ---------------------------------------------------------
    num_sc = int(config_global["num_spacecraft"])

    outer_header = [
        "run_uid",
        "master_row_idx",
        "rank",
        "od_step_idx",
        "event_type",
        "termination_reason",
        "epoch_start_jdtdb",
        "epoch_end_jdtdb",
        "processed_epoch_first_jdtdb",
        "processed_epoch_last_jdtdb",
        "processed_epoch_count",
        "had_detection",
        "n_detections",
        "detecting_ids",
        "od_time_sec",
        "attcoord_time_sec",
        "true_meas_len",
        "true_meas_norm",
        "true_meas_0",
        "true_meas_1",
        "noisy_meas_len",
        "noisy_meas_norm",
        "noisy_meas_0",
        "noisy_meas_1",
    ]
    outer_header += [f"x_est_{i}" for i in range(6)]
    outer_header += [f"x_true_{i}" for i in range(6)]
    outer_header += [f"P_diag_{i}" for i in range(6)]
    outer_header += [
        "P_pos_trace",
        "P_vel_trace",
        "P_pos_sigma_3d",
        "P_vel_sigma_3d",
        "NIS_last",
        "NIS_mean",
        "NIS_count",
        "OD_convergence_streak",
    ]
    outer_header += ["pos_err_norm", "vel_err_norm"]

    for sc_id in range(num_sc):
        outer_header += [f"sc{sc_id}_state_pre_{i}" for i in range(6)]
    for sc_id in range(num_sc):
        outer_header += [f"sc{sc_id}_pointing_pre_{i}" for i in range(3)]
    for sc_id in range(num_sc):
        outer_header += [f"sc{sc_id}_state_post_{i}" for i in range(6)]
    for sc_id in range(num_sc):
        outer_header += [f"sc{sc_id}_pointing_post_{i}" for i in range(3)]

    outer_header += [
        "chosen_candidate_idx",
        "chosen_candidate_epoch_jdtdb",
        "progress_status",
    ]

    inner_header = [
        "run_uid",
        "master_row_idx",
        "rank",
        "od_step_idx",
        "inner_update_idx",
        "epoch_jdtdb",
    ]
    inner_header += [f"x_prior_{i}" for i in range(6)]
    inner_header += [f"x_post_{i}" for i in range(6)]
    inner_header += [f"P_prior_diag_{i}" for i in range(6)]
    inner_header += [f"P_post_diag_{i}" for i in range(6)]
    inner_header += [f"x_true_{i}" for i in range(6)]
    inner_header += [f"observer_state_{i}" for i in range(6)]
    inner_header += [f"true_meas_{i}" for i in range(2)]
    inner_header += [f"noisy_meas_{i}" for i in range(2)]
    inner_header += ["detecting_ids"]

    attcoord_header = [
        "run_uid",
        "master_row_idx",
        "rank",
        "od_step_idx",
        "candidate_idx",
        "candidate_epoch_jdtdb",
        "is_chosen",
    ]
    attcoord_header += [f"x_pred_{i}" for i in range(6)]
    attcoord_header += [f"P_pred_diag_{i}" for i in range(6)]
    for sc_id in range(num_sc):
        attcoord_header += [f"sc{sc_id}_state_{i}" for i in range(6)]
    for sc_id in range(num_sc):
        attcoord_header += [f"u_cmd_sc{sc_id}_{i}" for i in range(3)]
    attcoord_header += ["J", "coverage_score"]

    optimizer_header = [
        "run_uid",
        "master_row_idx",
        "rank",
        "od_step_idx",
        "optimizer_row_idx",
        "restart_idx",
        "use_fixed_agent",
        "idx_fix",
        "J",
    ]
    optimizer_header += [f"x_param_{i}" for i in range(12)]
    optimizer_header += [f"x_free_param_{i}" for i in range(12)]
    for sc_id in range(num_sc):
        optimizer_header += [f"u_sc{sc_id}_{i}" for i in range(3)]
    optimizer_header += [f"slew_sc{i}" for i in range(num_sc)]

    od_master_header = [
        "master_row_idx",
        "run_uid",
        "rank",
        "OD_RUN_UID",
        "OD_RANK",
        "OD_RESULT_SAVED_AS",
        "OD_OUTER_CSV_PATH",
        "OD_EVENT_TYPE",
        "OD_TERMINATION_REASON",
        "OD_PROGRESS_STATUS",
        "OD_FINAL_TIME_JDTDB",
        "OD_N_STEPS",
        "OD_HAD_DETECTION",
        "OD_N_DETECTIONS",
        "OD_DETECTING_IDS",
        "OD_PROCESSED_EPOCH_FIRST_JDTDB",
        "OD_PROCESSED_EPOCH_LAST_JDTDB",
        "OD_PROCESSED_EPOCH_COUNT",
        "OD_TIME_SEC",
        "OD_ATTCOORD_TIME_SEC",
        "OD_LAST_POS_RMSE",
        "OD_LAST_VEL_RMSE",
        "OD_P_POS_TRACE",
        "OD_P_VEL_TRACE",
        "OD_P_POS_SIGMA_3D",
        "OD_P_VEL_SIGMA_3D",
        "OD_NIS_LAST",
        "OD_NIS_MEAN",
        "OD_NIS_COUNT",
        "OD_CONVERGENCE_STREAK",
        "OD_CHOSEN_CANDIDATE_IDX",
        "OD_CHOSEN_CANDIDATE_EPOCH_JDTDB",
        "OD_FINAL_STATE",
        "OD_FINAL_TRUE_STATE",
        "OD_FINAL_P_DIAG",
        "OD_MASTER_WRITE_UTC",
    ]
    for sc_id in range(num_sc):
        od_master_header += [f"OD_FINAL_SC{sc_id}_STATE", f"OD_FINAL_SC{sc_id}_POINTING"]

    # ---------------------------------------------------------
    # Main row loop
    # ---------------------------------------------------------
    job_start_time = time.time()

    for m_idx in my_indices:
        row = df_master.iloc[m_idx]

        config = copy.deepcopy(config_global)

        occluded = int(row["OCCLUDED_BY_EMS"]) == 1

        if occluded:
            config["ems"]["R_em"] = 0.0
            config["ems"]["alpha_s_deg"] = 0.0
            print("Doing OD without EMS Exclusion...")
        else:
            print("Doing OD with EMS Exclusion")

        saved_as_str = str(row.get("IOD_DATA_SAVED_AS", "") or "")
        uid = uid_from_saved_as(saved_as_str, m_idx)

        if is_done(uid):
            skipped += 1
            continue

        base = f"{uid}__OD_{config['dynamics']}_{config['orbit']}_{config['observer']}_{config['optimizer']}"
        outer_csv_path = os.path.join(outer_dir, f"{base}__outer.csv")
        run_dir = os.path.join(detail_root, uid)
        progress_path = os.path.join(run_dir, "progress.json")
        inner_csv_path = os.path.join(run_dir, "inner_kf_updates.csv")
        attcoord_csv_path = os.path.join(run_dir, "attcoord_candidates.csv")
        optimizer_csv_path = os.path.join(run_dir, "optimizer_history.csv")
        checkpoint_path = _checkpoint_path_for(run_dir, checkpoint_name)
        row_paused_for_walltime = False

        try:
            # ----------------------------------------------
            # Setup from iod
            # ---------------------------------------------
            setup = od_setup_from_iod(config, row, util=util, sp=sp)

            # --------------------------
            # Initialization (first step)
            # --------------------------
            x0_est = setup["iod"]["ast_iod_eme_ae_kms"]
            P0_est = setup["iod"]["cov"]["P_cart_eme"]
            x_true0 = setup["frames"]["ast_eme_ae_kms"]
            t0_jdtdb = setup["epochs"]["ae_jdtdb"]

            # Process noise initialization
            r_noise = config['radial_process_noise']
            t_noise = config['track_process_noise']
            n_noise = config['normal_process_noise']

            # Measurement noise initialization
            ra_noise = config['sigma_ra']
            dec_noise = config['sigma_dec']
            pointing_noise = config['sigma_pointing']
            meas_noise = config['sigma_meas_noise']

            # ukf weight values
            alpha = config['alpha_ukf']
            beta = config['beta_ukf']
            kappa = config['kappa_ukf']
            epsilon = config['epsilon_ukf']
            adaptive_R_config = config['adaptive_R']

            # ---------------------------------------------
            # build objects
            # -------------------------------------------
            n_body_propagator = nbody.NBodyPropagator(spice=sp, config=config)

            ukf = OD_UKF(
                x0=setup["x0_eme_kms"],
                P0=setup["P0_eme"],
                Sa_rtn=(r_noise, t_noise, n_noise),
                meas_units="mas",
                sigma_ra=ra_noise,
                sigma_dec=dec_noise,
                sigma_pointing=pointing_noise,
                sigma_meas=None,  # do not use unless you want to artificially inflate R
                adaptive_R_window=adaptive_R_config['window'],
                adaptive_R_min_samples=adaptive_R_config['min_samples'],
                adaptive_R_psd_floor=adaptive_R_config['psd_floor'],
                sigma_rho_single_obs_km=adaptive_R_config['single_obs_rho'],
                sigma_rhodot_single_obs_km_s=adaptive_R_config['single_obs_rhodot'],
                p_injection_decay_tau=adaptive_R_config.get('p_injection_decay_tau', 1.0),
                multi_observer_maturity_threshold=adaptive_R_config.get('multi_observer_maturity_threshold', 3),
                mature_single_observer_injection=adaptive_R_config.get('mature_single_observer_injection', 'decayed'),
                ukf_alpha=alpha,
                ukf_beta=beta,
                ukf_kappa=kappa,
                eps=epsilon,
            )

            attitude_coordination = AttitudeCoordinator(config)

            minimoon = Asteroid(
                row['ID_AST'],
                row['INDEX_USED'],
                config,
                current_state_eme=setup['frames']['ast_eme_ae_kms'],
                current_epoch=setup['epochs']['ae_jdtdb']
            )

            formation = Formation(config)
            sc1_ini_index = formation.get_index_from_pos(util.parse_vec_cell(row["SPACECRAFT_1_INI_POS(km)"]))
            formation.recall_formation(sc1_ini_index, config)
            formation.match_spacecraft_trajectory_full(int(row['TOTAL_LENGTH']), config)
            formation.set_spacecraft_states(
                setup["frames"]["sc_eme_ae_kms"],
                set_epochs_from_row=row
            )

            formation.set_spacecraft_pointings(setup["frames"]["sc_pointing_eme_cartesian"])
            formation.currently_detecting = tuple([int(setup["sc_detecting_id"])])

            timer = SimTime(
                config,
                current_od_index=0,
                current_epoch=t0_jdtdb,
                iod_time=row['COMPUTATION_TIME_SEC'],
                current_integration_epoch=t0_jdtdb,
                current_integration_index=row['INDEX_USED']
            )

            status_every = int(config.get("od_status_every", 1))

            last_pos_rmse = np.nan
            last_vel_rmse = np.nan
            termination_reason = ""
            progress_status = "running"

            prog = _read_progress(progress_path)
            last_completed_outer = _resume_last_outer_idx(outer_csv_path)
            if prog is not None and prog.get("completed", False):
                write_done(uid)
                skipped += 1
                continue

            resume_after_step = -1 if last_completed_outer is None else int(last_completed_outer)

            od_convergence_streak = 0
            last_od_stop_metrics = {
                "trace_pos": np.nan,
                "trace_vel": np.nan,
                "sigma_pos_3d": np.nan,
                "sigma_vel_3d": np.nan,
                "nis_last": np.nan,
                "nis_mean": np.nan,
                "nis_count": 0,
                "convergence_streak": 0,
            }

            # True checkpoint resume: restore the filter/timer/formation state directly
            # instead of replaying from the beginning. The old resume_after_step guard is
            # kept only as a duplicate-write safety net.
            if checkpoint_enabled and os.path.exists(checkpoint_path):
                chk = _read_od_checkpoint(checkpoint_path)
                od_convergence_streak, restored_metrics = _restore_od_checkpoint(
                    chk, ukf=ukf, timer=timer, minimoon=minimoon, formation=formation
                )
                if restored_metrics:
                    last_od_stop_metrics.update(restored_metrics)
                resume_after_step = int(chk.get("last_completed_outer_step", resume_after_step))
                print(
                    f"[OD r{rank} uid={uid}] Resumed from checkpoint at "
                    f"od_step_idx={timer.curr_od_index}, epoch_jdtdb={timer.curr_epoch}",
                    flush=True,
                )

            while True:
                if _walltime_near_limit(job_start_time, walltime_cfg):
                    termination_reason = "walltime_checkpoint"
                    progress_status = "paused"
                    if checkpoint_enabled:
                        _save_od_checkpoint(
                            checkpoint_path, uid=uid, m_idx=m_idx, rank=rank, ukf=ukf, timer=timer,
                            minimoon=minimoon, formation=formation,
                            od_convergence_streak=od_convergence_streak,
                            last_od_stop_metrics=last_od_stop_metrics,
                            last_completed_outer_step=int(getattr(timer, "curr_od_index", -1)),
                            outer_csv_path=outer_csv_path,
                            termination_reason=termination_reason,
                            progress_status=progress_status,
                        )
                    _write_incomplete_progress(
                        progress_path, uid=uid, timer=timer, outer_csv_path=outer_csv_path,
                        termination_reason=termination_reason, checkpoint_path=(checkpoint_path if checkpoint_enabled else None)
                    )
                    row_paused_for_walltime = True
                    print(f"[OD r{rank} uid={uid}] Pausing before walltime limit; checkpoint saved.", flush=True)
                    break

                if timer.curr_epoch > timer.end_time or timer.check_search_times():
                    print("Over Time")
                    termination_reason = "time_limit"
                    progress_status = "completed"
                    event_type = "termination_time_limit"

                    sc_states_pre = np.asarray(formation.get_spacecraft_states(), dtype=float)
                    sc_pointings_pre = np.asarray(formation.get_spacecraft_pointings(), dtype=float)
                    sc_states_post = sc_states_pre.copy()
                    sc_pointings_post = sc_pointings_pre.copy()

                    x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                    x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                    P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                    pos_err, vel_err = _best_effort_state_error(x_est, x_true)

                    row_out = {
                        "run_uid": uid,
                        "master_row_idx": m_idx,
                        "rank": rank,
                        "od_step_idx": int(timer.curr_od_index),
                        "event_type": event_type,
                        "termination_reason": termination_reason,
                        "epoch_start_jdtdb": float(timer.curr_epoch),
                        "epoch_end_jdtdb": float(timer.curr_epoch),
                        "processed_epoch_first_jdtdb": np.nan,
                        "processed_epoch_last_jdtdb": np.nan,
                        "processed_epoch_count": 0,
                        "had_detection": False,
                        "n_detections": 0,
                        "detecting_ids": "",
                        "od_time_sec": np.nan,
                        "attcoord_time_sec": np.nan,
                        "true_meas_len": 0,
                        "true_meas_norm": np.nan,
                        "true_meas_0": np.nan,
                        "true_meas_1": np.nan,
                        "noisy_meas_len": 0,
                        "noisy_meas_norm": np.nan,
                        "noisy_meas_0": np.nan,
                        "noisy_meas_1": np.nan,
                        "pos_err_norm": pos_err,
                        "vel_err_norm": vel_err,
                        "chosen_candidate_idx": np.nan,
                        "chosen_candidate_epoch_jdtdb": np.nan,
                        "progress_status": progress_status,
                    }
                    for i in range(6):
                        row_out[f"x_est_{i}"] = x_est[i]
                        row_out[f"x_true_{i}"] = x_true[i]
                        row_out[f"P_diag_{i}"] = P_diag[i]
                    for sc_id in range(num_sc):
                        pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                        pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                        post_s = _flatten_vec(sc_states_post[sc_id], 6)
                        post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                        for i in range(6):
                            row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                            row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                        for i in range(3):
                            row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                            row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                    if timer.curr_od_index > resume_after_step:
                        _append_row(outer_csv_path, row_out, outer_header)
                    break

                if od_max_steps is not None and timer.curr_od_index >= od_max_steps:
                    print("Out of Iterations")
                    termination_reason = "step_limit"
                    progress_status = "completed"
                    event_type = "termination_step_limit"

                    sc_states_pre = np.asarray(formation.get_spacecraft_states(), dtype=float)
                    sc_pointings_pre = np.asarray(formation.get_spacecraft_pointings(), dtype=float)
                    sc_states_post = sc_states_pre.copy()
                    sc_pointings_post = sc_pointings_pre.copy()

                    x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                    x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                    P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                    pos_err, vel_err = _best_effort_state_error(x_est, x_true)

                    row_out = {
                        "run_uid": uid,
                        "master_row_idx": m_idx,
                        "rank": rank,
                        "od_step_idx": int(timer.curr_od_index),
                        "event_type": event_type,
                        "termination_reason": termination_reason,
                        "epoch_start_jdtdb": float(timer.curr_epoch),
                        "epoch_end_jdtdb": float(timer.curr_epoch),
                        "processed_epoch_first_jdtdb": np.nan,
                        "processed_epoch_last_jdtdb": np.nan,
                        "processed_epoch_count": 0,
                        "had_detection": False,
                        "n_detections": 0,
                        "detecting_ids": "",
                        "od_time_sec": np.nan,
                        "attcoord_time_sec": np.nan,
                        "true_meas_len": 0,
                        "true_meas_norm": np.nan,
                        "true_meas_0": np.nan,
                        "true_meas_1": np.nan,
                        "noisy_meas_len": 0,
                        "noisy_meas_norm": np.nan,
                        "noisy_meas_0": np.nan,
                        "noisy_meas_1": np.nan,
                        "pos_err_norm": pos_err,
                        "vel_err_norm": vel_err,
                        "chosen_candidate_idx": np.nan,
                        "chosen_candidate_epoch_jdtdb": np.nan,
                        "progress_status": progress_status,
                    }
                    for i in range(6):
                        row_out[f"x_est_{i}"] = x_est[i]
                        row_out[f"x_true_{i}"] = x_true[i]
                        row_out[f"P_diag_{i}"] = P_diag[i]
                    for sc_id in range(num_sc):
                        pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                        pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                        post_s = _flatten_vec(sc_states_post[sc_id], 6)
                        post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                        for i in range(6):
                            row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                            row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                        for i in range(3):
                            row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                            row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                    if timer.curr_od_index > resume_after_step:
                        _append_row(outer_csv_path, row_out, outer_header)
                    break

                epoch_start = float(timer.curr_epoch)
                sc_states_pre = np.asarray(formation.get_spacecraft_states(), dtype=float)
                sc_pointings_pre = np.asarray(formation.get_spacecraft_pointings(), dtype=float)

                processed_epoch_first = np.nan
                processed_epoch_last = np.nan
                processed_epoch_count = 0
                had_detection = True
                n_detections = np.nan
                detecting_ids_str = ""
                od_time = np.nan
                attcoord_time = np.nan
                chosen_candidate_idx = np.nan
                chosen_candidate_epoch_jdtdb = np.nan
                p_meas_k = None
                n_meas_k = None

                # update according to iod
                if timer.curr_od_index == 0:
                    event_type = "initial_attcoord"

                    # -------------- REGULAR IOD STEP -------------------
                    attcoord_startime = time.time()

                    timer.set_attcoord_searchtimes()

                    x_ts, P_ts = ukf.propagate_priors(
                        timer.curr_epoch,
                        timer.attcoord_searchtimes_jdtdb,
                        n_body_propagator.propagate_multiple_objects
                    )

                    sc_eme_states_kms_piecewise, anchor_info = util.piecewise_anchor_and_propagate_spacecraft_trajs(
                        formation=formation, minimoon=minimoon, timer=timer,
                        t_targets_jdtdb=timer.attcoord_searchtimes_jdtdb,
                        n_body_propagator=n_body_propagator, return_anchor_info=True
                    )

                    ast_eme_state_kms = minimoon.curr_state_eme
                    ast_eme_traj_kms = n_body_propagator.propagate(ast_eme_state_kms, timer.curr_epoch,
                                                                   timer.attcoord_searchtimes_jdtdb)

                    ast_truth_original_eclip = np.array(minimoon.orbit.loc[:, ["Geo x", "Geo y", "Geo z", "Geo vx",
                                                                               "Geo vy", "Geo vz"]])
                    ast_truth_original_eclip[:, :3] *= config['AU_TO_M'] / 1000
                    ast_truth_original_eclip[:, 3:] *= config['AU_TO_M'] / 1000 / config['SECONDS_PER_DAY']
                    ast_truth_original_eme = util.geo_eclip_to_geo_eme_generic(ast_truth_original_eclip,
                                                                               hint=("time", "state"))

                    # to visualize the possible att coord scenarios
                    viz_prop_flag_ini = False
                    if viz_prop_flag_ini:
                        util.plot_priors_positions_and_cov_2d(
                            x_ts,
                            P_ts,
                            sc_trajs_km2=sc_eme_states_kms_piecewise,
                            stride=config['two_d_prop']['stride'],
                            n_std=config['two_d_prop']['stride'],
                            planes=("xy", "xz", "yz"),
                            title_prefix="Asteroid prior + spacecraft"
                        )

                        util.plot_matched_trajectory_full_range(
                            formation=formation,
                            idx_start=0,
                            idx_stop=2400,
                            frame="GEO_EME",
                            plot_3d=True,
                            plot_xy=False
                        )


                    sc_pointings_eme = formation.get_spacecraft_pointings()

                    sc0 = formation.spacecraft[0]
                    theta_h_rad = util.fov_deg2_to_half_angle_rad(sc0.fov)
                    tau_max = sc0.reaction_wheel_torque
                    h_max = sc0.reaction_wheel_momentum
                    m_m = sc0.mass
                    l_m = sc0.length
                    m_t = sc0.telescope_mass
                    d_t = sc0.telescope_diameter
                    l_t = sc0.telescope_length
                    z_0 = sc0.telescope_offset
                    I_max = (1 / 6) * m_m * (l_m) ** 2 + (1 /12) * m_t * (3*(d_t / 2) ** 2 + l_t ** 2) + m_t * z_0 ** 2
                    alpha_max = 1.63 * tau_max / I_max
                    omega_max = 1.63 * h_max / I_max

                    res_kcoverage, result_mean, result_kcoverage_series, result_mean_series, mean_time = attitude_coordination.step(
                        sc_eme_states_kms_piecewise[:, :, :3],
                        sc_pointings_eme,
                        x_ts[:, :3],
                        P_ts[:, :3, :3],
                        timer.attcoord_searchtimes,
                        theta_h_rad,
                        alpha_max,
                        omega_max,
                        sc_pointings_eme[ list(formation.currently_detecting), :],
                        list(formation.currently_detecting),
                        d_M=config['d_mahal'],
                        use_fixed_agent=True,
                        fixed_agent_idx=int(formation.currently_detecting[0]),
                        fixed_agent_u=sc_pointings_eme[int(formation.currently_detecting[0]), :],
                        coverage_point=ast_eme_traj_kms[:, :3]
                    )

                    attcoord_endtime = time.time()
                    timer.set_attcoord_time(attcoord_endtime - attcoord_startime - mean_time)
                    attcoord_time = attcoord_endtime - attcoord_startime - mean_time
                    od_time = 0.0

                    timer.set_slew_time(res_kcoverage.chosen_dt)

                    best_epoch = res_kcoverage.chosen_dt
                    best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                    best_epoch_jdtdb = float(timer.attcoord_searchtimes_jdtdb[best_idx])
                    k_anchor_best = int(anchor_info["anchor_k_of_t"][best_idx])
                    t_anchor_best = float(anchor_info["anchor_epoch_of_t"][best_idx])

                    best_attitudes = res_kcoverage.u_cmd
                    formation.set_spacecraft_pointings(best_attitudes)

                    best_positions = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :])
                    formation.set_spacecraft_states(best_positions)

                    ukf.x = np.squeeze(x_ts[best_idx, :])
                    ukf.P = np.squeeze(P_ts[best_idx, :, :])

                    minimoon.set_state(np.squeeze(ast_eme_traj_kms[best_idx, :]))

                    timer.step(best_epoch_jdtdb, k_anchor_best, t_anchor_best)

                    chosen_candidate_idx = best_idx
                    chosen_candidate_epoch_jdtdb = best_epoch_jdtdb
                    had_detection = True
                    n_detections = 1
                    detecting_ids_str = str(int(setup["sc_detecting_id"]))

                    # visualize att_coord result
                    att_coord_viz_flag_ini = True
                    if att_coord_viz_flag_ini:
                        best_epoch = res_kcoverage.chosen_dt
                        best_idx = np.where(timer.attcoord_searchtimes == best_epoch)[0]

                        theta_h_rad = util.fov_deg2_to_half_angle_rad(config["fov"])
                        ems_center_xy = np.array(config["ems"]["p_em"][:2])
                        ems_center_xyz = np.array(config["ems"]["p_em"])
                        ems_radius = config["ems"]['R_em']
                        ray_length = config['ray_length']

                        sc_eme_ae_kms = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :3])
                        sc_pointing_eme_cartesian = res_kcoverage.u_cmd
                        ang_eme = util.proj_angle_xy_from_plus_x_ccw(sc_pointing_eme_cartesian)
                        ast_truth_eme = np.squeeze(ast_eme_traj_kms[best_idx, :3])

                        ast_iod_eme = np.squeeze(x_ts[best_idx, :3])
                        P_cart_eme = np.squeeze(P_ts[best_idx, :3, :3])

                        ast_truth_eme = np.asarray(ast_truth_eme).reshape(-1)

                        best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                        T = len(timer.attcoord_searchtimes)

                        two_d_pot = False
                        if two_d_pot:
                            def is_valid(idx):
                                J = result_kcoverage_series[idx]["J"]
                                return not (J is None or (isinstance(J, float) and math.isnan(J)))

                            cands = [best_idx, max(0, best_idx - 1), min(T - 1, best_idx + 1)]
                            selected = []
                            for c in cands:
                                if c not in selected and is_valid(c):
                                    selected.append(c)
                            if len(selected) < 3:
                                for c in range(T):
                                    if len(selected) >= 3:
                                        break
                                    if c not in selected and is_valid(c):
                                        selected.append(c)
                            if len(selected) < 3:
                                for c in range(T):
                                    if len(selected) >= 3:
                                        break
                                    if c not in selected:
                                        selected.append(c)

                            planes = [((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ")]

                            def cov2_from_cov3(P3, axes):
                                i, j = axes
                                return P3[np.ix_([i, j], [i, j])]

                            def angles_in_plane(u_cmd_xyz, axes):
                                a, b = axes
                                u = np.asarray(u_cmd_xyz, dtype=float)
                                u2 = u[:, [a, b]]
                                return np.arctan2(u2[:, 1], u2[:, 0])

                            for idx in selected:
                                epoch = timer.attcoord_searchtimes[idx]
                                sc_xyz = np.asarray(sc_eme_states_kms_piecewise[idx, :, :3], dtype=float)
                                ast_truth = np.asarray(ast_eme_traj_kms[idx, :3], dtype=float).reshape(3,)
                                ast_mean = np.asarray(x_ts[idx, :3], dtype=float).reshape(3,)
                                P3 = np.asarray(P_ts[idx, :3, :3], dtype=float).reshape(3, 3)

                                u_cmd_xyz = np.asarray(result_kcoverage_series[idx]["u"], dtype=float)
                                u_curr_xyz = np.asarray(sc_pointings_eme, dtype=float)

                                fig, axes = plt.subplots(1, 3, figsize=(8, 14))
                                fig.suptitle(f"2D Projections @ epoch {epoch}", y=0.99)

                                for ax, (axpair, name) in zip(axes, planes):
                                    agents_2d = sc_xyz[:, list(axpair)]
                                    u_curr_2d = u_curr_xyz[:, list(axpair)]
                                    mean_2d = ast_mean[list(axpair)]
                                    truth_2d = ast_truth[list(axpair)]
                                    cov_2d = cov2_from_cov3(P3, axpair)
                                    mean_traj_2d = np.asarray(x_ts[:, :3], dtype=float)[:, list(axpair)]
                                    truth_traj_2d = np.asarray(ast_eme_traj_kms[:, :3], dtype=float)[:, list(axpair)]
                                    ems_center_2d = np.asarray(ems_center_xyz, dtype=float)[list(axpair)]
                                    ang_2d = angles_in_plane(u_cmd_xyz, axpair)

                                    util.plot_od_scenario_2d(
                                        t_label=f"{epoch} ({name})",
                                        agents_xy=agents_2d,
                                        pointing_angles_rad=ang_2d,
                                        theta_h_rad=theta_h_rad,
                                        ray_length=ray_length * 2,
                                        u_curr_agents_xy=u_curr_2d,
                                        boresight_line_len=ray_length * 0.1,
                                        target_mean_xy=mean_2d,
                                        target_mean_xy_traj=mean_traj_2d,
                                        target_cov_xy=cov_2d,
                                        d_mahal=config['d_mahal'],
                                        true_target_xy=truth_2d,
                                        true_target_xy_traj=truth_traj_2d,
                                        ems_center_xy=ems_center_2d,
                                        ems_radius=ems_radius,
                                        xlim=config['two_d_prop']['xlim'], ylim=config['two_d_prop']['ylim'],
                                        agent_orbit_tracks_xy=None,
                                        ax=ax,
                                        title=None
                                    )

                                plt.tight_layout()

                        agents_xyz = sc_eme_ae_kms
                        target_cov_xyz = P_cart_eme

                        def range_sigma_from_cov(P_eci, r_obj, r_obs):
                            """
                            P_eci : 6x6 or 3x3 covariance in inertial frame
                            r_obj : object inertial position, shape (3,)
                            r_obs : observer inertial position, shape (3,)

                            Returns range standard deviation and variance.
                            """
                            P_r = P_eci[:3, :3] if P_eci.shape == (6, 6) else P_eci

                            rho_vec = np.asarray(r_obj) - np.asarray(r_obs)
                            rho_hat = rho_vec / np.linalg.norm(rho_vec)

                            var_rho = rho_hat @ P_r @ rho_hat
                            sigma_rho = np.sqrt(max(var_rho, 0.0))

                            return sigma_rho, var_rho

                        sigma_rho, var_rho = range_sigma_from_cov(P_cart_eme, ast_iod_eme[:3],
                                                                  agents_xyz[int(formation.currently_detecting[0])])
                        print(sigma_rho, var_rho)

                        def build_slew_history_from_opt_series(opt_series, ids, *, key_u="u", normalize=True):
                            ids = [int(i) for i in ids]
                            out = {i: [] for i in ids}
                            for row_idx, rowh in enumerate(opt_series):
                                if key_u not in rowh:
                                    raise KeyError(f"Row {row_idx} missing key '{key_u}'")
                                U = np.asarray(rowh[key_u], dtype=float)
                                if U.ndim != 2 or U.shape[1] != 3:
                                    raise ValueError(f"Row {row_idx} '{key_u}' must be (M,3), got {U.shape}")
                                M = U.shape[0]
                                for i in ids:
                                    if not (0 <= i < M):
                                        raise IndexError(f"Row {row_idx}: id {i} out of range for M={M}")
                                    out[i].append(U[i].copy())
                            for i in ids:
                                Ui = np.asarray(out[i], dtype=float)
                                if Ui.size == 0:
                                    Ui = Ui.reshape(0, 3)
                                if normalize and Ui.shape[0] > 0:
                                    n = np.linalg.norm(Ui, axis=1, keepdims=True)
                                    n = np.maximum(n, 1e-12)
                                    Ui = Ui / n
                                out[i] = Ui
                            return out

                        ids = config['three_d_prop'].get('history_ids')
                        if ids is None:
                            ids = list(range(config['num_spacecraft']))
                        slew_history = build_slew_history_from_opt_series(res_kcoverage.extra["history"], ids)

                        fig, ax = util.plot_od_scenario_3d_old(
                            t_label=best_epoch,
                            agents_xyz=agents_xyz,
                            u_opt_agents_xyz=sc_pointing_eme_cartesian,
                            theta_h_rad=theta_h_rad,
                            ray_length=ray_length,
                            u_curr_agents_xyz=sc_pointings_eme,
                            boresight_line_len=ray_length * 0.1,
                            u_init_agents_xyz=None,
                            init_boresight_line_len=ray_length * 1.5,
                            xlim=config['three_d_prop']['xlim'], ylim=config['three_d_prop']['ylim'],
                            zlim=config['three_d_prop']['zlim'],
                            Nx=config['three_d_prop']['Nx'], Ny=config['three_d_prop']['Ny'],
                            Nz=config['three_d_prop']['Nz'],
                            max_points_for_scatter=config['three_d_prop']['max_points'],
                            target_mean_xyz=ast_iod_eme[:3],
                            target_cov_xyz=target_cov_xyz,
                            d_mahal=config['d_mahal'],
                            true_target_xyz=ast_truth_eme[:3],
                            target_mean_traj_xyz=x_ts[:, :3],
                            true_target_traj_xyz=ast_eme_traj_kms[:, :3],
                            true_target_traj_xyz_2=ast_truth_original_eme[:, :3],
                            ems_center_xyz=ems_center_xyz,
                            ems_radius=ems_radius,
                            show_coverage=True,
                            show_uncertainty=True,
                            show_truth=True,
                            show_ems=True,
                            show_fov_cones=True,
                            title="3D OD Scenario Demo (EME)",
                            slew_history=slew_history,
                            slew_history_line_len=ray_length,
                            agent_orbit_tracks_xyz=[sc_eme_states_kms_piecewise[:, i, :3] for i in range(config['num_spacecraft'])]
                        )

                        plot_cost_diagnosis = True
                        if plot_cost_diagnosis:
                            util.plot_attcoord_costs_from_series(
                                result_kcoverage_series,
                                title="Dual coverage score vs objective (best per epoch)"
                            )

                            thetas, phis = util.plot_theta_phi_over_history(
                                res_kcoverage.extra["history"],
                                int(config["num_spacecraft"]),
                                deg=True
                            )

                        viz_cost_map = False
                        if viz_cost_map:
                            idx_fix = int(formation.currently_detecting[0])
                            idx_free = 1 - idx_fix
                            u_fix = sc_pointings_eme[idx_fix]

                            TH_free_deg, PH_free_deg, J_grid = compute_J_grid_theta_phi_single_free(
                                ast_iod_eme[:3], target_cov_xyz, agents_xyz, sc_pointings_eme, theta_h_rad,
                                idx_fix=idx_fix,
                                u_fix=u_fix,
                                idx_free=idx_free,
                                theta_range_rad=(-0.5 * np.pi, 0.5 * np.pi),
                                phi_range_rad=(0.0, 2 * np.pi),
                                d_M=config['d_mahal'],
                                kappa_sigma=config['optimizer_att_coord']['kappa_sigma'],
                                lambda_k1=config['optimizer_att_coord']['lambda_k1'],
                                n_mc=config['opt_map']['n_mc'],
                                n_grid_theta=config['opt_map']['n_grid_theta'],
                                n_grid_phi=config['opt_map']['n_grid_phi'],
                            )

                            fig3d = plt.figure(figsize=(9, 6))
                            ax3d = fig3d.add_subplot(111, projection="3d")
                            ax3d.plot_surface(
                                TH_free_deg, PH_free_deg, J_grid,
                                rstride=config['opt_map']['rstride'], cstride=config['opt_map']['cstride'],
                                linewidth=config['opt_map']['linewidth_3d'], alpha=config['opt_map']['alpha']
                            )
                            ax3d.set_xlabel(rf'$\theta_{{{idx_free}}}$ (deg)')
                            ax3d.set_ylabel(rf'$\phi_{{{idx_free}}}$ (deg)')
                            ax3d.set_zlabel(r'$J_t$')
                            ax3d.set_title(rf'$J_t(\theta,\phi)$ for free agent {idx_free} (fixed agent {idx_fix})')
                            plt.tight_layout()

                            plt.figure(figsize=(7, 5.5))
                            cs = plt.contourf(TH_free_deg, PH_free_deg, J_grid, levels=config['opt_map']['levels'])
                            plt.colorbar(cs, label=r'$J_t$')
                            plt.xlabel(rf'$\theta_{{{idx_free}}}$ (deg)')
                            plt.ylabel(rf'$\phi_{{{idx_free}}}$ (deg)')
                            plt.title(rf'$J_t(\theta,\phi)$ for free agent {idx_free} (fixed agent {idx_fix})')
                            plt.grid(alpha=0.3)

                            restart_indices = sorted({entry["restart"] for entry in res_kcoverage.extra["history"]})
                            colors = ['white', 'yellow', 'cyan', 'magenta', 'green', 'orange']
                            markers = ['o', 's', '^', 'D', 'x', '+']

                            for k, r in enumerate(restart_indices):
                                path_entries = [entry for entry in res_kcoverage.extra["history"] if entry["restart"] == r]
                                if not path_entries:
                                    continue

                                theta_path_deg = []
                                phi_path_deg = []

                                for entry in path_entries:
                                    if entry.get("use_fixed_agent", False) and ("x_free" in entry) and (
                                            entry.get("idx_fix", None) is not None):
                                        xk = np.asarray(entry["x_free"], dtype=float).ravel()
                                        th = xk[0]
                                        ph = xk[1]
                                    else:
                                        xk = np.asarray(entry["x"], dtype=float).ravel()
                                        th = xk[2 * idx_free]
                                        ph = xk[2 * idx_free + 1]

                                    theta_path_deg.append(np.rad2deg(th))
                                    phi_path_deg.append(np.rad2deg(ph))

                                theta_path_deg = np.asarray(theta_path_deg)
                                phi_path_deg = np.asarray(phi_path_deg)

                                col = colors[k % len(colors)]
                                m = markers[k % len(markers)]

                                label = f"Trial {r}"
                                if r == 0:
                                    label += " (warm start)"

                                plt.plot(
                                    theta_path_deg, phi_path_deg,
                                    linestyle='-',
                                    marker=m,
                                    color=col,
                                    lw=config['opt_map']['linewidth_2d'],
                                    ms=config['opt_map']['marker_size'],
                                    label=label
                                )

                            plt.legend(loc='upper right')

                    view_other_epoch_res_flag = False
                    if view_other_epoch_res_flag:
                        desired_idx = [5, 6, 7, 8, 9]
                        for des_idx in desired_idx:

                            row = result_kcoverage_series[des_idx]

                            best_epoch = row["dt"]
                            best_idx = np.where(timer.attcoord_searchtimes == best_epoch)[0]

                            theta_h_rad = util.fov_deg2_to_half_angle_rad(config["fov"])
                            ems_center_xyz = np.array(config["ems"]["p_em"])
                            ems_radius = config["ems"]['R_em']
                            ray_length = config['ray_length']

                            sc_eme_ae_kms = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :3])
                            sc_pointing_eme_cartesian = row['u']
                            ang_eme = util.proj_angle_xy_from_plus_x_ccw(sc_pointing_eme_cartesian)
                            ast_truth_eme = np.squeeze(ast_eme_traj_kms[best_idx, :3])

                            ast_iod_eme = np.squeeze(x_ts[best_idx, :3])
                            P_cart_eme = np.squeeze(P_ts[best_idx, :3, :3])

                            ast_truth_eme = np.asarray(ast_truth_eme).reshape(-1)

                            best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                            T = len(timer.attcoord_searchtimes)

                            agents_xyz = sc_eme_ae_kms
                            target_cov_xyz = P_cart_eme

                            def build_slew_history_from_opt_series(opt_series, ids, *, key_u="u", normalize=True):
                                ids = [int(i) for i in ids]
                                out = {i: [] for i in ids}

                                for row_idx, rowh in enumerate(opt_series):
                                    if key_u not in rowh:
                                        raise KeyError(f"Row {row_idx} missing key '{key_u}'")
                                    U = np.asarray(rowh[key_u], dtype=float)
                                    if U.ndim != 2 or U.shape[1] != 3:
                                        raise ValueError(f"Row {row_idx} '{key_u}' must be (M,3), got {U.shape}")
                                    M = U.shape[0]
                                    for i in ids:
                                        if not (0 <= i < M):
                                            raise IndexError(f"Row {row_idx}: id {i} out of range for M={M}")
                                        out[i].append(U[i].copy())

                                for i in ids:
                                    Ui = np.asarray(out[i], dtype=float)
                                    if Ui.size == 0:
                                        Ui = Ui.reshape(0, 3)

                                    if normalize and Ui.shape[0] > 0:
                                        n = np.linalg.norm(Ui, axis=1, keepdims=True)
                                        n = np.maximum(n, 1e-12)
                                        Ui = Ui / n

                                    out[i] = Ui

                                return out

                            ids = config['three_d_prop'].get('history_ids')
                            if ids is None:
                                ids = list(range(config['num_spacecraft']))

                            slew_history = build_slew_history_from_opt_series(row["history"], ids)

                            fig, ax = util.plot_od_scenario_3d_new(
                                t_label=best_epoch,
                                agents_xyz=agents_xyz,
                                u_opt_agents_xyz=sc_pointing_eme_cartesian,
                                theta_h_rad=theta_h_rad,
                                ray_length=ray_length,
                                u_curr_agents_xyz=sc_pointings_eme,
                                boresight_line_len=ray_length * 0.1,
                                u_init_agents_xyz=None,
                                init_boresight_line_len=ray_length * 1.5,
                                agent_orbit_tracks_xyz=[sc_eme_states_kms_piecewise[:, i, :3] for i in
                                                        range(config['num_spacecraft'])],
                                spacecraft_orbit_xyz=None,
                                xlim=config['three_d_prop']['xlim'], ylim=config['three_d_prop']['ylim'],
                                zlim=config['three_d_prop']['zlim'],
                                target_mean_xyz=ast_iod_eme[:3],
                                target_cov_xyz=target_cov_xyz,
                                d_mahal=config['d_mahal'],
                                true_target_xyz=ast_truth_eme[:3],
                                target_mean_traj_xyz=x_ts[:, :3],
                                true_target_traj_xyz=ast_eme_traj_kms[:, :3],
                                true_target_traj_xyz_2=ast_truth_original_eme[:, :3],
                                ems_center_xyz=ems_center_xyz,
                                ems_radius=ems_radius,
                                show_uncertainty=True,
                                show_truth=True,  # green circle
                                show_ems=True,
                                show_fov_cones=True,
                                show_legend=True,
                                show_target_mean_traj=True,
                                show_true_target_traj=True,
                                show_true_target_traj_2=False,
                                show_init_boresights=False,
                                show_current_boresights=False,
                                show_slew_angle_annotations=False,
                                show_agent_name_annotations=True,
                                show_agent_orbit_tracks=True,
                                show_spacecraft_orbit=False,
                                show_coverage=True,

                                Nx=60,
                                Ny=60,
                                Nz=40,

                                # coverage display
                                show_pair_coverage=True,
                                show_triple_coverage=True,
                                pair_only_exact=True,

                                pair_coverage_alpha=0.25,
                                triple_coverage_alpha=0.35,

                                # FOV styling
                                fov_style="surface",  # "surface", "wire", "both"
                                fov_surface_alpha=0.12,
                                fov_surface_color="lightskyblue",
                                fov_n_rays=2,
                                fov_n_circle=64,
                                fov_n_len=20,

                                # styling
                                title=None,
                                label_fontsize=9,
                                label_offset_px=10,
                                slew_label_offset_px=16,
                                fill_alpha=0.10,
                                sparse_wire=True,

                                init_boresight_lw=1.5,
                                init_boresight_alpha=0.95,

                                # slew history
                                slew_history=slew_history,
                                slew_history_line_len=ray_length,
                                slew_history_lw=1.8,
                                slew_history_alpha=0.85,
                                slew_history_cmap="viridis",
                                slew_history_every=1,
                                slew_history_colorbar=True,
                                slew_history_colorbar_label="Slew history step",
                                slew_history_norm_mode="per_agent",
                            )



                #-------------------------
                # Regular OD Step
                # -------------------------
                else:
                    event_type = "regular_update"

                    #######################################################
                    # Gather Tracklet data and Detect
                    ######################################################
                    p_meas_k, n_meas_k, sc_states_k, ast_states_k, epochs_k, detection_res_k = formation.detect(
                        minimoon.curr_state_eme,
                        timer.curr_epoch,
                        n_body_propagator,
                        config
                    )

                    # optional visualization block unchanged
                    confirm_meas = False
                    if confirm_meas:
                        ems_center_xyz = np.array(config["ems"]["p_em"])
                        ems_radius = config["ems"]['R_em']
                        util.plot_detection_geometry_3d(
                            perfect_meas=p_meas_k,
                            noisy_meas=n_meas_k,
                            sc_states=sc_states_k,
                            ast_states=ast_states_k,
                            detection_results=detection_res_k,
                            epochs=epochs_k,
                            los_stride=2,
                            use_true_range_for_los=True,
                            # EMS sphere
                            ems_center_xyz=ems_center_xyz,
                            ems_radius=ems_radius,
                            show_ems=True,
                            ems_alpha=0.10,
                            show_ems_wires=True,
                            title="Detection Geometry",
                            save_path=None,
                            show=False,
                        )
                        two_d_too = False
                        if two_d_too:
                            plane = ("x", "y")
                            plane_str = "".join(plane)
                            util.plot_detection_geometry_2d(
                                perfect_meas=p_meas_k,
                                noisy_meas=n_meas_k,
                                sc_states=sc_states_k,
                                ast_states=ast_states_k,
                                detection_results=detection_res_k,
                                plane=plane,
                                los_stride=2,
                                save_path=None,
                                show=False,
                            )

                    # print(detection_res_k)
                    # plt.show()

                    # -------------------------
                    # Prediction + UKF update only
                    # -------------------------
                    od_time_start = time.time()
                    od_meas_result = process_tracklet_until_update_with_prior_epoch(
                        ukf=ukf,
                        prior_epoch_jdtdb=timer.curr_epoch,
                        n_body_propagator=n_body_propagator,
                        epochs_k=epochs_k,
                        noisy_meas_k=n_meas_k,
                        sc_states_k=sc_states_k,
                        detection_res_k=detection_res_k,
                    )
                    od_time = time.time() - od_time_start

                    had_detection = od_meas_result["had_detection"]
                    detecting_ids = od_meas_result["detecting_ids"]
                    n_detections = od_meas_result["n_detections"]
                    processed_epochs = od_meas_result["processed_epochs"]

                    x_post_k = od_meas_result["posterior_x"]
                    P_post_k = od_meas_result["posterior_P"]
                    update_history_k = od_meas_result["update_history"]

                    formation.currently_detecting = detecting_ids

                    ast_eme_state_kms_during_detection = minimoon.curr_state_eme

                    update_history = od_meas_result["update_history"]
                    processed_epochs = od_meas_result["processed_epochs"]

                    x_est_hist = np.array([entry["x_post"] for entry in update_history])  # (K,6)
                    P_est_hist = np.array([entry["P_post"] for entry in update_history])  # (K,6,6)
                    x_pred_hist = np.array([entry["x_prior"] for entry in update_history])
                    P_pred_hist = np.array([entry["P_prior"] for entry in update_history])

                    if len(processed_epochs) > 0:
                        processed_epoch_first, processed_epoch_last, processed_epoch_count = _first_last_count(processed_epochs)

                    ast_eme_traj_kms_during_detection = n_body_propagator.propagate(
                        ast_eme_state_kms_during_detection,
                        timer.curr_epoch,
                        processed_epochs
                    ) if len(processed_epochs) > 0 else np.empty((0, 6))

                    propped_initial_state = n_body_propagator.propagate(
                        x_est_hist[0], timer.curr_epoch, processed_epochs
                    ) if len(update_history) > 0 else np.empty((0, 6))

                    view_update = False
                    if view_update and len(update_history) > 0:
                        fig, ax = util.plot_od_trajectory_with_measurements_3d(
                            x_est_hist=x_est_hist,
                            ast_true_hist=ast_eme_traj_kms_during_detection,
                            P_est_hist=P_est_hist,
                            noisy_meas=n_meas_k,
                            sc_states=sc_states_k,
                            detection_results=detection_res_k,
                            x_pred_hist=propped_initial_state,
                            d_mahal=3.0,
                            los_stride=2,
                            los_scale_before=0.015,
                            los_scale_after=0.025,
                            title="OD estimate vs truth with LOS and 3σ ellipsoid",
                            save_path=None,
                            show=False,
                        )
                        show_two_d = True
                        if show_two_d:
                            fig, ax = util.plot_od_trajectory_with_measurements_2d(
                                x_est_hist=x_est_hist,
                                ast_true_hist=ast_eme_traj_kms_during_detection,
                                P_est_hist=P_est_hist,
                                noisy_meas=n_meas_k,
                                sc_states=sc_states_k,
                                detection_results=detection_res_k,
                                plane=("x", "y"),
                                x_pred_hist=propped_initial_state,
                                d_mahal=3.0,
                                los_stride=2,
                                los_scale_before=0.015,
                                los_scale_after=0.025,
                                title="OD estimate vs truth with LOS and 3σ ellipse",
                                save_path=None,
                                show=False,
                            )

                    # -----------------------------------------------------
                    # Inner KF diagnostics
                    # -----------------------------------------------------
                    detecting_ids_str = _ids_to_str(detecting_ids)

                    if log_inner_kf and timer.curr_od_index > resume_after_step:
                        for j, entry in enumerate(update_history):
                            row_in = {
                                "run_uid": uid,
                                "master_row_idx": m_idx,
                                "rank": rank,
                                "od_step_idx": int(timer.curr_od_index),
                                "inner_update_idx": j,
                                "epoch_jdtdb": float(processed_epochs[j]) if j < len(processed_epochs) else np.nan,
                                "detecting_ids": detecting_ids_str,
                            }

                            xpr = _flatten_vec(entry.get("x_prior", None), 6)
                            xpo = _flatten_vec(entry.get("x_post", None), 6)
                            Ppr = _safe_diag(entry.get("P_prior", None), 6)
                            Ppo = _safe_diag(entry.get("P_post", None), 6)

                            for i in range(6):
                                row_in[f"x_prior_{i}"] = xpr[i]
                                row_in[f"x_post_{i}"] = xpo[i]
                                row_in[f"P_prior_diag_{i}"] = Ppr[i]
                                row_in[f"P_post_diag_{i}"] = Ppo[i]

                            xtrue_i = [np.nan] * 6
                            if len(ast_eme_traj_kms_during_detection) > j:
                                xtrue_i = _flatten_vec(ast_eme_traj_kms_during_detection[j], 6)
                            for i in range(6):
                                row_in[f"x_true_{i}"] = xtrue_i[i]

                            obs_i = _extract_inner_obs_state(sc_states_k, j, detecting_ids)
                            for i in range(6):
                                row_in[f"observer_state_{i}"] = obs_i[i]

                            true_i, noisy_i = _extract_inner_meas_pair(p_meas_k, n_meas_k, j, detecting_ids)
                            for i in range(2):
                                row_in[f"true_meas_{i}"] = true_i[i]
                                row_in[f"noisy_meas_{i}"] = noisy_i[i]

                            _append_row(inner_csv_path, row_in, inner_header)

                    # No detections are no longer terminal. The UKF/SPKF function
                    # now runs a prediction-only pass through the missed tracklet,
                    # and attitude coordination below continues from that propagated prior.
                    if False and ((not had_detection) or (int(n_detections) == 0)):
                        termination_reason = "no_detection"
                        progress_status = "completed"
                        event_type = "termination_no_detection"

                        sc_states_post = np.asarray(formation.get_spacecraft_states(), dtype=float)
                        sc_pointings_post = np.asarray(formation.get_spacecraft_pointings(), dtype=float)

                        x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                        x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                        P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                        pos_err, vel_err = _best_effort_state_error(x_est, x_true)

                        row_out = {
                            "run_uid": uid,
                            "master_row_idx": m_idx,
                            "rank": rank,
                            "od_step_idx": int(timer.curr_od_index),
                            "event_type": event_type,
                            "termination_reason": termination_reason,
                            "epoch_start_jdtdb": epoch_start,
                            "epoch_end_jdtdb": epoch_start,
                            "processed_epoch_first_jdtdb": processed_epoch_first,
                            "processed_epoch_last_jdtdb": processed_epoch_last,
                            "processed_epoch_count": processed_epoch_count,
                            "had_detection": had_detection,
                            "n_detections": 0,
                            "detecting_ids": "",
                            "od_time_sec": od_time,
                            "attcoord_time_sec": np.nan,
                            "true_meas_len": 0,
                            "true_meas_norm": np.nan,
                            "true_meas_0": np.nan,
                            "true_meas_1": np.nan,
                            "noisy_meas_len": 0,
                            "noisy_meas_norm": np.nan,
                            "noisy_meas_0": np.nan,
                            "noisy_meas_1": np.nan,
                            "pos_err_norm": pos_err,
                            "vel_err_norm": vel_err,
                            "chosen_candidate_idx": np.nan,
                            "chosen_candidate_epoch_jdtdb": np.nan,
                            "progress_status": progress_status,
                        }
                        for i in range(6):
                            row_out[f"x_est_{i}"] = x_est[i]
                            row_out[f"x_true_{i}"] = x_true[i]
                            row_out[f"P_diag_{i}"] = P_diag[i]

                        for sc_id in range(num_sc):
                            pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                            pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                            post_s = _flatten_vec(sc_states_post[sc_id], 6)
                            post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                            for i in range(6):
                                row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                                row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                            for i in range(3):
                                row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                                row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                        if timer.curr_od_index > resume_after_step:
                            _append_row(outer_csv_path, row_out, outer_header)

                        _write_progress(progress_path, {
                            "uid": uid,
                            "last_completed_outer_step": int(timer.curr_od_index),
                            "last_epoch_jdtdb": float(epoch_start),
                            "completed": True,
                            "termination_reason": termination_reason,
                            "outer_csv_path": outer_csv_path,
                        })
                        break

                    # -----------------------------------------------------
                    # Optional OD stop criterion: covariance trace + NIS
                    # -----------------------------------------------------
                    should_stop_od, od_convergence_streak, last_od_stop_metrics, od_stop_reason = _od_stop_metrics_and_decision(
                        config,
                        getattr(ukf, "P", None),
                        update_history,
                        od_convergence_streak,
                    )

                    # A prediction-only no-detection step should not by itself satisfy
                    # OD convergence/termination logic. Keep the metrics for logging,
                    # but require at least one actual detection for stop decisions.
                    if (not had_detection) or (int(n_detections) == 0):
                        should_stop_od = False
                        od_convergence_streak = 0
                        last_od_stop_metrics["convergence_streak"] = 0
                        od_stop_reason = "no_detection_continue"

                    if should_stop_od:
                        termination_reason = od_stop_reason
                        progress_status = "completed"
                        event_type = "termination_convergence"

                        sc_states_post = np.asarray(formation.get_spacecraft_states(), dtype=float)
                        sc_pointings_post = np.asarray(formation.get_spacecraft_pointings(), dtype=float)

                        x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                        x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                        P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                        pos_err, vel_err = _best_effort_state_error(x_est, x_true)
                        true_meas_len, true_meas_norm, true_pair = _summarize_meas_outer(p_meas_k)
                        noisy_meas_len, noisy_meas_norm, noisy_pair = _summarize_meas_outer(n_meas_k)

                        row_out = {
                            "run_uid": uid,
                            "master_row_idx": m_idx,
                            "rank": rank,
                            "od_step_idx": int(timer.curr_od_index),
                            "event_type": event_type,
                            "termination_reason": termination_reason,
                            "epoch_start_jdtdb": epoch_start,
                            "epoch_end_jdtdb": float(processed_epoch_last) if np.isfinite(processed_epoch_last) else epoch_start,
                            "processed_epoch_first_jdtdb": processed_epoch_first,
                            "processed_epoch_last_jdtdb": processed_epoch_last,
                            "processed_epoch_count": processed_epoch_count,
                            "had_detection": had_detection,
                            "n_detections": n_detections,
                            "detecting_ids": detecting_ids_str,
                            "od_time_sec": od_time,
                            "attcoord_time_sec": np.nan,
                            "true_meas_len": true_meas_len,
                            "true_meas_norm": true_meas_norm,
                            "true_meas_0": true_pair[0],
                            "true_meas_1": true_pair[1],
                            "noisy_meas_len": noisy_meas_len,
                            "noisy_meas_norm": noisy_meas_norm,
                            "noisy_meas_0": noisy_pair[0],
                            "noisy_meas_1": noisy_pair[1],
                            "P_pos_trace": last_od_stop_metrics["trace_pos"],
                            "P_vel_trace": last_od_stop_metrics["trace_vel"],
                            "P_pos_sigma_3d": last_od_stop_metrics["sigma_pos_3d"],
                            "P_vel_sigma_3d": last_od_stop_metrics["sigma_vel_3d"],
                            "NIS_last": last_od_stop_metrics["nis_last"],
                            "NIS_mean": last_od_stop_metrics["nis_mean"],
                            "NIS_count": last_od_stop_metrics["nis_count"],
                            "OD_convergence_streak": last_od_stop_metrics["convergence_streak"],
                            "pos_err_norm": pos_err,
                            "vel_err_norm": vel_err,
                            "chosen_candidate_idx": np.nan,
                            "chosen_candidate_epoch_jdtdb": np.nan,
                            "progress_status": progress_status,
                        }
                        for i in range(6):
                            row_out[f"x_est_{i}"] = x_est[i]
                            row_out[f"x_true_{i}"] = x_true[i]
                            row_out[f"P_diag_{i}"] = P_diag[i]

                        for sc_id in range(num_sc):
                            pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                            pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                            post_s = _flatten_vec(sc_states_post[sc_id], 6)
                            post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                            for i in range(6):
                                row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                                row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                            for i in range(3):
                                row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                                row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                        if timer.curr_od_index > resume_after_step:
                            _append_row(outer_csv_path, row_out, outer_header)

                        _write_progress(progress_path, {
                            "uid": uid,
                            "last_completed_outer_step": int(timer.curr_od_index),
                            "last_epoch_jdtdb": float(row_out["epoch_end_jdtdb"]),
                            "completed": True,
                            "termination_reason": termination_reason,
                            "outer_csv_path": outer_csv_path,
                        })
                        break

                    # -----------------------------------------------------
                    # Perform Attitude Coordination - with uncertainty growth
                    # -------------------------------------------------------
                    attcoord_startime = time.time()

                    timer.set_attcoord_searchtimes(od_time=od_time)

                    x_ts, P_ts = ukf.propagate_priors(
                        epochs_k[-1],
                        timer.attcoord_searchtimes_jdtdb,
                        n_body_propagator.propagate_multiple_objects
                    )

                    sc_eme_states_kms_piecewise, anchor_info = util.piecewise_anchor_and_propagate_spacecraft_trajs(
                        formation=formation, minimoon=minimoon, timer=timer,
                        t_targets_jdtdb=timer.attcoord_searchtimes_jdtdb,
                        n_body_propagator=n_body_propagator, return_anchor_info=True
                    )

                    ast_eme_state_kms = minimoon.curr_state_eme
                    ast_eme_traj_kms = n_body_propagator.propagate(ast_eme_state_kms, timer.curr_epoch,
                                                                   timer.attcoord_searchtimes_jdtdb)

                    ast_truth_original_eclip = np.array(minimoon.orbit.loc[:, ["Geo x", "Geo y", "Geo z", "Geo vx",
                                                                               "Geo vy", "Geo vz"]])
                    ast_truth_original_eclip[:, :3] *= config['AU_TO_M'] / 1000
                    ast_truth_original_eclip[:, 3:] *= config['AU_TO_M'] / 1000 / config['SECONDS_PER_DAY']
                    ast_truth_original_eme = util.geo_eclip_to_geo_eme_generic(ast_truth_original_eclip,
                                                                               hint=("time", "state"))

                    viz_prop_flag = False
                    if viz_prop_flag:
                        util.plot_priors_positions_and_cov_2d(
                            x_ts,
                            P_ts,
                            sc_trajs_km2=sc_eme_states_kms_piecewise,
                            stride=config['two_d_prop']['stride'],
                            n_std=config['two_d_prop']['stride'],
                            planes=("xy", "xz", "yz"),
                            title_prefix="Asteroid prior + spacecraft"
                        )

                        plot_matched_flag = False
                        if plot_matched_flag:
                            util.plot_matched_trajectory_full_range(
                                formation=formation,
                                idx_start=0,
                                idx_stop=2400,
                                frame="GEO_EME",
                                plot_3d=True,
                                plot_xy=False
                            )

                    sc_pointings_eme = formation.get_spacecraft_pointings()

                    sc0 = formation.spacecraft[0]
                    theta_h_rad = util.fov_deg2_to_half_angle_rad(sc0.fov)
                    tau_max = sc0.reaction_wheel_torque
                    h_max = sc0.reaction_wheel_momentum
                    m_m = sc0.mass
                    l_m = sc0.length
                    m_t = sc0.telescope_mass
                    d_t = sc0.telescope_diameter
                    l_t = sc0.telescope_length
                    z_0 = sc0.telescope_offset
                    I_max = (1 / 6) * m_m * (l_m) ** 2 + (1 / 12) * m_t * (
                                3 * (d_t / 2) ** 2 + l_t ** 2) + m_t * z_0 ** 2
                    alpha_max = 1.63 * tau_max / I_max
                    omega_max = 1.63 * h_max / I_max

                    use_tracking = bool(config_global['use_tracking'])
                    detecting_list = list(formation.currently_detecting)
                    detecting_u = sc_pointings_eme[detecting_list, :] if len(detecting_list) > 0 else np.empty((0, 3))
                    first_detecting_idx = int(detecting_list[0]) if len(detecting_list) > 0 else None
                    first_detecting_u = sc_pointings_eme[first_detecting_idx, :] if first_detecting_idx is not None else None

                    if use_tracking:
                        # Tracking mode does not require a fixed detector. With no detections,
                        # pass an empty fixed/detecting set and let the coordinator optimize from
                        # the propagated prior.
                        res_kcoverage, result_mean, result_kcoverage_series, result_mean_series, mean_time = attitude_coordination.step(
                            sc_eme_states_kms_piecewise[:, :, :3],
                            sc_pointings_eme,
                            x_ts[:, :3],
                            P_ts[:, :3, :3],
                            timer.attcoord_searchtimes,
                            theta_h_rad,
                            alpha_max,
                            omega_max,
                            detecting_u,
                            detecting_list,
                            d_M=config['d_mahal'],
                            use_fixed_agent=False,
                            fixed_agent_idx=None,
                            fixed_agent_u=first_detecting_u,
                            coverage_point=ast_eme_traj_kms[:, :3],
                        )
                    else:
                        if len(detecting_list) == 1:
                            res_kcoverage, result_mean, result_kcoverage_series, result_mean_series, mean_time = attitude_coordination.step(
                                sc_eme_states_kms_piecewise[:, :, :3],
                                sc_pointings_eme,
                                x_ts[:, :3],
                                P_ts[:, :3, :3],
                                timer.attcoord_searchtimes,
                                theta_h_rad,
                                alpha_max,
                                omega_max,
                                detecting_u,
                                detecting_list,
                                d_M=config['d_mahal'],
                                use_fixed_agent=True,
                                fixed_agent_idx=first_detecting_idx,
                                fixed_agent_u=first_detecting_u,
                                coverage_point=ast_eme_traj_kms[:, :3]
                            )
                        else:
                            # Multi-detection and no-detection cases both use free-agent
                            # coordination. In the no-detection case detecting_u/list are empty,
                            # so no spacecraft is treated as fixed by the caller.
                            res_kcoverage, result_mean, result_kcoverage_series, result_mean_series, mean_time = attitude_coordination.step(
                                sc_eme_states_kms_piecewise[:, :, :3],
                                sc_pointings_eme,
                                x_ts[:, :3],
                                P_ts[:, :3, :3],
                                timer.attcoord_searchtimes,
                                theta_h_rad,
                                alpha_max,
                                omega_max,
                                detecting_u,
                                detecting_list,
                                d_M=config['d_mahal'],
                                use_fixed_agent=False,
                                fixed_agent_idx=first_detecting_idx,
                                fixed_agent_u=first_detecting_u,
                                coverage_point=ast_eme_traj_kms[:, :3]
                            )

                    attcoord_endtime = time.time()
                    timer.set_attcoord_time(attcoord_endtime - attcoord_startime - mean_time)
                    attcoord_time = attcoord_endtime - attcoord_startime - mean_time

                    timer.set_slew_time(res_kcoverage.chosen_dt)

                    best_epoch = res_kcoverage.chosen_dt
                    best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                    best_epoch_jdtdb = float(timer.attcoord_searchtimes_jdtdb[best_idx])
                    k_anchor_best = int(anchor_info["anchor_k_of_t"][best_idx])
                    t_anchor_best = float(anchor_info["anchor_epoch_of_t"][best_idx])

                    best_attitudes = res_kcoverage.u_cmd
                    formation.set_spacecraft_pointings(best_attitudes)

                    best_positions = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :])
                    formation.set_spacecraft_states(best_positions)

                    ukf.x = np.squeeze(x_ts[best_idx, :])
                    ukf.P = np.squeeze(P_ts[best_idx, :, :])

                    minimoon.set_state(np.squeeze(ast_eme_traj_kms[best_idx, :]))

                    timer.step(best_epoch_jdtdb, k_anchor_best, t_anchor_best)

                    chosen_candidate_idx = best_idx
                    chosen_candidate_epoch_jdtdb = best_epoch_jdtdb

                    # detailed attcoord logging
                    if log_attcoord and timer.curr_od_index > resume_after_step:
                        for cand_idx in range(len(timer.attcoord_searchtimes_jdtdb)):
                            row_att = {
                                "run_uid": uid,
                                "master_row_idx": m_idx,
                                "rank": rank,
                                "od_step_idx": int(timer.curr_od_index),
                                "candidate_idx": cand_idx,
                                "candidate_epoch_jdtdb": float(timer.attcoord_searchtimes_jdtdb[cand_idx]),
                                "is_chosen": int(cand_idx == best_idx),
                            }
                            xpred = _flatten_vec(x_ts[cand_idx], 6)
                            Ppred = _safe_diag(P_ts[cand_idx], 6)
                            for i in range(6):
                                row_att[f"x_pred_{i}"] = xpred[i]
                                row_att[f"P_pred_diag_{i}"] = Ppred[i]

                            sc_cand = np.asarray(sc_eme_states_kms_piecewise[cand_idx], dtype=float)
                            for sc_id in range(num_sc):
                                vals = _flatten_vec(sc_cand[sc_id], 6)
                                for i in range(6):
                                    row_att[f"sc{sc_id}_state_{i}"] = vals[i]

                            u_cand = None
                            J_cand = np.nan
                            coverage_cand = np.nan
                            if cand_idx < len(result_kcoverage_series):
                                if isinstance(result_kcoverage_series[cand_idx], dict):
                                    u_cand = result_kcoverage_series[cand_idx].get("u", None)
                                    J_cand = _safe_scalar(result_kcoverage_series[cand_idx].get("J", np.nan))
                                    coverage_cand = _safe_scalar(result_kcoverage_series[cand_idx].get("kcoverage", np.nan))
                            if u_cand is None:
                                u_cand = best_attitudes if cand_idx == best_idx else np.full((num_sc, 3), np.nan)
                            u_cand = np.asarray(u_cand, dtype=float)
                            for sc_id in range(num_sc):
                                vals = _flatten_vec(u_cand[sc_id], 3)
                                for i in range(3):
                                    row_att[f"u_cmd_sc{sc_id}_{i}"] = vals[i]

                            row_att["J"] = J_cand
                            row_att["coverage_score"] = coverage_cand
                            _append_row(attcoord_csv_path, row_att, attcoord_header)

                    # optimizer logging based on actual history structure
                    if log_optimizer and hasattr(res_kcoverage, "extra") and isinstance(res_kcoverage.extra, dict):
                        hist = res_kcoverage.extra.get("history", [])
                        if timer.curr_od_index > resume_after_step:
                            for h_idx, entry in enumerate(hist):
                                row_opt = {
                                    "run_uid": uid,
                                    "master_row_idx": m_idx,
                                    "rank": rank,
                                    "od_step_idx": int(timer.curr_od_index),
                                    "optimizer_row_idx": h_idx,
                                    "restart_idx": entry.get("restart", np.nan),
                                    "use_fixed_agent": entry.get("use_fixed_agent", np.nan),
                                    "idx_fix": entry.get("idx_fix", np.nan),
                                    "J": _safe_scalar(entry.get("J", np.nan)),
                                }

                                x_full = _flatten_vec(entry.get("x", None), 12)
                                x_free = _flatten_vec(entry.get("x_free", None), 12)
                                for i in range(12):
                                    row_opt[f"x_param_{i}"] = x_full[i]
                                    row_opt[f"x_free_param_{i}"] = x_free[i]

                                u_hist = np.asarray(entry.get("u", np.full((num_sc, 3), np.nan)), dtype=float)
                                if u_hist.ndim == 1:
                                    u_hist = u_hist.reshape(-1, 3)
                                for sc_id in range(min(num_sc, u_hist.shape[0])):
                                    vals = _flatten_vec(u_hist[sc_id], 3)
                                    for i in range(3):
                                        row_opt[f"u_sc{sc_id}_{i}"] = vals[i]

                                slews_hist = np.asarray(entry.get("slew", np.full((num_sc,), np.nan))).reshape(-1)
                                for sc_id in range(num_sc):
                                    row_opt[f"slew_sc{sc_id}"] = float(slews_hist[sc_id]) if sc_id < slews_hist.size else np.nan

                                _append_row(optimizer_csv_path, row_opt, optimizer_header)

                    # visualize att_coord result
                    # The visualization block assumes at least one detecting spacecraft
                    # in a few diagnostic calculations, so skip it for prediction-only
                    # no-detection steps.
                    att_coord_viz_flag = False
                    if att_coord_viz_flag and had_detection:
                        best_epoch = res_kcoverage.chosen_dt
                        best_idx = np.where(timer.attcoord_searchtimes == best_epoch)[0]

                        theta_h_rad = util.fov_deg2_to_half_angle_rad(config["fov"])
                        ems_center_xy = np.array(config["ems"]["p_em"][:2])
                        ems_center_xyz = np.array(config["ems"]["p_em"])
                        ems_radius = config["ems"]['R_em']
                        ray_length = config['ray_length']

                        sc_eme_ae_kms = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :3])
                        sc_pointing_eme_cartesian = res_kcoverage.u_cmd
                        ang_eme = util.proj_angle_xy_from_plus_x_ccw(sc_pointing_eme_cartesian)
                        ast_truth_eme = np.squeeze(ast_eme_traj_kms[best_idx, :3])

                        ast_iod_eme = np.squeeze(x_ts[best_idx, :3])
                        P_cart_eme = np.squeeze(P_ts[best_idx, :3, :3])

                        ast_truth_eme = np.asarray(ast_truth_eme).reshape(-1)

                        best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                        T = len(timer.attcoord_searchtimes)

                        two_d_pot = False
                        if two_d_pot:
                            def is_valid(idx):
                                J = result_kcoverage_series[idx]["J"]
                                return not (J is None or (isinstance(J, float) and math.isnan(J)))

                            cands = [best_idx, max(0, best_idx - 1), min(T - 1, best_idx + 1)]
                            selected = []

                            for c in cands:
                                if c not in selected and is_valid(c):
                                    selected.append(c)

                            if len(selected) < 3:
                                for c in range(T):
                                    if len(selected) >= 3:
                                        break
                                    if c not in selected and is_valid(c):
                                        selected.append(c)

                            if len(selected) < 3:
                                for c in range(T):
                                    if len(selected) >= 3:
                                        break
                                    if c not in selected:
                                        selected.append(c)

                            planes = [((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ")]

                            def cov2_from_cov3(P3, axes):
                                i, j = axes
                                return P3[np.ix_([i, j], [i, j])]

                            def angles_in_plane(u_cmd_xyz, axes):
                                a, b = axes
                                u = np.asarray(u_cmd_xyz, dtype=float)
                                u2 = u[:, [a, b]]
                                return np.arctan2(u2[:, 1], u2[:, 0])

                            for idx in selected:
                                epoch = timer.attcoord_searchtimes[idx]
                                sc_xyz = np.asarray(sc_eme_states_kms_piecewise[idx, :, :3], dtype=float)
                                ast_truth = np.asarray(ast_eme_traj_kms[idx, :3], dtype=float).reshape(3,)
                                ast_mean = np.asarray(x_ts[idx, :3], dtype=float).reshape(3,)
                                P3 = np.asarray(P_ts[idx, :3, :3], dtype=float).reshape(3, 3)
                                u_cmd_xyz = np.asarray(result_kcoverage_series[idx]["u"], dtype=float)
                                u_curr_xyz = np.asarray(sc_pointings_eme, dtype=float)

                                fig, axes = plt.subplots(1, 3, figsize=(8, 14))
                                fig.suptitle(f"2D Projections @ epoch {epoch}", y=0.99)

                                for ax, (axpair, name) in zip(axes, planes):
                                    agents_2d = sc_xyz[:, list(axpair)]
                                    u_curr_2d = u_curr_xyz[:, list(axpair)]
                                    mean_2d = ast_mean[list(axpair)]
                                    truth_2d = ast_truth[list(axpair)]
                                    cov_2d = cov2_from_cov3(P3, axpair)
                                    mean_traj_2d = np.asarray(x_ts[:, :3], dtype=float)[:, list(axpair)]
                                    truth_traj_2d = np.asarray(ast_eme_traj_kms[:, :3], dtype=float)[:, list(axpair)]
                                    ems_center_2d = np.asarray(ems_center_xyz, dtype=float)[list(axpair)]
                                    ang_2d = angles_in_plane(u_cmd_xyz, axpair)

                                    util.plot_od_scenario_2d(
                                        t_label=f"{epoch} ({name})",
                                        agents_xy=agents_2d,
                                        pointing_angles_rad=ang_2d,
                                        theta_h_rad=theta_h_rad,
                                        ray_length=ray_length * 2,
                                        u_curr_agents_xy=u_curr_2d,
                                        boresight_line_len=ray_length * 0.1,
                                        target_mean_xy=mean_2d,
                                        target_mean_xy_traj=mean_traj_2d,
                                        target_cov_xy=cov_2d,
                                        d_mahal=config['d_mahal'],
                                        true_target_xy=truth_2d,
                                        true_target_xy_traj=truth_traj_2d,
                                        ems_center_xy=ems_center_2d,
                                        ems_radius=ems_radius,
                                        xlim=config['two_d_prop']['xlim'], ylim=config['two_d_prop']['ylim'],
                                        agent_orbit_tracks_xy=None,
                                        ax=ax,
                                        title=None
                                    )

                                plt.tight_layout()



                        agents_xyz = sc_eme_ae_kms
                        target_cov_xyz = P_cart_eme

                        def range_sigma_from_cov(P_eci, r_obj, r_obs):
                            """
                            P_eci : 6x6 or 3x3 covariance in inertial frame
                            r_obj : object inertial position, shape (3,)
                            r_obs : observer inertial position, shape (3,)

                            Returns range standard deviation and variance.
                            """
                            P_r = P_eci[:3, :3] if P_eci.shape == (6, 6) else P_eci

                            rho_vec = np.asarray(r_obj) - np.asarray(r_obs)
                            rho_hat = rho_vec / np.linalg.norm(rho_vec)

                            var_rho = rho_hat @ P_r @ rho_hat
                            sigma_rho = np.sqrt(max(var_rho, 0.0))

                            return sigma_rho, var_rho

                        sigma_rho, var_rho = range_sigma_from_cov(P_cart_eme, ast_iod_eme[:3],
                                                                  agents_xyz[formation.currently_detecting[0]])
                        print(sigma_rho, var_rho)

                        def build_slew_history_from_opt_series(opt_series, ids, *, key_u="u", normalize=True):
                            ids = [int(i) for i in ids]
                            out = {i: [] for i in ids}

                            for row_idx, rowh in enumerate(opt_series):
                                if key_u not in rowh:
                                    raise KeyError(f"Row {row_idx} missing key '{key_u}'")
                                U = np.asarray(rowh[key_u], dtype=float)
                                if U.ndim != 2 or U.shape[1] != 3:
                                    raise ValueError(f"Row {row_idx} '{key_u}' must be (M,3), got {U.shape}")
                                M = U.shape[0]
                                for i in ids:
                                    if not (0 <= i < M):
                                        raise IndexError(f"Row {row_idx}: id {i} out of range for M={M}")
                                    out[i].append(U[i].copy())

                            for i in ids:
                                Ui = np.asarray(out[i], dtype=float)
                                if Ui.size == 0:
                                    Ui = Ui.reshape(0, 3)

                                if normalize and Ui.shape[0] > 0:
                                    n = np.linalg.norm(Ui, axis=1, keepdims=True)
                                    n = np.maximum(n, 1e-12)
                                    Ui = Ui / n

                                out[i] = Ui

                            return out

                        ids = config['three_d_prop'].get('history_ids')
                        if ids is None:
                            ids = list(range(config['num_spacecraft']))

                        slew_history = build_slew_history_from_opt_series(res_kcoverage.extra["history"], ids)

                        fig, ax = util.plot_od_scenario_3d_new(
                            t_label=best_epoch,
                            agents_xyz=agents_xyz,
                            u_opt_agents_xyz=sc_pointing_eme_cartesian,
                            theta_h_rad=theta_h_rad,
                            ray_length=ray_length,
                            u_curr_agents_xyz=sc_pointings_eme,
                            boresight_line_len=ray_length * 0.1,
                            u_init_agents_xyz=None,
                            init_boresight_line_len=ray_length * 1.5,
                            agent_orbit_tracks_xyz=[sc_eme_states_kms_piecewise[:, i, :3] for i in
                                                    range(config['num_spacecraft'])],
                            spacecraft_orbit_xyz=None,
                            xlim=config['three_d_prop']['xlim'], ylim=config['three_d_prop']['ylim'],
                            zlim=config['three_d_prop']['zlim'],
                            target_mean_xyz=ast_iod_eme[:3],
                            target_cov_xyz=target_cov_xyz,
                            d_mahal=config['d_mahal'],
                            true_target_xyz=ast_truth_eme[:3],
                            target_mean_traj_xyz=x_ts[:, :3],
                            true_target_traj_xyz=ast_eme_traj_kms[:, :3],
                            true_target_traj_xyz_2=ast_truth_original_eme[:, :3],
                            ems_center_xyz=ems_center_xyz,
                            ems_radius=ems_radius,
                            show_uncertainty=True,
                            show_truth=True,  # green circle
                            show_ems=True,
                            show_fov_cones=True,
                            show_legend=True,
                            show_target_mean_traj=True,
                            show_true_target_traj=True,
                            show_true_target_traj_2=False,
                            show_init_boresights=False,
                            show_current_boresights=False,
                            show_slew_angle_annotations=False,
                            show_agent_name_annotations=True,
                            show_agent_orbit_tracks=True,
                            show_spacecraft_orbit=False,
                            show_coverage=False,

                            Nx=60,
                            Ny=60,
                            Nz=40,

                            # coverage display
                            show_pair_coverage=True,
                            show_triple_coverage=True,
                            pair_only_exact=True,

                            pair_coverage_alpha=0.25,
                            triple_coverage_alpha=0.35,

                            # FOV styling
                            fov_style="surface",  # "surface", "wire", "both"
                            fov_surface_alpha=0.12,
                            fov_surface_color="lightskyblue",
                            fov_n_rays=2,
                            fov_n_circle=64,
                            fov_n_len=20,

                            # styling
                            title=None,
                            label_fontsize=9,
                            label_offset_px=10,
                            slew_label_offset_px=16,
                            fill_alpha=0.10,
                            sparse_wire=True,

                            init_boresight_lw=1.5,
                            init_boresight_alpha=0.95,

                            # slew history
                            slew_history=None,
                            slew_history_line_len=ray_length,
                            slew_history_lw=1.8,
                            slew_history_alpha=0.85,
                            slew_history_cmap="viridis",
                            slew_history_every=1,
                            slew_history_colorbar=True,
                            slew_history_colorbar_label="Slew history step",
                            slew_history_norm_mode="per_agent",
                        )

                        cost_func_plots = True
                        if cost_func_plots:
                            util.plot_attcoord_costs_from_series(
                                result_kcoverage_series,
                                title="Dual coverage score vs objective (best per epoch)"
                            )

                            thetas, phis = util.plot_theta_phi_over_history(
                                res_kcoverage.extra["history"],
                                int(config["num_spacecraft"]),
                                deg=True
                            )

                        viz_cost_map = False
                        if viz_cost_map:
                            idx_fix = formation.currently_detecting
                            idx_free = 1 - idx_fix
                            u_fix = sc_pointings_eme[idx_fix]

                            TH_free_deg, PH_free_deg, J_grid = compute_J_grid_theta_phi_single_free(
                                ast_iod_eme[:3], target_cov_xyz, agents_xyz, sc_pointings_eme, theta_h_rad,
                                idx_fix=idx_fix,
                                u_fix=u_fix,
                                idx_free=idx_free,
                                theta_range_rad=(-0.5 * np.pi, 0.5 * np.pi),
                                phi_range_rad=(0.0, 2 * np.pi),
                                d_M=config['d_mahal'], kappa_sigma=config['optimizer_att_coord']['kappa_sigma'],
                                lambda_k1=config['optimizer_att_coord']['lambda_k1'],
                                n_mc=config['opt_map']['n_mc'],
                                n_grid_theta=config['opt_map']['n_grid_theta'],
                                n_grid_phi=config['opt_map']['n_grid_phi'],
                            )

                            fig3d = plt.figure(figsize=(9, 6))
                            ax3d = fig3d.add_subplot(111, projection="3d")
                            ax3d.plot_surface(
                                TH_free_deg, PH_free_deg, J_grid,
                                rstride=config['opt_map']['rstride'], cstride=config['opt_map']['cstride'],
                                linewidth=config['opt_map']['linewidth_3d'], alpha=config['opt_map']['alpha']
                            )
                            ax3d.set_xlabel(rf'$\theta_{{{idx_free}}}$ (deg)')
                            ax3d.set_ylabel(rf'$\phi_{{{idx_free}}}$ (deg)')
                            ax3d.set_zlabel(r'$J_t$')
                            ax3d.set_title(rf'$J_t(\theta,\phi)$ for free agent {idx_free} (fixed agent {idx_fix})')
                            plt.tight_layout()

                            plt.figure(figsize=(7, 5.5))
                            cs = plt.contourf(TH_free_deg, PH_free_deg, J_grid, levels=config['opt_map']['levels'])
                            plt.colorbar(cs, label=r'$J_t$')
                            plt.xlabel(rf'$\theta_{{{idx_free}}}$ (deg)')
                            plt.ylabel(rf'$\phi_{{{idx_free}}}$ (deg)')
                            plt.title(rf'$J_t(\theta,\phi)$ for free agent {idx_free} (fixed agent {idx_fix})')
                            plt.grid(alpha=0.3)

                            restart_indices = sorted({entry["restart"] for entry in res_kcoverage.extra["history"]})
                            colors = ['white', 'yellow', 'cyan', 'magenta', 'green', 'orange']
                            markers = ['o', 's', '^', 'D', 'x', '+']

                            for k, r in enumerate(restart_indices):
                                path_entries = [entry for entry in res_kcoverage.extra["history"] if entry["restart"] == r]
                                if not path_entries:
                                    continue

                                theta_path_deg = []
                                phi_path_deg = []

                                for entry in path_entries:
                                    if entry.get("use_fixed_agent", False) and ("x_free" in entry) and (
                                            entry.get("idx_fix", None) is not None):
                                        xk = np.asarray(entry["x_free"], dtype=float).ravel()
                                        th = xk[0]
                                        ph = xk[1]
                                    else:
                                        xk = np.asarray(entry["x"], dtype=float).ravel()
                                        th = xk[2 * idx_free]
                                        ph = xk[2 * idx_free + 1]

                                    theta_path_deg.append(np.rad2deg(th))
                                    phi_path_deg.append(np.rad2deg(ph))

                                theta_path_deg = np.asarray(theta_path_deg)
                                phi_path_deg = np.asarray(phi_path_deg)

                                col = colors[k % len(colors)]
                                m = markers[k % len(markers)]

                                label = f"Trial {r}"
                                if r == 0:
                                    label += " (warm start)"

                                plt.plot(
                                    theta_path_deg, phi_path_deg,
                                    linestyle='-',
                                    marker=m,
                                    color=col,
                                    lw=config['opt_map']['linewidth_2d'],
                                    ms=config['opt_map']['marker_size'],
                                    label=label
                                )

                            plt.legend(loc='upper right')

                    view_other_epoch_res_flag = False
                    if view_other_epoch_res_flag:
                        desired_idx = [15, 16, 17, 18, 19, 20, 21, 22]
                        for des_idx in desired_idx:

                            row = result_kcoverage_series[des_idx]

                            best_epoch = row["dt"]
                            best_idx = np.where(timer.attcoord_searchtimes == best_epoch)[0]

                            theta_h_rad = util.fov_deg2_to_half_angle_rad(config["fov"])
                            ems_center_xyz = np.array(config["ems"]["p_em"])
                            ems_radius = config["ems"]['R_em']
                            ray_length = config['ray_length']

                            sc_eme_ae_kms = np.squeeze(sc_eme_states_kms_piecewise[best_idx, :, :3])
                            sc_pointing_eme_cartesian = row['u']
                            ang_eme = util.proj_angle_xy_from_plus_x_ccw(sc_pointing_eme_cartesian)
                            ast_truth_eme = np.squeeze(ast_eme_traj_kms[best_idx, :3])

                            ast_iod_eme = np.squeeze(x_ts[best_idx, :3])
                            P_cart_eme = np.squeeze(P_ts[best_idx, :3, :3])

                            ast_truth_eme = np.asarray(ast_truth_eme).reshape(-1)

                            best_idx = int(np.where(timer.attcoord_searchtimes == best_epoch)[0][0])
                            T = len(timer.attcoord_searchtimes)

                            agents_xyz = sc_eme_ae_kms
                            target_cov_xyz = P_cart_eme


                            def build_slew_history_from_opt_series(opt_series, ids, *, key_u="u", normalize=True):
                                ids = [int(i) for i in ids]
                                out = {i: [] for i in ids}

                                for row_idx, rowh in enumerate(opt_series):
                                    if key_u not in rowh:
                                        raise KeyError(f"Row {row_idx} missing key '{key_u}'")
                                    U = np.asarray(rowh[key_u], dtype=float)
                                    if U.ndim != 2 or U.shape[1] != 3:
                                        raise ValueError(f"Row {row_idx} '{key_u}' must be (M,3), got {U.shape}")
                                    M = U.shape[0]
                                    for i in ids:
                                        if not (0 <= i < M):
                                            raise IndexError(f"Row {row_idx}: id {i} out of range for M={M}")
                                        out[i].append(U[i].copy())

                                for i in ids:
                                    Ui = np.asarray(out[i], dtype=float)
                                    if Ui.size == 0:
                                        Ui = Ui.reshape(0, 3)

                                    if normalize and Ui.shape[0] > 0:
                                        n = np.linalg.norm(Ui, axis=1, keepdims=True)
                                        n = np.maximum(n, 1e-12)
                                        Ui = Ui / n

                                    out[i] = Ui

                                return out

                            ids = config['three_d_prop'].get('history_ids')
                            if ids is None:
                                ids = list(range(config['num_spacecraft']))

                            slew_history = build_slew_history_from_opt_series(row["history"], ids)

                            fig, ax = util.plot_od_scenario_3d_new(
                                t_label=best_epoch,
                                agents_xyz=agents_xyz,
                                u_opt_agents_xyz=sc_pointing_eme_cartesian,
                                theta_h_rad=theta_h_rad,
                                ray_length=ray_length,
                                u_curr_agents_xyz=sc_pointings_eme,
                                boresight_line_len=ray_length * 0.1,
                                u_init_agents_xyz=None,
                                init_boresight_line_len=ray_length * 1.5,
                                agent_orbit_tracks_xyz=[sc_eme_states_kms_piecewise[:, i, :3] for i in
                                                        range(config['num_spacecraft'])],
                                spacecraft_orbit_xyz=None,
                                xlim=config['three_d_prop']['xlim'], ylim=config['three_d_prop']['ylim'],
                                zlim=config['three_d_prop']['zlim'],
                                target_mean_xyz=ast_iod_eme[:3],
                                target_cov_xyz=target_cov_xyz,
                                d_mahal=config['d_mahal'],
                                true_target_xyz=ast_truth_eme[:3],
                                target_mean_traj_xyz=x_ts[:, :3],
                                true_target_traj_xyz=ast_eme_traj_kms[:, :3],
                                true_target_traj_xyz_2=ast_truth_original_eme[:, :3],
                                ems_center_xyz=ems_center_xyz,
                                ems_radius=ems_radius,
                                show_uncertainty=True,
                                show_truth=True,  # green circle
                                show_ems=True,
                                show_fov_cones=True,
                                show_legend=True,
                                show_target_mean_traj=True,
                                show_true_target_traj=True,
                                show_true_target_traj_2=False,
                                show_init_boresights=False,
                                show_current_boresights=False,
                                show_slew_angle_annotations=False,
                                show_agent_name_annotations=True,
                                show_agent_orbit_tracks=True,
                                show_spacecraft_orbit=False,
                                show_coverage=True,

                                Nx=60,
                                Ny=60,
                                Nz=40,

                                # coverage display
                                show_pair_coverage=True,
                                show_triple_coverage=True,
                                pair_only_exact=True,

                                pair_coverage_alpha=0.25,
                                triple_coverage_alpha=0.35,

                                # FOV styling
                                fov_style="surface",  # "surface", "wire", "both"
                                fov_surface_alpha=0.12,
                                fov_surface_color="lightskyblue",
                                fov_n_rays=2,
                                fov_n_circle=64,
                                fov_n_len=20,

                                # styling
                                title=None,
                                label_fontsize=9,
                                label_offset_px=10,
                                slew_label_offset_px=16,
                                fill_alpha=0.10,
                                sparse_wire=True,

                                init_boresight_lw=1.5,
                                init_boresight_alpha=0.95,

                                # slew history
                                slew_history=slew_history,
                                slew_history_line_len=ray_length,
                                slew_history_lw=1.8,
                                slew_history_alpha=0.85,
                                slew_history_cmap="viridis",
                                slew_history_every=1,
                                slew_history_colorbar=True,
                                slew_history_colorbar_label="Slew history step",
                                slew_history_norm_mode="per_agent",
                            )


                    # plt.show()

                # ---------------------------------------------------------
                # Outer-loop diagnostics row
                # ---------------------------------------------------------
                sc_states_post = np.asarray(formation.get_spacecraft_states(), dtype=float)
                sc_pointings_post = np.asarray(formation.get_spacecraft_pointings(), dtype=float)

                x_est = _flatten_vec(getattr(ukf, "x", None), 6)
                x_true = _flatten_vec(getattr(minimoon, "curr_state_eme", None), 6)
                P_diag = _safe_diag(getattr(ukf, "P", None), 6)
                pos_err, vel_err = _best_effort_state_error(x_est, x_true)

                true_meas_len, true_meas_norm, true_pair = _summarize_meas_outer(p_meas_k)
                noisy_meas_len, noisy_meas_norm, noisy_pair = _summarize_meas_outer(n_meas_k)

                row_out = {
                    "run_uid": uid,
                    "master_row_idx": m_idx,
                    "rank": rank,
                    "od_step_idx": int(timer.curr_od_index),
                    "event_type": event_type,
                    "termination_reason": termination_reason,
                    "epoch_start_jdtdb": epoch_start,
                    "epoch_end_jdtdb": float(timer.curr_epoch),
                    "processed_epoch_first_jdtdb": processed_epoch_first,
                    "processed_epoch_last_jdtdb": processed_epoch_last,
                    "processed_epoch_count": processed_epoch_count,
                    "had_detection": had_detection,
                    "n_detections": n_detections,
                    "detecting_ids": detecting_ids_str,
                    "od_time_sec": od_time,
                    "attcoord_time_sec": attcoord_time,
                    "true_meas_len": true_meas_len,
                    "true_meas_norm": true_meas_norm,
                    "true_meas_0": true_pair[0],
                    "true_meas_1": true_pair[1],
                    "noisy_meas_len": noisy_meas_len,
                    "noisy_meas_norm": noisy_meas_norm,
                    "noisy_meas_0": noisy_pair[0],
                    "noisy_meas_1": noisy_pair[1],
                    "P_pos_trace": last_od_stop_metrics["trace_pos"],
                    "P_vel_trace": last_od_stop_metrics["trace_vel"],
                    "P_pos_sigma_3d": last_od_stop_metrics["sigma_pos_3d"],
                    "P_vel_sigma_3d": last_od_stop_metrics["sigma_vel_3d"],
                    "NIS_last": last_od_stop_metrics["nis_last"],
                    "NIS_mean": last_od_stop_metrics["nis_mean"],
                    "NIS_count": last_od_stop_metrics["nis_count"],
                    "OD_convergence_streak": last_od_stop_metrics["convergence_streak"],
                    "pos_err_norm": pos_err,
                    "vel_err_norm": vel_err,
                    "chosen_candidate_idx": chosen_candidate_idx,
                    "chosen_candidate_epoch_jdtdb": chosen_candidate_epoch_jdtdb,
                    "progress_status": progress_status,
                }

                for i in range(6):
                    row_out[f"x_est_{i}"] = x_est[i]
                    row_out[f"x_true_{i}"] = x_true[i]
                    row_out[f"P_diag_{i}"] = P_diag[i]

                for sc_id in range(num_sc):
                    pre_s = _flatten_vec(sc_states_pre[sc_id], 6)
                    pre_u = _flatten_vec(sc_pointings_pre[sc_id], 3)
                    post_s = _flatten_vec(sc_states_post[sc_id], 6)
                    post_u = _flatten_vec(sc_pointings_post[sc_id], 3)
                    for i in range(6):
                        row_out[f"sc{sc_id}_state_pre_{i}"] = pre_s[i]
                        row_out[f"sc{sc_id}_state_post_{i}"] = post_s[i]
                    for i in range(3):
                        row_out[f"sc{sc_id}_pointing_pre_{i}"] = pre_u[i]
                        row_out[f"sc{sc_id}_pointing_post_{i}"] = post_u[i]

                if timer.curr_od_index > resume_after_step:
                    _append_row(outer_csv_path, row_out, outer_header)
                    _write_progress(progress_path, {
                        "uid": uid,
                        "last_completed_outer_step": int(timer.curr_od_index),
                        "last_epoch_jdtdb": float(timer.curr_epoch),
                        "completed": False,
                        "termination_reason": "",
                        "outer_csv_path": outer_csv_path,
                        "checkpoint_path": (checkpoint_path if checkpoint_enabled else None),
                    })
                    if checkpoint_enabled:
                        _save_od_checkpoint(
                            checkpoint_path, uid=uid, m_idx=m_idx, rank=rank, ukf=ukf, timer=timer,
                            minimoon=minimoon, formation=formation,
                            od_convergence_streak=od_convergence_streak,
                            last_od_stop_metrics=last_od_stop_metrics,
                            last_completed_outer_step=int(timer.curr_od_index),
                            outer_csv_path=outer_csv_path,
                            termination_reason="",
                            progress_status=progress_status,
                        )

                print_od_status(
                    timer=timer,
                    ukf=ukf,
                    minimoon=minimoon,
                    formation=formation,
                    x_true=None,
                    n_detections=n_detections,
                    status_every=status_every,
                    prefix=f"[OD r{rank} uid={uid}]",
                )

            # after loop
            final_epoch = float(getattr(timer, "curr_epoch", np.nan))
            final_steps = int(getattr(timer, "curr_od_index", 0))

            if row_paused_for_walltime:
                # Incomplete by design: leave no .done marker so the next job resumes this row.
                # Stop this rank's row loop too; near walltime, starting another row is unsafe.
                break

            if od_master_enabled:
                od_summary = _build_od_master_row_from_outer_last(
                    uid=uid, m_idx=m_idx, rank=rank, outer_csv_path=outer_csv_path, num_sc=num_sc,
                    termination_reason_fallback=termination_reason,
                    final_epoch_fallback=final_epoch,
                    final_steps_fallback=final_steps,
                )
                _append_od_master_row(od_master_rank_path, od_summary, od_master_header)

            _write_progress(progress_path, {
                "uid": uid,
                "last_completed_outer_step": final_steps,
                "last_epoch_jdtdb": final_epoch,
                "completed": True,
                "termination_reason": termination_reason,
                "outer_csv_path": outer_csv_path,
                "checkpoint_path": (checkpoint_path if checkpoint_enabled else None),
            })
            write_done(uid)
            processed += 1

        except Exception as e:
            errors += 1
            traceback.print_exc()

            try:
                row_out = {
                    "run_uid": uid,
                    "master_row_idx": m_idx,
                    "rank": rank,
                    "od_step_idx": int(getattr(timer, "curr_od_index", -1)) if "timer" in locals() else -1,
                    "event_type": "termination_error",
                    "termination_reason": f"{type(e).__name__}: {e}",
                    "epoch_start_jdtdb": float(getattr(timer, "curr_epoch", np.nan)) if "timer" in locals() else np.nan,
                    "epoch_end_jdtdb": float(getattr(timer, "curr_epoch", np.nan)) if "timer" in locals() else np.nan,
                    "processed_epoch_first_jdtdb": np.nan,
                    "processed_epoch_last_jdtdb": np.nan,
                    "processed_epoch_count": 0,
                    "had_detection": False,
                    "n_detections": 0,
                    "detecting_ids": "",
                    "od_time_sec": np.nan,
                    "attcoord_time_sec": np.nan,
                    "true_meas_len": 0,
                    "true_meas_norm": np.nan,
                    "true_meas_0": np.nan,
                    "true_meas_1": np.nan,
                    "noisy_meas_len": 0,
                    "noisy_meas_norm": np.nan,
                    "noisy_meas_0": np.nan,
                    "noisy_meas_1": np.nan,
                    "pos_err_norm": np.nan,
                    "vel_err_norm": np.nan,
                    "chosen_candidate_idx": np.nan,
                    "chosen_candidate_epoch_jdtdb": np.nan,
                    "progress_status": "error",
                }
                for i in range(6):
                    row_out[f"x_est_{i}"] = np.nan
                    row_out[f"x_true_{i}"] = np.nan
                    row_out[f"P_diag_{i}"] = np.nan
                for sc_id in range(num_sc):
                    for i in range(6):
                        row_out[f"sc{sc_id}_state_pre_{i}"] = np.nan
                        row_out[f"sc{sc_id}_state_post_{i}"] = np.nan
                    for i in range(3):
                        row_out[f"sc{sc_id}_pointing_pre_{i}"] = np.nan
                        row_out[f"sc{sc_id}_pointing_post_{i}"] = np.nan

                _append_row(outer_csv_path, row_out, outer_header)
                err_final_epoch = float(getattr(timer, "curr_epoch", np.nan)) if "timer" in locals() else np.nan
                err_final_step = int(getattr(timer, "curr_od_index", -1)) if "timer" in locals() else -1
                _write_progress(progress_path, {
                    "uid": uid,
                    "last_completed_outer_step": err_final_step,
                    "last_epoch_jdtdb": err_final_epoch,
                    "completed": True,
                    "termination_reason": f"ERROR_TERMINAL: {type(e).__name__}: {e}",
                    "outer_csv_path": outer_csv_path,
                    "checkpoint_path": (checkpoint_path if checkpoint_enabled else None),
                })
                if od_master_enabled:
                    od_summary = _build_od_master_row_from_outer_last(
                        uid=uid, m_idx=m_idx, rank=rank, outer_csv_path=outer_csv_path, num_sc=num_sc,
                        termination_reason_fallback=f"ERROR_TERMINAL: {type(e).__name__}: {e}",
                        final_epoch_fallback=err_final_epoch,
                        final_steps_fallback=err_final_step,
                    )
                    _append_od_master_row(od_master_rank_path, od_summary, od_master_header)
                write_done(uid)
            except Exception:
                try:
                    write_done(uid)
                except Exception:
                    pass

    # ---------------------------------------------------------
    # Optional rank-0 merge: per-rank OD master rows -> MASTER_IOD.csv
    # ---------------------------------------------------------
    comm.Barrier()

    if rank == 0:
        merged_rows = 0
        if od_master_enabled and od_master_merge_at_end:
            try:
                merged_rows = _merge_od_master_rows_into_master(
                    master_fn, od_master_dir, od_master_glob, od_master_header
                )
                print(f"[Stage: OD] merged {merged_rows} OD summary rows into {master_fn}", flush=True)
            except Exception as e:
                print(f"[Stage: OD] WARNING: could not merge OD master rows into MASTER_IOD.csv: {e}", flush=True)
                traceback.print_exc()

        print(f"[Stage: OD] rank0 summary: processed={processed}, skipped={skipped}, errors={errors}")


def run_overall_OD(master, config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("Starting...")

    # -------------------------
    # Stage 1: Detection (skip if visible/spacecraft_X not empty)
    # -------------------------
    vis_dir = util._visible_dir(config)
    if rank == 0:
        os.makedirs(vis_dir, exist_ok=True)
        do_detection = (len(util._non_hidden_entries(vis_dir)) == 0)
        if do_detection:
            print(f"[Stage: detection] {vis_dir} is EMPTY - run detection")
        else:
            print(f"[Stage: detection] {vis_dir} is NOT empty - skip detection")
    else:
        do_detection = None
    do_detection = comm.bcast(do_detection, root=0)

    if do_detection:
        run_runs_x_minimoons_MPI(master, config)
    comm.Barrier()

    # -------------------------
    # Stage 2: IOD Data Generation
    # -------------------------
    iod_dir = util._iod_dir(config)
    top_dir = os.path.abspath(config['top_dir'])
    save_format = config.get('save_format', 'csv')

    if rank == 0:
        os.makedirs(iod_dir, exist_ok=True)

        bases = util._source_basenames_in_visible(vis_dir, save_format)
        # If no detection inputs exist, there is nothing to do here.
        if len(bases) == 0:
            do_iod = False
            msg = f"[Stage: IOD] No detection inputs found in {vis_dir} → skip"
        else:
            # A source is considered fully processed if marker exists
            done_markers = [os.path.join(iod_dir, f".done_{b}.json") for b in bases]
            done_flags = [os.path.exists(m) for m in done_markers]
            n_done = sum(done_flags)
            do_iod = not all(done_flags)
            msg = (f"[Stage: IOD] {'RUN' if do_iod else 'SKIP'} — "
                   f"{n_done}/{len(bases)} sources completed in {iod_dir}")
        print(msg)
    else:
        do_iod = None
    do_iod = comm.bcast(do_iod, root=0)

    if do_iod:
        run_sim_runnumbers_MPI_getIOD(config)
    else:
        # Even if we skip, ensure all ranks stay in sync
        comm.Barrier()

    # -------------------------
    # Optional Stage 3H: IOD hyperparameter tuning only
    # -------------------------
    if bool(config.get("run_IOD_hyperparameter", False)):
        if rank == 0:
            print("[Stage: IOD Hyperparameter] RUN — tuning mode enabled; normal IOD/OD stages will be skipped after tuning.", flush=True)
        run_IOD_hyperparameter(config)
        comm.Barrier()
        return

    # -----------------------
    # Stage 3: Running IOD (skip if MASTER rows already committed)
    # -----------------------
    if rank == 0:
        master_path = os.path.join(top_dir, "MASTER_IOD.csv")
        stage3_done_dir = os.path.join(iod_dir, "iod_stage3_done")

        if not os.path.exists(master_path):
            do_stage3 = False
            msg3 = f"[Stage: IOD Solve] No MASTER_IOD.csv in {top_dir} → skip"
        else:
            # Derive per-row UIDs exactly like run_IOD and see how many are done

            try:
                dfm = pd.read_csv(master_path)
                n_rows = len(dfm)
                if n_rows == 0:
                    do_stage3 = False
                    msg3 = f"[Stage: IOD Solve] MASTER_IOD.csv has 0 rows → skip"
                else:
                    # helper to derive uid from IOD_DATA_SAVED_AS with fallback
                    def uid_for_row(row_idx, saved_as_value):
                        s = str(saved_as_value or "")
                        uid = None
                        if s.strip():
                            first = s.split(";")[0].strip()
                            if first:
                                uid = os.path.splitext(os.path.basename(first))[0]
                        return uid if (uid and str(uid).strip()) else f"rowidx_{row_idx}"

                    # ensure marker dir exists
                    os.makedirs(stage3_done_dir, exist_ok=True)

                    # count how many rows have a done marker
                    done_count = 0
                    for i, saved_as in enumerate(dfm.get("IOD_DATA_SAVED_AS", pd.Series([None] * n_rows))):
                        uid = uid_for_row(i, saved_as)
                        marker = os.path.join(stage3_done_dir, f"{uid}.done")
                        if os.path.exists(marker):
                            done_count += 1

                    do_stage3 = (done_count < n_rows)
                    msg3 = (f"[Stage: IOD Solve] {'RUN' if do_stage3 else 'SKIP'} — "
                            f"{done_count}/{n_rows} rows committed in {stage3_done_dir}")
            except Exception as e:
                do_stage3 = False
                msg3 = f"[Stage: IOD Solve] Failed to inspect MASTER_IOD.csv: {e}"
        print(msg3)
    else:
        do_stage3 = None

    do_stage3 = comm.bcast(do_stage3, root=0)

    if do_stage3:
        run_IOD(config)
    else:
        comm.Barrier()

    # -----------------
    # Stage 4: Run the ATT.COOR. + OD Pipeline
    # ------------------
    if config['run_OD']:
        run_OD(config)

    return


###########################
# run sim
##########################

# Argument parser to get the config file path
parser = argparse.ArgumentParser(description="Run the spacecraft simulation")
parser.add_argument('--config', type=str, required=True, help="Path to the config file")
args = parser.parse_args()

# Load the config file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# get the master file
master = util.read_csv_comma_or_space(config['minimoon_master_file_path'], header=0)

###################################
# Run parallel for number of runs using MPI
####################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# run_sim_runnumbers_MPI(master, config)

####################################
# Run parrallel sim to get IOD data using MPI
###################################

# run_sim_runnumbers_MPI_getIOD_data_bychunk_eme(config)

###################################
# Run IOD simulation in parallel
###################################

# run_IOD_MPI(config)

###################################
# Testing ground for IOD
##################################

# run_IOD_testing(config)

###################################
# Hyperparameter tuning for IOD
##################################

# run_IOD_hyperparameter(config)  # only parallel over combos
# run_IOD_hyperparameter_run_par(config)  # parallel over combos and runs
# run_IOD_hyperparameter_run_par_resumable(config)  # par over combos and runs, visualization flag, can continue from where you left off
# run_IOD_hyperparameter_run_par_resumable_bhcolloc(config)

################################
# Overall OD
################################

run_overall_OD(master, config)
