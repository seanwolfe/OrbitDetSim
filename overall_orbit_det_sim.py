import yaml
import os
import pandas as pd
from Asteroid import Asteroid
from Formation import Formation
import numpy as np
import mpi4py.rc
from astropy import units as u
mpi4py.rc.threads = False
from mpi4py import MPI
import spiceypy as sp
import utilities as util
import n_body_integrator as nbody
import argparse
import itertools
import json
import csv
import gc
import glob
import datetime as dt


# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
sp.furnsh("de430.bsp")
sp.furnsh('naif0012.tls')


def run_runs_x_minimoons_MPI(minimoon_master, config):
    """
    Distribute work over the Cartesian product of:
      run_number in [1..number_of_runs]  AND  minimoon_master rows [0..M-1]
    Saves outputs under a directory named `spacecraft_<num_spacecraft>` and
    includes that tag in the filename.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ----------------- config -----------------
    N_runs   = int(config['number_of_runs'])
    M_mm     = int(len(minimoon_master))
    rows_per_part = int(config.get('number_of_rows_per_part', 50_000))
    save_format   = config.get('save_format', 'csv')  # 'csv' or 'parquet'
    base_out      = str(config['output_df_file_name'])
    base_seed     = int(config.get('seed', 12345))

    # required: num_spacecraft in config
    if 'num_spacecraft' not in config:
        raise ValueError("config must include 'num_spacecraft'.")
    num_sc = int(config['num_spacecraft'])

    # ------------- output directory -----------
    base_dir  = os.path.dirname(base_out) or "."
    base_name = os.path.basename(base_out)
    out_dir   = os.path.join(base_dir, f"spacecraft_{num_sc}")

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    comm.Barrier()  # ensure dir exists before others write

    # ------------- task decomposition ----------
    # Total tasks = N_runs * M_mm. Assign a disjoint slice per rank.
    T = N_runs * M_mm
    task_indices = np.array_split(np.arange(T), size)[rank]

    # ------------- buffers / counters ----------
    cols = ["run_number", "object_id", "spacecraft_number",
            "values", "total_length", "spacecraft_1_ini_pos"]
    df_buffer   = pd.DataFrame(columns=cols)
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
        # Map linear index -> (run_no, mm_idx)
        run_idx, mm_idx = divmod(int(t), M_mm)
        run_no = run_idx + 1

        # Reproducible RNG per (run_no, mm_idx)
        ss = np.random.SeedSequence([base_seed, run_no, mm_idx])
        # If your downstream code uses global np.random, you can seed it:
        np.random.seed(ss.generate_state(1)[0] & 0xFFFFFFFF)

        # ======= do the work for ONE (run, minimoon) =======
        master_i = minimoon_master.iloc[mm_idx]
        current_minimoon = Asteroid(
            master_i['Object id'],
            master_i['Min_SunEarthL1_V_index'],
            config
        )
        formation = Formation(config)

        asteroid_pos = current_minimoon.orbit.loc[:, ['Synodic x', 'Synodic y', 'Synodic z']].values
        earth_pos    = np.zeros_like(asteroid_pos)
        moon_pos     = current_minimoon.orbit.loc[:, ['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']].values

        formation.match_spacecraft_trajectory(len(asteroid_pos[:, 0]), config)

        new_rows = []
        for jdx, spacecraft in enumerate(formation.spacecraft):
            sc_pos   = spacecraft.matched_trajectory
            visible  = spacecraft.asteroid_in_fov_batch(asteroid_pos, sc_pos, earth_pos, moon_pos, config)
            visible  = np.asarray(visible)
            len_vis  = len(visible)
            visible  = visible[visible >= 0]

            new_rows.append({
                "run_number": run_no,
                "object_id": current_minimoon.id,
                "spacecraft_number": jdx + 1,
                "values": tuple(visible),
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


def run_sim_runnumbers_MPI_getIOD(config):
    """
    MPI stage that reads detection 'visible' files from:
        visible_files_folder/spacecraft_{num_spacecraft}
    and produces IOD files into:
        IOD_folder_path/spacecraft_{num_spacecraft}
    Also appends a single MASTER_IOD.csv (resumable, duplicate-safe).
    """
    # assumes your modules are already imported somewhere above:
    # util, nbody, sp (SPICE), etc.

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ---------- paths / config ----------
    num_sc   = int(config['num_spacecraft'])
    vis_root = os.path.abspath(config['visible_files_folder'])
    vis_dir  = os.path.join(vis_root, f"spacecraft_{num_sc}")           # INPUTS
    iod_root = os.path.abspath(config['IOD_folder_path'])
    iod_dir  = os.path.join(iod_root, f"spacecraft_{num_sc}")           # OUTPUTS
    save_format = config.get('save_format', 'csv')  # 'csv' | 'parquet' | 'both'

    # Master CSV + per-row done markers (for resumability)
    master_name = "MASTER_IOD.csv"
    master_path = os.path.join(iod_dir, master_name)
    master_done_dir = os.path.join(iod_dir, "master_rows")

    if rank == 0:
        os.makedirs(iod_dir, exist_ok=True)
        os.makedirs(master_done_dir, exist_ok=True)
        if not os.path.isdir(vis_dir):
            raise FileNotFoundError(f"Visible files dir not found: {vis_dir}")
    comm.Barrier()

    # ---------- master column schema ----------
    # Dynamic columns for the spacecraft-dependent sections
    helio_sc_cols = [f"HELIO_SC_{i+1}(kms)" for i in range(num_sc)]
    pointing_cols = [f"POINTING_SC_{i+1}"   for i in range(num_sc)]

    master_columns = (
            ["ID_AST",
             "EPOCH_AST(jdtdb)",
             "HELIO_AST(kms)",
             "EARTH_HELIO_EA(kms)",
             "EPOCH_SC(jdtdb)",
             "DETECTING_SC_ID"]
            + helio_sc_cols
            + ["EARTH_HELIO_SE(kms)"]
            + pointing_cols
            + ["IOD_DATA_SAVED_AS"]  # <-- NEW
    )

    # Helpers for serialization (CSV-safe and lightweight)
    def serialize_vec(v):
        """1D array-like -> 'v0,v1,...' """
        return ",".join(f"{float(x):.16g}" for x in np.asarray(v).ravel())

    def serialize_mat(M):
        """2D array-like -> 'r0;r1;...' where r = 'c0,c1,...' """
        A = np.asarray(M)
        if A.ndim == 1:
            return serialize_vec(A)
        return ";".join(serialize_vec(row) for row in A)

    def write_master_header_if_needed():
        if not os.path.exists(master_path):
            # create with header
            df_empty = pd.DataFrame(columns=master_columns)
            df_empty.to_csv(master_path, index=False)

    def master_row_done_path(row_uid):
        # row_uid = 'minimoon-{mm_id}_sc-{sc_id}_index-{idx0}_{src_base}'
        return os.path.join(master_done_dir, f"{row_uid}.done")

    def master_row_already_done(row_uid):
        return os.path.exists(master_row_done_path(row_uid))

    def mark_master_row_done(row_uid):
        with open(master_row_done_path(row_uid), "w") as f:
            f.write("ok\n")

    # Ensure header exists
    if rank == 0:
        write_master_header_if_needed()
    comm.Barrier()

    # ---------- list inputs on rank 0, then broadcast ----------
    if rank == 0:
        if hasattr(util, 'get_all_files'):
            all_files = util.get_all_files(vis_dir, save_format)
        else:
            patterns = []
            if save_format in ('csv', 'both'):
                patterns.append(os.path.join(vis_dir, "*.csv"))
            if save_format in ('parquet', 'both'):
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

    all_files   = comm.bcast(all_files, root=0)
    total_files = comm.bcast(total_files, root=0)

    # ---------- helpers ----------
    def base_source_name(path):
        return os.path.splitext(os.path.basename(path))[0]

    def done_marker_path(basename):
        return os.path.join(iod_dir, f".done_{basename}.json")

    def outputs_for_source_exist(basename):
        counts = {}
        if save_format in ('csv', 'both'):
            counts['csv'] = len(glob.glob(os.path.join(iod_dir, f"*_{basename}.csv")))
        if save_format in ('parquet', 'both'):
            counts['parquet'] = len(glob.glob(os.path.join(iod_dir, f"*_{basename}.parquet")))
        return counts

    def row_outputs_exist(base_path):
        csv_exists = os.path.exists(base_path + ".csv")
        pq_exists  = os.path.exists(base_path + ".parquet")
        if save_format == 'csv':
            return csv_exists
        elif save_format == 'parquet':
            return pq_exists
        else:  # both
            return csv_exists and pq_exists

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
            run_data = util.read_master(file_i, config)

            run_data["min_nonnegative"] = run_data["values"].apply(
                lambda x: min(x) if np.any(np.asarray(x) >= 0) else np.nan
            )

            detected_pop = run_data[~np.isnan(run_data["min_nonnegative"])]

            if len(detected_pop) == 0:
                with open(done_marker_path(src_base), "w") as f:
                    json.dump({
                        "source_file": os.path.basename(file_i),
                        "time_utc": dt.datetime.utcnow().isoformat() + "Z",
                        "num_rows_expected": 0,
                        "num_rows_csv": 0,
                        "num_rows_parquet": 0
                    }, f, indent=2)
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

        # -------- per-rank buffer of master rows (list of dicts) --------
        master_rows_buffer = []

        if len(my_chunk) > 0:

            # get spacecraft initial states / boresights aligned to my_chunk
            detected_appended_pop_chunk, all_sc_states, detecting_id, boresights = util.get_scs_initial_states(my_chunk, config)

            # ---------- process my rows ----------
            zdx = 0
            for _, detected_minimoon in detected_appended_pop_chunk.iterrows():

                mm_id = detected_minimoon.name[1]
                sc_id = detected_minimoon.name[2]
                idx0  = int(detected_minimoon['min_nonnegative'])

                file_name = f"minimoon-{mm_id}_sc-{sc_id}_index-{idx0}_{src_base}"
                base_path = os.path.join(iod_dir, file_name)
                row_uid   = file_name  # used for master BEJCT_ID + dedupe

                # If this specific row was already appended to master in a previous run, skip early
                # (we also skip heavy work if per-row outputs already exist)
                if master_row_already_done(row_uid) and row_outputs_exist(base_path):
                    continue

                # ---- heavy computations (kept as in your original flow) ----
                orbit_path = os.path.join(config['minimoon_files_folder'], f"{mm_id}.csv")
                orbit = pd.read_csv(orbit_path, sep=' ', header=0, names=config['minimoon_column_names'])

                asteroid_state_helio = orbit.loc[idx0, ['Helio x','Helio y','Helio z','Helio vx','Helio vy','Helio vz']].values
                asteroid_state_helio[:3] *= (config['AU_TO_M'] / config['KM_TO_M'])
                asteroid_state_helio[3:] *= (config['AU_TO_M'] / config['KM_TO_M'] / config['SECONDS_PER_DAY'])
                asteroid_epoch = orbit.loc[idx0, 'Julian Date']

                num_frames = int(config['number_of_frames'])
                step_days  = config['time_between_frames'] / config['SECONDS_PER_DAY']
                epochs = asteroid_epoch + step_days * np.arange(num_frames)
                total_window_s = num_frames * config['time_between_frames']

                asteroid_integrated_states, asteroid_earth_states = nbody.integrate_n_body(
                    asteroid_state_helio, asteroid_epoch, total_window_s,
                    config['time_between_frames'], type="ASTEROID"
                )
                asteroid_state = util.helio_eclip_to_sun_earth_corotating_batch_full(
                    asteroid_integrated_states, asteroid_earth_states
                )

                sc_geo_eci = detected_minimoon[['GEO_ECLIP_X_(km)','GEO_ECLIP_Y_(km)','GEO_ECLIP_Z_(km)',
                                                'GEO_ECLIP_Vx_(km/s)','GEO_ECLIP_Vy_(km/s)','GEO_ECLIP_Vz_(km/s)']].to_numpy()
                sc_epoch   = detected_minimoon['sc_epoch']

                sun_geo_state = sp.spkgeo(10, sp.str2et(sc_epoch), "ECLIPJ2000", 399)[0]
                sc_helio_ini  = sc_geo_eci - sun_geo_state

                sc_int_states, earth_states = nbody.integrate_n_body(
                    sc_helio_ini, sc_epoch, total_window_s,
                    config['time_between_frames'], type="SPACECRAFT"
                )
                sc_secr = util.helio_eclip_to_sun_earth_corotating_batch_full(sc_int_states, earth_states)

                sc_geo  = util.sun_earth_corotating_to_geo_eclip_batch_full(sc_secr, asteroid_earth_states)
                ast_geo = util.sun_earth_corotating_to_geo_eclip_batch_full(asteroid_state, asteroid_earth_states)

                sc_geo_eme  = util.ecliptic_to_eme_batch(sc_geo)
                ast_geo_eme = util.ecliptic_to_eme_batch(ast_geo)

                x_rel = ast_geo_eme[0, :] - sc_geo_eme[0, :]
                y_rel = ast_geo_eme[1, :] - sc_geo_eme[1, :]
                z_rel = ast_geo_eme[2, :] - sc_geo_eme[2, :]
                r_xy = np.hypot(x_rel, y_rel)
                r    = np.sqrt(r_xy**2 + z_rel**2)
                eps  = 1e-12
                sin_ra  = y_rel / np.maximum(r_xy, eps)
                cos_ra  = x_rel / np.maximum(r_xy, eps)
                sin_dec = z_rel / np.maximum(r, eps)

                sc_secr_ini = sc_secr[:, 0]
                earth_helio_ini = asteroid_earth_states[:, 0]
                sc_helio_ini_state = util.sun_earth_corotating_to_helio_eclip_single(sc_secr_ini, earth_helio_ini)
                sc_helio_states, asteroid_earth_states_2 = nbody.integrate_n_body(
                    sc_helio_ini_state, asteroid_epoch, total_window_s,
                    config['time_between_frames'], type="SPACECRAFT-ASTEROIDTIME"
                )
                sc_eme_states = util.helio_eclip_to_geo_eme_batch(sc_helio_states, asteroid_earth_states)

                x_rel_p = ast_geo_eme[0, :] - sc_eme_states[0, :]
                y_rel_p = ast_geo_eme[1, :] - sc_eme_states[1, :]
                z_rel_p = ast_geo_eme[2, :] - sc_eme_states[2, :]
                r_xy_p  = np.hypot(x_rel_p, y_rel_p)
                r_p     = np.sqrt(r_xy_p**2 + z_rel_p**2)
                sin_ra_p  = y_rel_p / np.maximum(r_xy_p, eps)
                cos_ra_p  = x_rel_p / np.maximum(r_xy_p, eps)
                sin_dec_p = z_rel_p / np.maximum(r_p, eps)

                # Assemble per-row IOD dataframe (unchanged)
                data = np.array([
                    epochs,
                    ast_geo_eme[0,:], ast_geo_eme[1,:], ast_geo_eme[2,:],
                    ast_geo_eme[3,:], ast_geo_eme[4,:], ast_geo_eme[5,:],
                    sc_geo_eme[0,:],  sc_geo_eme[1,:],  sc_geo_eme[2,:],
                    sc_geo_eme[3,:],  sc_geo_eme[4,:],  sc_geo_eme[5,:],
                    sin_ra, cos_ra, sin_dec,
                    sc_eme_states[0,:], sc_eme_states[1,:], sc_eme_states[2,:],
                    sc_eme_states[3,:], sc_eme_states[4,:], sc_eme_states[5,:],
                    sin_ra_p, cos_ra_p, sin_dec_p
                ]).T
                df = pd.DataFrame(data, columns=config['IOD_data_columns_geo_and_phys'])

                # Write per-row outputs (respects save_format)
                if save_format in ('csv', 'both') and not os.path.exists(base_path + ".csv"):
                    df.to_csv(base_path + ".csv", index=False)
                if save_format in ('parquet', 'both') and not os.path.exists(base_path + ".parquet"):
                    df.to_parquet(base_path + ".parquet", index=False)

                saved_files = []
                if os.path.exists(base_path + ".parquet"):
                    saved_files.append(os.path.basename(base_path) + ".parquet")
                if os.path.exists(base_path + ".csv"):
                    saved_files.append(os.path.basename(base_path) + ".csv")
                saved_as_str = ";".join(saved_files) if saved_files else ""

                # -----------------------
                # Build values for MASTER
                # -----------------------
                # final asteroid id
                final_a = mm_id

                # final asteroid epoch (jdtdb) -> last epoch
                final_ae = float(epochs[-1])

                # final asteroid helio state (6)
                final_ha = asteroid_integrated_states[:, -1]

                # final earth helio state (6) at asteroid-time
                final_eha = asteroid_earth_states[:, -1]

                # detecting spacecraft epoch (jdtdb)
                in_et = sp.str2et(sc_epoch)
                in_jdtdb = sp.unitim(in_et, 'ET', 'JDTDB')
                final_se = float(in_jdtdb + total_window_s / config['SECONDS_PER_DAY'])

                # detecting s/c id
                final_did = int(sc_id)

                # all spacecraft initial states for this detection (geo-ecl), size = num_sc x 6
                sc_states = all_sc_states[zdx]  # matrix [num_sc, 6]

                final_hsc_list = []
                for fdx in range(0, len(sc_states)):
                    sc_geo_eci_fdx = sc_states[fdx, :]
                    sc_helio_ini_fdx = sc_geo_eci_fdx - sun_geo_state
                    sc_int_states_fdx, earth_states_fdx = nbody.integrate_n_body(
                        sc_helio_ini_fdx, sc_epoch, total_window_s,
                        config['time_between_frames'], type="SPACECRAFT"
                    )
                    final_hsc_list.append(sc_int_states_fdx[:, -1])

                all_final_hsc = np.vstack(final_hsc_list)   # shape [num_sc, 6]

                # final earth helio (spacecraft-time integration) — use last available
                final_hesc = earth_states_fdx[:, -1]

                # spacecraft boresights for this detection (length num_sc, each vector)
                final_sc_boresights = boresights[zdx]  # iterable length num_sc

                # -----------------------
                # Construct master row
                # -----------------------
                # Skip if this row_uid already marked done (dedupe on resume)
                if not master_row_already_done(row_uid):
                    row_dict = {
                        "ID_AST": final_a,
                        "EPOCH_AST(jdtdb)": f"{final_ae:.16f}",
                        "HELIO_AST(kms)": serialize_vec(final_ha),
                        "EARTH_HELIO_EA(kms)": serialize_vec(final_eha),
                        "EPOCH_SC(jdtdb)": f"{final_se:.16f}",
                        "DETECTING_SC_ID": int(final_did),
                        "EARTH_HELIO_SE(kms)": serialize_vec(final_hesc),
                        "IOD_DATA_SAVED_AS": saved_as_str,  # <-- NEW
                    }

                    # Insert HELIO_SC_i
                    for i in range(num_sc):
                        key = f"HELIO_SC_{i+1}(kms)"
                        row_dict[key] = serialize_vec(all_final_hsc[i, :]) if i < all_final_hsc.shape[0] else ""

                    # Insert POINTING_SC_i
                    for i in range(num_sc):
                        key = f"POINTING_SC_{i+1}"
                        # boresight might be vector or None
                        bs = final_sc_boresights[i] if i < len(final_sc_boresights) else None
                        row_dict[key] = serialize_vec(bs) if bs is not None else ""

                    # Ensure columns order and fill missing keys with ""
                    ordered_row = {col: row_dict.get(col, "") for col in master_columns}
                    master_rows_buffer.append((row_uid, ordered_row))  # keep uid alongside data

                zdx += 1

        # -------- send per-rank buffers to rank 0 for a single append --------
        gathered = comm.gather(master_rows_buffer, root=0)

        if rank == 0:
            # flatten list of (uid, row) pairs
            all_items = [item for chunk in gathered for item in chunk]  # [(uid, row_dict), ...]
            if all_items:
                # filter out any already-done uids
                filtered_items = [(uid, row) for (uid, row) in all_items if not master_row_already_done(uid)]
                if filtered_items:
                    write_master_header_if_needed()
                    rows_only = [row for (_, row) in filtered_items]
                    df_master_append = pd.DataFrame(rows_only, columns=master_columns)
                    df_master_append = df_master_append[master_columns]
                    df_master_append.to_csv(master_path, mode="a", header=False, index=False)
                    # mark each appended uid as done
                    for (uid, _) in filtered_items:
                        mark_master_row_done(uid)

        # All ranks finished their rows for this file
        comm.Barrier()

        # Rank 0: check completeness, write DONE marker if complete
        if rank == 0:
            counts = outputs_for_source_exist(src_base)
            ok_csv = (save_format in ('csv','both'))     and (counts.get('csv', 0)     >= (expected_rows or 0))
            ok_pq  = (save_format in ('parquet','both')) and (counts.get('parquet', 0) >= (expected_rows or 0))
            complete = (
                (save_format == 'csv'     and ok_csv) or
                (save_format == 'parquet' and ok_pq)  or
                (save_format == 'both'    and ok_csv and ok_pq) or
                (expected_rows == 0)
            )
            if complete:
                with open(done_marker_path(src_base), "w") as f:
                    json.dump({
                        "source_file": os.path.basename(file_i),
                        "time_utc": dt.datetime.utcnow().isoformat() + "Z",
                        "num_rows_expected": int(expected_rows or 0),
                        "num_rows_csv": int(counts.get('csv', 0)),
                        "num_rows_parquet": int(counts.get('parquet', 0))
                    }, f, indent=2)
                print(f"[IOD] DONE: {os.path.basename(file_i)} "
                      f"(expected {expected_rows}, csv={counts.get('csv',0)}, pq={counts.get('parquet',0)})",
                      flush=True)
            else:
                print(f"[IOD] PARTIAL (no marker): {os.path.basename(file_i)} "
                      f"(expected {expected_rows}, csv={counts.get('csv',0)}, pq={counts.get('parquet',0)})",
                      flush=True)

        comm.Barrier()

    return


def run_IOD(config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("Starting IOD...")

    viz_flag = bool(config.get('visualization_flag', 0))


    n_runs = int(config['n_runs_test'])

    # --- Helpers ----------------------------------------------------------------
    def task_stream():
        """Yield (combo, run_idx) without ever building a giant list."""
        idx = 0
        for r in range(n_runs):
            if idx % size == rank:
                yield combo, r
            idx += 1

    # Unique output names
    os.makedirs(config['error_file_dir'], exist_ok=True)
    meta_basename = f"{config['dynamics']}_{config['orbit']}_{config['observer']}_{config['optimizer']}"
    final_meta_path = os.path.join(config['error_file_dir'], meta_basename + "_meta_data.csv")
    rank_meta_path = os.path.join(config['error_file_dir'], f"{meta_basename}_rank{rank}.csv")

    # --- Determine already-completed keys (so we can resume) --------------------
    def key_of(row):
        """Return the canonical numeric key tuple for a row or CSV dict row."""
        return (
            int(row.get("RUN_NUMBER", 0) or 0),
        )

    def read_completed_keys(paths):
        done = set()
        for p in paths:
            if not os.path.exists(p):
                continue
            try:
                with open(p, "r", newline="") as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        done.add(key_of(row))
            except Exception:
                # ignore partially written/empty files
                pass
        return done

    # include both the global merged file and this rank's partial file
    already_done = read_completed_keys([final_meta_path, rank_meta_path])

    # Prepare writer (append mode)
    rank_header = [
        # --- all parameters ---
        "NUMBER_OF_OBSERVATIONS",
        "TIME_DELTA_DAYS",  # numeric version of TIME_DELTA
        "TOTAL_POINTS",
        "SAMPLING_METHOD",
        "LAYER_RATIOS",  # JSON string
        "INPUT_RANGE",  # JSON string
        "HIDDEN_DIMENSION",
        "PHYSICS_WEIGHT",
        "LAMBDA_DIST",
        "WEIGHT_SCALE_FACTOR",
        "NUMBER_OF_ITERATIONS",
        "TEMPERATURE",
        "X_TOLERANCE",
        "F_TOLERANCE",
        "MAX_FUNCTION_EVAL",
        "MAX_ITERATiONS",  # keep your exact key spelling
        "G_TOLERANCE",
        "RUN_NUMBER",
        "MIN_RHO",
        "MAX_RHO",
        "MIN_RHO_DOT",
        "MAX_RHO_DOT",
        "DELTA_RHO",
        "DELTA_RHO_DOT",

        # --- identifiers ---
        "RUN_NUMBER",

        # --- metrics/extras ---
        "POS_RMSE",
        "VEL_RMSE",
        "COMPUTATION_TIME_SEC",
        "OPTIMAL_BH_ITERATION",
        "FILE_USED",
        "SAVED_AS",
    ]

    rank_file_exists = os.path.exists(rank_meta_path)
    need_header = (not rank_file_exists) or (os.path.getsize(rank_meta_path) == 0)

    rank_fh = open(rank_meta_path, "a", newline="")
    rank_writer = csv.DictWriter(rank_fh, fieldnames=rank_header)
    if need_header:
        rank_writer.writeheader()

    def task_key(combo, run_idx):
        """Return the numeric key matching read_completed_keys() shape."""
        dr = float(combo[0])
        drd = float(combo[1]) if len(combo) > 1 else 0.0
        t = float(combo[2]) if len(combo) > 2 else 0.0
        lr = float(combo[3]) if len(combo) > 3 else 0.0
        dtf = float(combo[4]) if len(combo) > 4 else 0.0
        return (dr, drd, t, lr, dtf, int(run_idx))

    # --- Main streamed work loop -----------------------------------------------
    for (combo, run_idx) in task_stream():
        # skip if already completed
        if task_key(combo, run_idx) in already_done:
            continue

        if run_idx == n_runs - 1:
            print(f"[Rank {rank}] Running hyperparameter combo {combo}, run={run_idx + 1}", flush=True)

        # --------- CASE SWITCH (unchanged logic, just use `combo`) --------------
        dynamics, orbit, observer, optimizer = config['dynamics'], config['orbit'], config['observer'], config['optimizer']

        # You can keep your original blocks; only show one as example for brevity:
        if dynamics == 'NBD' and observer == 'SPACE' and optimizer == 'CONSTRAINED_BASIN_HOPPING':
            import PIELM_basinhopping_w_range_nbody as pielm_ctsn

            m_2 = combo[3]
            m_12 = (1 - m_2) / 2
            parameters = {
                'NUMBER_OF_OBSERVATIONS': 16,
                'TIME_DELTA': combo[4] * u.day,
                'TOTAL_POINTS': 100,
                'SAMPLING_METHOD': "uniform",
                'LAYER_RATIOS': [(0., m_12), (m_12, m_12 + m_2), (m_12 + m_2, 1.)],
                'INPUT_RANGE': (-1, 1),
                'HIDDEN_DIMENSION': 100,
                'PHYSICS_WEIGHT': 1e2,
                'LAMBDA_DIST': 1e-3,
                'WEIGHT_SCALE_FACTOR': 1e-2,
                'NUMBER_OF_ITERATIONS': 2,
                'TEMPERATURE': combo[2],
                'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15,
                'MAX_FUNCTION_EVAL': 1000, 'MAX_ITERATiONS': 1000,
                'G_TOLERANCE': 1e-15, 'RUN_NUMBER': run_idx,
                'MIN_RHO': 0.0006684587122, 'MAX_RHO': 0.06684587122,
                'MIN_RHO_DOT': -0.05033557046, 'MAX_RHO_DOT': 0.05033557046,
                'DELTA_RHO': combo[0], 'DELTA_RHO_DOT': combo[1]
            }
            if viz_flag:
                config['lambda'] = parameters['TIME_DELTA']
                config['run_idx'] = run_idx
            data = pielm_ctsn.generate_data(config, parameters)
            results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time, optimal_bh = pielm_ctsn.run(data, config, parameters)

        # (Keep all your other elif blocks exactly as you have them)
        # ------------------------------------------------------------------------

        # Build output filename for the per-run IOD CSV (unchanged)
        def combo_to_string_local(c): return "_" + "_".join(str(x) for x in c)
        combo_str = combo_to_string_local(combo)
        file_used = data[9]
        file_name = f"{dynamics}_{orbit}_{observer}_{optimizer}_run_{run_idx}{combo_str}.csv"

        os.makedirs(config['error_file_dir'], exist_ok=True)
        file_path = os.path.join(config['error_file_dir'], file_name)

        # Generate IOD file and RMSEs
        rmse_df = util.generate_iod_file(file_path, final_pos, final_vel, true_pos, true_vel, epochs)

        pos_rmse = float(np.sqrt(((rmse_df[["IOD_X","IOD_Y","IOD_Z"]].values - rmse_df[["TRUE_X","TRUE_Y","TRUE_Z"]].values) ** 2).mean()))
        vel_rmse = float(np.sqrt(((rmse_df[["IOD_VX","IOD_VY","IOD_VZ"]].values - rmse_df[["TRUE_VX","TRUE_VY","TRUE_VZ"]].values) ** 2).mean()))

        def params_to_row(parameters, run_idx, extras):
            """
            Build a flat row dict for CSV:
            - includes ALL parameters (normalized)
            - adds metrics/extras (pos_rmse, etc.)
            """
            row = {}

            for k, v in parameters.items():
                # Normalize astropy quantities
                if isinstance(v, u.Quantity):
                    # choose canonical units per key
                    if k == "TIME_DELTA":
                        row["TIME_DELTA_DAYS"] = v.to(u.day).value  # numeric float
                    else:
                        # generic: store value in SI if you prefer
                        row[k] = v.to_base_units().value
                    continue

                # numpy scalars
                if isinstance(v, (np.floating, np.integer)):
                    row[k] = v.item()
                    continue

                # plain scalars
                if isinstance(v, (int, float)):
                    row[k] = v
                    continue

                # arrays / sequences -> JSON
                if isinstance(v, (list, tuple, np.ndarray)):
                    if k == "LAYER_RATIOS":
                        row[k] = v[1][1] - v[1][0]
                    else:
                        row[k] = json.dumps(v if not isinstance(v, np.ndarray) else v.tolist())
                    continue

                # everything else as string
                row[k] = str(v)

            # Ensure RUN_NUMBER and RUN_IDX exist and are ints
            row["RUN_NUMBER"] = int(parameters.get("RUN_NUMBER", run_idx))

            # Attach extras/metrics (already numeric/strings)
            row.update(extras)
            return row

        def build_extras(pos_rmse, vel_rmse, comp_time, optimal_bh, file_used, file_name):
            return {
                "POS_RMSE": float(pos_rmse),
                "VEL_RMSE": float(vel_rmse),
                "COMPUTATION_TIME_SEC": float(comp_time),
                "OPTIMAL_BH_ITERATION": float(optimal_bh),
                "FILE_USED": str(file_used),
                "SAVED_AS": str(file_name),
            }

        extras = build_extras(pos_rmse, vel_rmse, comp_time, optimal_bh, file_used, file_name)
        row = params_to_row(parameters, run_idx, extras)

        # Write immediately
        rank_writer.writerow(row)
        rank_fh.flush()
        os.fsync(rank_fh.fileno())

        # Optional: visualize only in dev runs
        if viz_flag:
            # ... your existing viz code here (unchanged) ...
            pass

        # Free large objects deterministically
        del results, positions, velocities, nlls_start
        del final_pos, final_vel, true_pos, true_vel, epochs, rmse_df, data
        gc.collect()

    # Close our per-rank file
    rank_fh.close()

    # --------- Merge (rank 0) ---------------------------------------------------
    comm.Barrier()
    if rank == 0:
        # Find all rank csvs + any previous global file; merge & de-dup on key
        rank_files = glob.glob(os.path.join(config['error_file_dir'], f"{meta_basename}_rank*.csv"))

        rows, seen = [], set()

        sources = ([final_meta_path] if os.path.exists(final_meta_path) else []) + rank_files

        for src in sources:
            try:
                with open(src, "r", newline="") as f:
                    rdr = csv.DictReader(f)

                    for r in rdr:
                        k = key_of(r)
                        if k not in seen:
                            rows.append(r)
                            seen.add(k)
            except Exception:
                pass

        # Write merged final file
        with open(final_meta_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rank_header)
            w.writeheader()
            w.writerows(rows)

    comm.Barrier()
    return


def _visible_dir(config):
    num_sc = int(config['num_spacecraft'])
    root   = os.path.abspath(config['visible_files_folder'])
    return os.path.join(root, f"spacecraft_{num_sc}")


def _iod_dir(config):
    num_sc = int(config['num_spacecraft'])
    root   = os.path.abspath(config['IOD_folder_path'])
    return os.path.join(root, f"spacecraft_{num_sc}")


def _non_hidden_entries(path):
    return [e for e in os.scandir(path) if not e.name.startswith('.')]


def _source_basenames_in_visible(vis_dir, save_format):
    # Prefer your util if available; otherwise glob by format(s)
    try:
        files = util.get_all_files(vis_dir, save_format)
    except Exception:
        import glob
        files = []
        if save_format in ('csv', 'both'):
            files += glob.glob(os.path.join(vis_dir, '*.csv'))
        if save_format in ('parquet', 'both'):
            files += glob.glob(os.path.join(vis_dir, '*.parquet'))
    files = sorted(files)
    bases = [os.path.splitext(os.path.basename(p))[0] for p in files]
    return bases


def run_overall_OD(master, config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Starting...")

    # -------------------------
    # Stage 1: Detection (skip if visible/spacecraft_X not empty)
    # -------------------------
    vis_dir = _visible_dir(config)
    if rank == 0:
        os.makedirs(vis_dir, exist_ok=True)
        do_detection = (len(_non_hidden_entries(vis_dir)) == 0)
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
    iod_dir = _iod_dir(config)
    save_format = config.get('save_format', 'csv')

    if rank == 0:
        os.makedirs(iod_dir, exist_ok=True)

        bases = _source_basenames_in_visible(vis_dir, save_format)
        # If no detection inputs exist, there is nothing to do here.
        if len(bases) == 0:
            do_iod = False
            msg = f"[Stage: IOD] No detection inputs found in {vis_dir} → skip"
        else:
            # A source is considered fully processed if marker exists
            done_markers = [os.path.join(iod_dir, f".done_{b}.json") for b in bases]
            done_flags   = [os.path.exists(m) for m in done_markers]
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

    #-----------------------
    # Stage 3: Running IOD
    #-----------------------

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
master = util.parse_master_new_new_new(config['minimoon_master_file_path'])

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