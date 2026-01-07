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
    rows_per_part = int(config.get('number_of_rows_per_part', 50000))
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

    viz_flag = bool(config.get('visualization_flag', 0))

    # Paths
    iod_dir   = util._iod_dir(config)
    master_fn = os.path.join(iod_dir, "MASTER_IOD.csv")
    if rank == 0 and not os.path.exists(master_fn):
        print(f"[Stage: IOD Solve] No MASTER_IOD.csv at {master_fn} → skip")
    comm.Barrier()
    if not os.path.exists(master_fn):
        return

    # Rank 0: inspect master; broadcast row count / columns
    if rank == 0:
        try:
            _head = pd.read_csv(master_fn, nrows=1)
            n_rows = sum(1 for _ in open(master_fn, "r", encoding="utf-8")) - 1
            cols   = list(_head.columns)
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
        s = str(saved_as_str or "")
        if not s.strip():
            return None
        first = s.split(";")[0].strip()
        if not first:
            return None
        return os.path.splitext(os.path.basename(first))[0]

    def done_marker_path(uid: str, row_index: int):
        # Use a stable UID; fall back to the row index to avoid collisions
        safe_uid = uid if (uid and str(uid).strip()) else f"rowidx_{row_index}"
        return os.path.join(stage3_done_dir, f"{safe_uid}.done")

    def is_done(uid: str, row_index: int) -> bool:
        return os.path.exists(done_marker_path(uid, row_index))

    # Load MASTER once for reading (workers)
    df_master = pd.read_csv(master_fn)

    # Column order spec (final write order)
    detection_block = [
        "ID_AST","EPOCH_AST(jdtdb)","HELIO_AST(kms)","EARTH_HELIO_EA(kms)",
        "EPOCH_SC(jdtdb)","DETECTING_SC_ID",
        "HELIO_SC_1(kms)","HELIO_SC_2(kms)","HELIO_SC_3(kms)","HELIO_SC_4(kms)",
        "EARTH_HELIO_SE(kms)",
        "POINTING_SC_1","POINTING_SC_2","POINTING_SC_3","POINTING_SC_4",
    ]
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
        "POS_RMSE","VEL_RMSE","COMPUTATION_TIME_SEC","OPTIMAL_BH_ITERATION",
        "IOD_RESULT_SAVED_AS", "IOD_FINAL_STATE"
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
        safe_uid = master_uid if (master_uid and str(master_uid).strip()) else f"rowidx_{m_idx}"

        # resume-skip (per row)
        if is_done(master_uid, m_idx):
            skipped += 1
            continue

        try:

            # ------------------- YOUR IOD PIPELINE (single run) -------------------
            dynamics, orbit, observer, optimizer = (
                config['dynamics'], config['orbit'], config['observer'], config['optimizer']
            )

            if dynamics == 'NBD' and observer == 'SPACE' and optimizer == 'CONSTRAINED_BASIN_HOPPING':
                import PIELM_basinhopping_w_range_nbody as pielm_ctsn

                # Fixed parameters you provided
                m_2  = 0.1
                m_12 = (1.0 - m_2) / 2.0

                parameters = {
                    'NUMBER_OF_OBSERVATIONS': 16,
                    'TIME_DELTA': 0.6 * u.day,
                    'TOTAL_POINTS': 250,
                    'SAMPLING_METHOD': "uniform",
                    'LAYER_RATIOS': [(0., m_12), (m_12, m_12 + m_2), (m_12 + m_2, 1.)],
                    'INPUT_RANGE': (-1, 1),
                    'HIDDEN_DIMENSION': 250,
                    'PHYSICS_WEIGHT': 1e3,
                    'LAMBDA_DIST': 1e-2,
                    'WEIGHT_SCALE_FACTOR': 1e-2,
                    'NUMBER_OF_ITERATIONS': 20,
                    'TEMPERATURE': 1e8,
                    'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15,
                    'MAX_FUNCTION_EVAL': 1000,
                    'MAX_ITERATiONS': 1000,
                    'G_TOLERANCE': 1e-15,
                    'MIN_RHO': 0.0006684587122,
                    'MAX_RHO': 0.06684587122,
                    'MIN_RHO_DOT': -0.05033557046,
                    'MAX_RHO_DOT':  0.05033557046,
                    'DELTA_RHO': 6.68459e-8,
                    'DELTA_RHO_DOT': 0.00167785234,
                }

                if viz_flag:
                    config['lambda'] = parameters['TIME_DELTA']

                # Build data from MASTER row (uses new master logic)
                data = pielm_ctsn.generate_data(config, parameters, master_row=row)

                (results, positions, velocities, nlls_start,
                 final_pos, final_vel, true_pos, true_vel,
                 epochs, comp_time, optimal_bh) = pielm_ctsn.run(data, config, parameters)

            else:
                # Add other branches unchanged if needed
                raise NotImplementedError("Add other solver branches as in your code.")

            # ------- Compute metrics & write unique per-row result file -------
            out_dir = config['error_file_dir']
            os.makedirs(out_dir, exist_ok=True)

            # Unique filename per MASTER row (avoid overwrite)
            base_name = f"{safe_uid}__{config['dynamics']}_{config['orbit']}_{config['observer']}_{config['optimizer']}.csv"
            file_name = base_name
            file_path = os.path.join(out_dir, file_name)
            ctr = 1
            while os.path.exists(file_path):
                file_name = f"{safe_uid}__{config['dynamics']}_{config['orbit']}_{config['observer']}_{config['optimizer']}__{ctr}.csv"
                file_path = os.path.join(out_dir, file_name)
                ctr += 1

            rmse_df = util.generate_iod_file(file_path, final_pos, final_vel, true_pos, true_vel, epochs)

            pos_rmse = float(np.sqrt(((rmse_df[["IOD_X","IOD_Y","IOD_Z"]].values - rmse_df[["TRUE_X","TRUE_Y","TRUE_Z"]].values) ** 2).mean()))
            vel_rmse = float(np.sqrt(((rmse_df[["IOD_VX","IOD_VY","IOD_VZ"]].values - rmse_df[["TRUE_VX","TRUE_VY","TRUE_VZ"]].values) ** 2).mean()))

            try:
                final_xyz = np.asarray(final_pos[0][-2, :], dtype=float).reshape(3, )
                final_vxyz = np.asarray(final_vel[0][-2, :], dtype=float).reshape(3, )
                final_state = np.concatenate([final_xyz, final_vxyz]).tolist()
                final_state_str = json.dumps(final_state)

            except Exception:
                final_state_str = ""

            # ---- Flatten parameters to columns (TIME_DELTA → TIME_DELTA_DAYS) ----
            def params_to_update(parameters):
                upd = {}
                for k, v in parameters.items():
                    if isinstance(v, u.Quantity):
                        if k == "TIME_DELTA":
                            upd["TIME_DELTA_DAYS"] = v.to(u.day).value
                        else:
                            upd[k] = v.to_base_units().value
                        continue
                    if isinstance(v, (np.floating, np.integer)):
                        upd[k] = v.item(); continue
                    if isinstance(v, (int, float)):
                        upd[k] = v; continue
                    if isinstance(v, (list, tuple, np.ndarray)):
                        if k == "LAYER_RATIOS":
                            upd[k] = v[1][1] - v[1][0]
                        else:
                            upd[k] = json.dumps(v if not isinstance(v, np.ndarray) else v.tolist())
                        continue
                    upd[k] = str(v)
                return upd

            upd = {
                "_row_index": int(m_idx),                 # internal key for placement
                # (MASTER_* kept internal, not written)
                "POS_RMSE": float(pos_rmse),
                "VEL_RMSE": float(vel_rmse),
                "COMPUTATION_TIME_SEC": float(comp_time),
                "OPTIMAL_BH_ITERATION": float(optimal_bh),
                "IOD_RESULT_SAVED_AS": str(file_name),
                "IOD_FINAL_STATE": final_state_str,
            }
            upd.update(params_to_update(parameters))

            updates.append(upd)
            processed += 1

            if viz_flag:
                pass

            del (results, positions, velocities, nlls_start,
                 final_pos, final_vel, true_pos, true_vel, epochs, rmse_df, data)
            gc.collect()

        except Exception:
            # Do not mark .done here; committing happens only after master write
            errors += 1
            continue

    # ===== Gather updates to rank 0, write MASTER in-place with order, then commit markers =====
    gathered = comm.gather(updates, root=0)
    committed_uids = None  # list to broadcast (uids derived again to write markers)

    if rank == 0:
        all_updates = [f for chunk in gathered for f in chunk]

        if all_updates:
            # Reload master to minimize race
            df = pd.read_csv(master_fn)

            # Ensure columns exist (avoid adding internal skip keys)
            new_cols = set().union(*(set(d.keys()) for d in all_updates)) - set(internal_skip_keys)
            missing = [c for c in new_cols if c not in df.columns]
            for c in missing:
                # Guess dtype: strings for these, else NaN numeric
                if c in ("SAMPLING_METHOD","INPUT_RANGE","IOD_RESULT_SAVED_AS"):
                    df[c] = ""
                else:
                    df[c] = np.nan

            # Apply updates row-by-row
            for upd in all_updates:
                ri = upd["_row_index"]
                for k, v in upd.items():
                    if k in internal_skip_keys:
                        continue
                    df.at[ri, k] = v

            # Column reordering:
            current_cols = list(df.columns)

            # Start with detection block (keep only those that exist)
            ordered = [c for c in detection_block if c in current_cols]

            # Then IOD_DATA_SAVED_AS if present
            ordered += [c for c in saved_as_col if c in current_cols]

            # Then parameter block (in given order)
            ordered += [c for c in param_block if c in current_cols]

            # Then metrics block
            ordered += [c for c in metrics_block if c in current_cols]

            # Finally, any leftover columns not in the ordered list
            leftovers = [c for c in current_cols if c not in set(ordered)]
            # Explicitly drop internal columns if they somehow exist
            leftovers = [c for c in leftovers if c not in internal_skip_keys and c not in ("MASTER_UID","MASTER_ROW_INDEX","FILE_USED")]

            final_cols = ordered + leftovers
            df = df[final_cols]

            # Atomic write-back
            tmp = master_fn + ".tmp"
            df.to_csv(tmp, index=False)
            os.replace(tmp, master_fn)

            # After successful write, collect committed UIDs for marker creation
            # (derive from MASTER rows themselves)
            committed_uids = []
            for upd in all_updates:
                ri = upd["_row_index"]
                saved_as_val = df.loc[ri, "IOD_DATA_SAVED_AS"] if "IOD_DATA_SAVED_AS" in df.columns else ""
                uid = row_uid_from_saved_as(saved_as_val)
                safe_uid = uid if (uid and str(uid).strip()) else f"rowidx_{ri}"
                committed_uids.append(safe_uid)
        else:
            committed_uids = []

    # Broadcast committed list; workers create .done files now (commit-after-write)
    committed_uids = comm.bcast(committed_uids, root=0)
    for uid in committed_uids:
        try:
            marker_path = os.path.join(stage3_done_dir, f"{uid}.done")
            with open(marker_path, "w") as f:
                json.dump({"uid": uid, "status": "ok"}, f)
        except Exception:
            pass

    if rank == 0:
        print(f"[Stage: IOD Solve] updated_rows={len(committed_uids)}, processed={processed}, skipped={skipped}, errors={errors}")

    comm.Barrier()
    return


def run_OD(config):
    """
    Stage 4: Orbit Determination (OD)
    - Round-robin split of MASTER rows across MPI ranks
    - For each row: while-loop over time until end condition
    - Initialize from IOD outputs on the first step
    - Append per-step results to a unique OD log file
    - After finishing a row, rank 0 updates MASTER with OD summary and log path,
      then commits per-row .done markers (commit-after-write)

    Expected config keys (with defaults if absent):
      - error_file_dir: directory for OD logs (same place you store IOD result CSVs)
      - od_duration_days: duration of the OD simulation window (default: 1.0 day)
      - od_step_seconds: time step in seconds (default: 600)
      - od_max_steps: optional hard cap on steps (default: None)
      - visualization_flag: (0/1) to allow you to toggle plotting/extra dumps
    """

    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Paths & MASTER
    iod_dir   = util._iod_dir(config)
    master_fn = os.path.join(iod_dir, "MASTER_IOD.csv")
    if rank == 0 and not os.path.exists(master_fn):
        print(f"[Stage: OD] No MASTER_IOD.csv at {master_fn} → skip")
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

    # Helpers
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

    # Load MASTER (workers)
    df_master = pd.read_csv(master_fn)

    # OD config
    out_dir = config.get('od_file_dir', os.path.join(iod_dir, "od_outputs"))
    os.makedirs(out_dir, exist_ok=True)

    od_duration_days = float(config.get('od_duration_days', 1.0))
    od_max_steps     = config.get('od_max_steps', None)
    od_max_steps     = int(od_max_steps) if (od_max_steps is not None) else None

    # Columns we’ll add/update in MASTER (we’ll enforce a nice order on write)
    od_metrics_cols = [
        "OD_RESULT_SAVED_AS",      # per-row OD time-series log filename
        "OD_FINAL_TIME_JDTDB",     # final epoch reached (jdtdb)
        "OD_N_STEPS",              # number of OD steps performed
        "OD_LAST_POS_RMSE",        # last-step position RMSE [km] or your chosen units
        "OD_LAST_VEL_RMSE",        # last-step velocity RMSE [km/s]
    ]

    # Collect per-rank updates to MASTER rows
    updates = []
    processed = skipped = errors = 0

    for m_idx in my_indices:
        row = df_master.iloc[m_idx]

        # Derive a stable UID for this master row (same scheme as Stage 3)
        saved_as_str = str(row.get("IOD_DATA_SAVED_AS", "") or "")
        uid = uid_from_saved_as(saved_as_str, m_idx)

        # Resume: skip if already fully done
        if is_done(uid):
            skipped += 1
            continue

        try:
            # --------------------------
            # Initialization (first step)
            # --------------------------
            # Epochs
            try:
                # Prefer SC epoch if present; else asteroid epoch
                t0_jdtdb = float(row.get("EPOCH_AST(jdtdb)", np.nan))
            except Exception:
                t0_jdtdb = np.nan
            if not np.isfinite(t0_jdtdb):
                raise RuntimeError("Cannot determine initial epoch (EPOCH_AST missing).")

            # End time and step size
            t_end   = t0_jdtdb + od_duration_days

            # Load IOD result to initialize OD (from Stage 3)
            iod_result_name = str(row.get("IOD_RESULT_SAVED_AS", "") or "")
            if not iod_result_name.strip():
                # If user wants OD to initialize from raw IOD_DATA_SAVED_AS (time series) instead,
                # you can fallback. We keep it strict here:
                raise FileNotFoundError("IOD_RESULT_SAVED_AS missing in MASTER row; cannot initialize OD.")
            iod_result_path = os.path.join(out_dir, iod_result_name)
            if not os.path.exists(iod_result_path):
                # If files are elsewhere, adjust to your layout.
                # Alternatively store absolute paths in IOD_RESULT_SAVED_AS.
                raise FileNotFoundError(f"IOD result file not found: {iod_result_path}")

            # TODO: load your IOD outputs for initialization
            # Example (replace with your actual loader):
            # iod_init = util.load_iod_result(iod_result_path)
            # x0_est, P0_est = iod_init['x_est'], iod_init['P_est']  # state & covariance (example)
            # x_true0       = iod_init.get('x_true', None)          # if available
            # For now, use placeholders:
            x0_est  = None    # TODO: replace with real estimate
            P0_est  = None    # TODO: replace with real covariance
            x_true0 = None    # TODO: replace with truth if available

            # Prepare per-row OD log (unique filename, no overwrite)
            base = f"{uid}__OD_{config['dynamics']}_{config['orbit']}_{config['observer']}_{config['optimizer']}"
            od_log_path, od_log_name = make_unique_filename(out_dir, base, ".csv")

            # Open OD log and write header
            with open(od_log_path, "w", newline="") as f_log:
                log_writer = csv.writer(f_log)
                # Minimal suggested columns; extend as needed for your analysis
                log_writer.writerow([
                    "STEP_INDEX",
                    "EPOCH_JDTDB",
                    # estimated state vector (flatten as comma-separated strings if needed)
                    "X_EST",             # stringified state estimate
                    "P_EST_TRACE",       # scalar or compact representation
                    # true state, if available
                    "X_TRUE",
                    # control/attitude targets, if applicable
                    "ATTITUDE_CMD",
                    # errors/metrics
                    "POS_RMSE",
                    "VEL_RMSE",
                ])

                # ----------------------------------------------------------
                # Time loop
                # ----------------------------------------------------------
                step_idx = 0
                t_cur = t0_jdtdb

                # Working state (initialize from IOD)
                x_est = x0_est
                P_est = P0_est
                x_true = x_true0

                last_pos_rmse = np.nan
                last_vel_rmse = np.nan

                while True:
                    # End condition
                    if t_cur > t_end:
                        break
                    if od_max_steps is not None and step_idx >= od_max_steps:
                        break

                    if step_idx == 0:
                        # -------------- INITIALIZATION STEP ---------------
                        # TODO: any one-time initialization for your OD filter (e.g., set process noise, etc.)
                        # Example:
                        # od_state = od.init_filter(x0_est, P0_est, config)
                        # (We keep using x_est, P_est variables directly here.)
                        pass
                    else:
                        # -------------- REGULAR OD STEP -------------------
                        # 1) Generate measurements at time t_cur for this master row
                        #    (You can use MASTER columns and/or per-row IOD_DATA_SAVED_AS to drive geometry)
                        # TODO: implement your measurement generation:
                        # meas = util.generate_od_measurements(row, t_cur, config)
                        meas = None

                        # 2) Propagate the filter to t_cur + dt and update with measurements
                        # TODO: implement your OD step:
                        # x_est, P_est = od.run_step(x_est, P_est, meas, config, t_cur, dt=dt_day)
                        # Optionally also maintain a truth model for diagnostics:
                        # x_true = truth.propagate(x_true, dt_day, config) if x_true is not None else None
                        pass

                    # -------------- Post-step metrics & logging ----------
                    # TODO: compute RMSE (pos/vel) at this step if truth is available and uncertainty
                    # Example placeholders:
                    # last_pos_rmse, last_vel_rmse = util.compute_rmse(x_est, x_true)
                    # If not available, keep NaN or compute innovation-based proxies.
                    # For now, they remain as is.

                    # -------------- Attitude / Slew Planning (optional) --
                    # TODO: plan pointing/attitude for next step, if needed:
                    # att_cmd = util.plan_attitude(x_est, row, t_cur, config)
                    att_cmd = ""

                    # TODO: get time update
                    dt_day = 0.01

                    # TODO: get state of system after slew
                    # asteroid state, both true and predicted
                    # formation s/c states

                    # TODO: update state of the system


                    # -------------- Log the step -------------------------
                    # Stringify vectors/matrices compactly to keep CSV readable:
                    def _vec_to_str(v):
                        if v is None:
                            return ""
                        try:
                            arr = np.asarray(v).ravel()
                            return ",".join(f"{float(x):.9g}" for x in arr)
                        except Exception:
                            return str(v)

                    def _mat_trace(m):
                        if m is None:
                            return np.nan
                        try:
                            a = np.asarray(m)
                            return float(np.trace(a))
                        except Exception:
                            return np.nan

                    log_writer.writerow([
                        step_idx,
                        f"{t_cur:.9f}",
                        _vec_to_str(x_est),
                        _mat_trace(P_est),
                        _vec_to_str(x_true),
                        att_cmd,
                        f"{last_pos_rmse:.9g}" if np.isfinite(last_pos_rmse) else "",
                        f"{last_vel_rmse:.9g}" if np.isfinite(last_vel_rmse) else "",
                    ])

                    # -------------- Update time/state for next loop ------
                    t_cur += dt_day
                    step_idx += 1

            # At this point, the OD row is complete; prepare MASTER update
            upd = {
                "_row_index": int(m_idx),
                "OD_RESULT_SAVED_AS": od_log_name,
                "OD_FINAL_TIME_JDTDB": float(t_cur - dt_day),  # last time step we wrote
                "OD_N_STEPS": int(step_idx),
                "OD_LAST_POS_RMSE": float(last_pos_rmse) if np.isfinite(last_pos_rmse) else np.nan,
                "OD_LAST_VEL_RMSE": float(last_vel_rmse) if np.isfinite(last_vel_rmse) else np.nan,
            }
            updates.append(upd)
            processed += 1

            # tidy
            gc.collect()

        except Exception as e:
            # Don’t mark done here; only after MASTER commit
            errors += 1
            # Optional: you could write a per-row error note:
            # with open(os.path.join(od_done_dir, f"{uid}.err"), "w") as fe:
            #     fe.write(str(e))
            continue

    # ===== Gather updates → rank 0 writes MASTER (ordered) → broadcast committed UIDs → write .done =====
    gathered = comm.gather(updates, root=0)
    committed_uids = None

    if rank == 0:
        all_updates = [u for chunk in gathered for u in chunk]

        if all_updates:
            # Reload MASTER to minimize race
            df = pd.read_csv(master_fn)

            # Make sure OD columns exist (add missing)
            for c in od_metrics_cols:
                if c not in df.columns:
                    # choose dtype: strings for *_SAVED_AS, else numeric
                    if c.endswith("_SAVED_AS"):
                        df[c] = ""
                    else:
                        df[c] = np.nan

            # Apply updates
            for upd in all_updates:
                ri = upd["_row_index"]
                for k, v in upd.items():
                    if k == "_row_index":
                        continue
                    df.at[ri, k] = v

            # Optional: reorder columns — keep existing order but put OD columns at the end,
            # or place them after your metrics. If you want a strict order, add it here.
            # We’ll append OD columns after whatever currently exists and isn’t OD:
            existing_cols = list(df.columns)
            non_od = [c for c in existing_cols if c not in od_metrics_cols]
            final_cols = non_od + [c for c in od_metrics_cols if c in df.columns]
            df = df[final_cols]

            # Atomic write-back
            tmp = master_fn + ".tmp"
            df.to_csv(tmp, index=False)
            os.replace(tmp, master_fn)

            # Build UIDs of committed rows (from MASTER rows themselves)
            committed_uids = []
            for upd in all_updates:
                ri = upd["_row_index"]
                saved_as_val = df.loc[ri, "IOD_DATA_SAVED_AS"] if "IOD_DATA_SAVED_AS" in df.columns else ""
                uid_i = uid_from_saved_as(saved_as_val, ri)
                committed_uids.append(uid_i)
        else:
            committed_uids = []

    # Broadcast committed list; workers create .done files now (commit-after-write)
    committed_uids = comm.bcast(committed_uids, root=0)
    for uid in committed_uids:
        try:
            with open(done_marker_path(uid), "w") as f:
                json.dump({"uid": uid, "status": "ok"}, f)
        except Exception:
            pass

    if rank == 0:
        print(f"[Stage: OD] committed_rows={len(committed_uids)}, processed={processed}, skipped={skipped}, errors={errors}")

    comm.Barrier()
    return



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
    # Stage 3: Running IOD (skip if MASTER rows already committed)
    #-----------------------
    if rank == 0:
        master_path = os.path.join(iod_dir, "MASTER_IOD.csv")
        stage3_done_dir = os.path.join(iod_dir, "iod_stage3_done")

        if not os.path.exists(master_path):
            do_stage3 = False
            msg3 = f"[Stage: IOD Solve] No MASTER_IOD.csv in {iod_dir} → skip"
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
                    for i, saved_as in enumerate(dfm.get("IOD_DATA_SAVED_AS", pd.Series([None]*n_rows))):
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


    #-----------------
    # Stage 4: Run the ATT.COOR. + OD Pipeline
    #------------------
    # run_OD(master)

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