import torch
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


# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
sp.furnsh("de430.bsp")
sp.furnsh('naif0012.tls')


def run_sim_runnumbers_less_memory(run_number, minimoon_master, config):
    df_buffer = pd.DataFrame(
        columns=["run_number", "object_id", "spacecraft_number", "values", "total_length", "spacecraft_1_ini_pos"])
    part_number = 1
    num_of_rows = config['number_of_rows_per_part']
    save_format = config['save_format']  # default to 'csv' if not specified

    for idx, master_i in minimoon_master.iterrows():
        current_minimoon = Asteroid(master_i['Object id'], master_i['Min_SunEarthL1_V_index'], config)
        formation = Formation(config)

        asteroid_pos = current_minimoon.orbit.loc[:, ['Synodic x', 'Synodic y', 'Synodic z']].values
        earth_pos = np.zeros_like(asteroid_pos)
        moon_pos = current_minimoon.orbit.loc[:, ['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']].values

        formation.match_spacecraft_trajectory(len(asteroid_pos[:, 0]), config)

        new_data = []  # temporary buffer for this minimoon

        for jdx, spacecraft in enumerate(formation.spacecraft):
            # print(run_number, current_minimoon.id, jdx + 1)

            sc_pos = spacecraft.matched_trajectory
            visible = spacecraft.asteroid_in_fov_batch(asteroid_pos, sc_pos, earth_pos, moon_pos, config)
            visible = np.array(visible)  # ensures it's an array
            len_visible = len(visible)
            visible = visible[visible >= 0]

            new_data.append({
                "run_number": run_number,
                "object_id": current_minimoon.id,
                "spacecraft_number": jdx + 1,
                "values": tuple(visible),
                "total_length": len_visible,
                "spacecraft_1_ini_pos": tuple(formation.spacecraft[0].ini_position)
            })

        df_buffer = pd.concat([df_buffer, pd.DataFrame(new_data)], ignore_index=True)

        if len(df_buffer) >= num_of_rows or idx == len(minimoon_master) - 1:
            df_buffer.set_index(["run_number", "object_id", "spacecraft_number"], inplace=True)

            base_filename = f"{config['output_df_file_name']}_run_{run_number}_part_{part_number}"

            if save_format == 'csv':
                filename = base_filename + ".csv"
                df_buffer.to_csv(filename, sep=',', header=True, index=True)
            elif save_format == 'parquet':
                filename = base_filename + ".parquet"
                df_buffer.to_parquet(filename, index=True)
            else:
                raise ValueError(f"Unsupported save format: {save_format}")

            print(f"Saved {len(df_buffer)} rows to {filename}")

            df_buffer = pd.DataFrame(columns=["run_number", "object_id", "spacecraft_number", "values", "total_length",
                                              "spacecraft_1_ini_pos"])
            part_number += 1

    return


def run_sim_runnumbers_MPI(minimoon_master, config):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine chunk size
    chunk_size = config['number_of_runs'] // size
    remainder = config['number_of_runs'] % size

    # Calculate start and end indices for this process
    start = rank * chunk_size
    end = start + chunk_size
    if rank == size - 1:
        end += remainder  # Last process takes any remaining elements

    for idx in range(start, end):
        # run_sim_runnumbers(idx + 1, minimoon_master, config)
        run_sim_runnumbers_less_memory(idx + 1, minimoon_master, config)

        # flat_data = []
        # for jdx, run in enumerate(result):
        #     index = tuple(run[0])  # (run_number, object_id, spacecraft_number)
        #     values = run[1]  # List of values over time
        #
        #     flat_data.append((*index, tuple(values), tuple(scs[0].ini_position)))
        #
        # Convert to DataFrame
        # df = pd.DataFrame(flat_data, columns=["run_number", "object_id", "spacecraft_number", "values", "spacecraft_1_ini_pos"])

        # Set MultiIndex
        # df.set_index(["run_number", "object_id", "spacecraft_number"], inplace=True)
        #
        # df.to_csv(config['output_df_file_name'] + '_run_' + str(idx + 1) + '.csv', sep=',', header=True, index=True)

    return

    ################################################
    # Single results file implementation
    ###############################################
    # results = []
    # spacecraft = []
    # for idx in range(start, end):
    #     result, scs = run_sim_runnumbers(idx + 1, minimoon_master, config)
    #     results.append(result)
    #     spacecraft.append(scs[0].ini_position)
    #
    # Gather all local centers_x and centers_y at root process (rank 0)
    # all_results = comm.gather(results, root=0)
    # all_scs = comm.gather(spacecraft, root=0)
    #
    # if rank == 0:
    #     return all_results, all_scs
    # else:
    #     return


def run_sim_runnumbers_MPI_getIOD_data_bychunk_helio(config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Master (rank 0) gathers the list of all files ---
    if rank == 0:
        # all_files = util.get_all_files(config['visible_files_folder'], config['save_format'])
        all_files = util.get_all_files_run_number(config['visible_files_folder'], config['save_format'],
                                                  config['run_number'])
        print(all_files)
        num_files = 0
    else:
        all_files = None

    # --- Broadcast total list to all ranks ---
    all_files = comm.bcast(all_files, root=0)

    # --- Distribute work: each rank gets a subset ---
    for i, file_i in enumerate(all_files):
        if rank == 0:
            print(f"Files completed: {num_files}")
            num_files += 1
            print(file_i.split('/')[-1].split('.')[0])

            run_data = util.read_master(file_i, config)

            # Create a mask to filter nonnegative values
            run_data["min_nonnegative"] = run_data["values"].apply(
                lambda x: min(x) if np.any(np.array(x) >= 0) else np.nan)

            # Find the spacecraft with the minimum value for each object_id
            detected_pop = run_data[~np.isnan(run_data["min_nonnegative"])]
            missed_pop = run_data[np.isnan(run_data["min_nonnegative"])]

            # Split detected_pop into chunks (one chunk per rank)
            num_rows = len(detected_pop)
            chunk_size = num_rows // size  # Base number of rows per rank
            remainder = num_rows % size  # Remainder to be distributed

            # Create the chunks
            chunks = []
            start_row = 0
            for i in range(size):
                # If there is a remainder, give one more row to the current rank
                end_row = start_row + chunk_size + (1 if i < remainder else 0)
                chunks.append(detected_pop.iloc[start_row:end_row])
                start_row = end_row

        else:
            chunks = None

        # --- Now scatter manually ---
        if rank == 0:
            # Send each chunk individually
            for dest in range(1, size):
                comm.send(chunks[dest], dest=dest, tag=77)
            my_chunk = chunks[0]
        else:
            # Receive my chunk
            my_chunk = comm.recv(source=0, tag=77)

        # Now my_chunk is a small DataFrame (only my portion)
        print(f"Rank {rank} got {len(my_chunk)} rows.")

        comm.barrier()

        config['num_spacecraft'] = int(file_i.split('/')[-1].split('.')[0].split('_')[1])
        detected_appended_pop_chunk = util.get_sc_state_from_sc1_position(my_chunk, config)

        print(f"Rank {rank} computed appended population")

        # re integrate according to exposure time and slew time to get 16 samples
        for jdx, detected_minimoon in detected_appended_pop_chunk.iterrows():

            # print(str(rank) + ': ' + str(detected_minimoon.name))
            file_path = config['minimoon_files_folder'] + detected_minimoon.name[1] + '.csv'
            orbit = pd.read_csv(file_path, sep=' ', header=0, names=config['minimoon_column_names'])

            # asteroid ##############
            asteroid_state_helio = orbit.loc[
                detected_minimoon['min_nonnegative'], ['Helio x', 'Helio y', 'Helio z', 'Helio vx', 'Helio vy',
                                                       'Helio vz']].values  # au and au/d
            asteroid_state_helio[:3] *= config['AU_TO_M'] / config['KM_TO_M']  # to match spice
            asteroid_state_helio[3:] *= (config['AU_TO_M'] / config['KM_TO_M'] / config['SECONDS_PER_DAY'])

            asteroid_epoch = orbit.loc[detected_minimoon['min_nonnegative'], 'Julian Date']

            # calc epochs
            num_frames = config['number_of_frames']
            start_time = asteroid_epoch
            step = config['time_between_frames'] / config['SECONDS_PER_DAY']
            total_observation_window = num_frames * step  # epoch is in jd
            epochs = np.arange(start_time, start_time + total_observation_window, step)

            # integrate s/c traj
            asteroid_integrated_states, asteroid_earth_states = nbody.integrate_n_body(asteroid_state_helio,
                                                                                       asteroid_epoch,
                                                                                       total_observation_window *
                                                                                       config['SECONDS_PER_DAY'],
                                                                                       config['time_between_frames'],
                                                                                       type="ASTEROID")  # integrator takes seconds

            asteroid_states_aud = util.ms_to_aud(asteroid_integrated_states)

            earthasteroid_states_aud = util.ms_to_aud(asteroid_earth_states)
            asteroid_state = util.helio_eclip_to_sun_earth_corotating_batch_full(asteroid_states_aud,
                                                                                 earthasteroid_states_aud)

            ##########
            # spacecraft
            ###############
            spacecraft_state_geo = detected_minimoon[
                ['GEO_ECLIP_X_(km)', 'GEO_ECLIP_Y_(km)', 'GEO_ECLIP_Z_(km)', 'GEO_ECLIP_Vx_(km/s)',
                 'GEO_ECLIP_Vy_(km/s)',
                 'GEO_ECLIP_Vz_(km/s)']].to_numpy()
            spacecraft_epoch = detected_minimoon['sc_epoch']

            # convert to heliocentric for integration
            sun_geo_state = sp.spkgeo(10, sp.str2et(spacecraft_epoch), "ECLIPJ2000", 399)[0]
            spacecraft_state_helio = spacecraft_state_geo - sun_geo_state

            # integrate s/c traj
            integrated_states, earth_states = nbody.integrate_n_body(spacecraft_state_helio, spacecraft_epoch,
                                                                     total_observation_window * config[
                                                                         'SECONDS_PER_DAY'],
                                                                     config['time_between_frames'], type="SPACECRAFT")

            integrated_states_aud = util.ms_to_aud(integrated_states)
            earth_states_aud = util.ms_to_aud(earth_states)

            spacecraft_state = util.helio_eclip_to_sun_earth_corotating_batch_full(integrated_states_aud,
                                                                                   earth_states_aud)

            # convert back to helio based on asteroid epoch
            new_spacecraft_state_helio = util.sun_earth_corotating_to_helio_eclip_batch_full(spacecraft_state,
                                                                                             earthasteroid_states_aud)
            new_asteroid_state_helio = util.sun_earth_corotating_to_helio_eclip_batch_full(asteroid_state,
                                                                                           earthasteroid_states_aud)

            #############

            # calc ra and dec from helio
            x_rel = new_asteroid_state_helio[0, :] - new_spacecraft_state_helio[0, :]
            y_rel = new_asteroid_state_helio[1, :] - new_spacecraft_state_helio[1, :]
            z_rel = new_asteroid_state_helio[2, :] - new_spacecraft_state_helio[2, :]

            r_xy = np.sqrt(x_rel ** 2 + y_rel ** 2)
            r = np.sqrt(x_rel ** 2 + y_rel ** 2 + z_rel ** 2)

            sin_ra = y_rel / r_xy
            cos_ra = x_rel / r_xy
            sin_dec = z_rel / r

            # generate output file with epoch , ast xyz vxvyvz, detecting sc id xyz vxvyvz sinRA cosRA sinDec
            # file name: run-x_minimoon-y_sc-z_index-k.csv
            file_name = ('minimoon-' + str(detected_minimoon.name[1]) + '_sc-'
                         + str(detected_minimoon.name[2]) + '_index-' + str(int(detected_minimoon['min_nonnegative'])))
            file_path = (config['IOD_folder_path'] + '/' + file_name + '_' + file_i.split('/')[-1].split('.')[0] +
                         '.csv')

            # make dataframe
            data = np.array([epochs, new_asteroid_state_helio[0, :], new_asteroid_state_helio[1, :],
                             new_asteroid_state_helio[2, :], new_asteroid_state_helio[3, :],
                             new_asteroid_state_helio[4, :],
                             new_asteroid_state_helio[5, :], new_spacecraft_state_helio[0, :],
                             new_spacecraft_state_helio[1, :],
                             new_spacecraft_state_helio[2, :], new_spacecraft_state_helio[3, :],
                             new_spacecraft_state_helio[4, :],
                             new_spacecraft_state_helio[5, :], sin_ra, cos_ra, sin_dec]).T

            df = pd.DataFrame(data, columns=config['IOD_data_columns'])
            # Get desired format from config
            output_format = config['save_format']  # default to csv

            # Write based on format
            base_path, _ = os.path.splitext(file_path)

            if output_format in ['csv', 'both']:
                df.to_csv(base_path + '.csv', sep=',', header=True, index=False)

            if output_format in ['parquet', 'both']:
                df.to_parquet(base_path + '.parquet', index=False)

            vis = False
            if vis:
                #######################
                # for visualization
                ###################
                # get the original s/c orbit and get the original s/c starting points
                # create a formation object, it has s/c s randomly placed
                formation = Formation(config)

                # we already had a saved formation, saved according to the s/c 1 position, get the correspoding index in overall orbit file
                sc1_ini_index = formation.get_index_from_pos(detected_minimoon['spacecraft_1_ini_pos'])

                # re-initialize formation with this index
                formation.recall_formation(sc1_ini_index, config)

                # match the spacecraft trajectories to that of the asteroid in terms of length and sampling (asteroid sampled at one hour)
                formation.match_spacecraft_trajectory(len(detected_minimoon['values']), config)
                current_minimoon = Asteroid(detected_minimoon.name[1], 100, config)

                # calc ra and dec from sun-earth-co (visualized in a different frame)
                # calc ra and dec from helio
                x_rel = asteroid_state[0, :] + spacecraft_state[0, :]
                y_rel = asteroid_state[1, :] + spacecraft_state[1, :]
                z_rel = asteroid_state[2, :] - spacecraft_state[2, :]

                # r_xy = np.sqrt(x_rel ** 2 + y_rel ** 2)
                r = np.sqrt(x_rel ** 2 + y_rel ** 2 + z_rel ** 2)

                # sin_ra = y_rel / r_xy
                cos_ra = x_rel / r_xy
                sin_dec = z_rel / r

                # visualize it all
                util.viz(spacecraft_state[:3, :], asteroid_state[:3, :], current_minimoon, formation,
                         [sin_ra, cos_ra, sin_dec], config)
                ################

    return


def run_sim_runnumbers_MPI_getIOD_data_bychunk_geo(config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Master (rank 0) gathers the list of all files ---
    if rank == 0:
        all_files = util.get_all_files(config['visible_files_folder'], config['save_format'])
        # all_files = util.get_all_files_run_number(config['visible_files_folder'], config['save_format'], config['run_number'])
        print(all_files)
        num_files = 0
    else:
        all_files = None

    # --- Broadcast total list to all ranks ---
    all_files = comm.bcast(all_files, root=0)

    # --- Distribute work: each rank gets a subset ---
    for i, file_i in enumerate(all_files):
        if rank == 0:
            print(f"Files completed: {num_files}")
            num_files += 1
            print(file_i.split('/')[-1].split('.')[0])

            run_data = util.read_master(file_i, config)

            # Create a mask to filter nonnegative values
            run_data["min_nonnegative"] = run_data["values"].apply(
                lambda x: min(x) if np.any(np.array(x) >= 0) else np.nan)

            # Find the spacecraft with the minimum value for each object_id
            detected_pop = run_data[~np.isnan(run_data["min_nonnegative"])]
            missed_pop = run_data[np.isnan(run_data["min_nonnegative"])]

            # Split detected_pop into chunks (one chunk per rank)
            num_rows = len(detected_pop)
            chunk_size = num_rows // size  # Base number of rows per rank
            remainder = num_rows % size  # Remainder to be distributed

            # Create the chunks
            chunks = []
            start_row = 0
            for i in range(size):
                # If there is a remainder, give one more row to the current rank
                end_row = start_row + chunk_size + (1 if i < remainder else 0)
                chunks.append(detected_pop.iloc[start_row:end_row])
                start_row = end_row

        else:
            chunks = None

        # --- Now scatter manually ---
        if rank == 0:
            # Send each chunk individually
            for dest in range(1, size):
                comm.send(chunks[dest], dest=dest, tag=77)
            my_chunk = chunks[0]
        else:
            # Receive my chunk
            my_chunk = comm.recv(source=0, tag=77)

        # Now my_chunk is a small DataFrame (only my portion)
        print(f"Rank {rank} got {len(my_chunk)} rows.")

        comm.barrier()

        config['num_spacecraft'] = int(file_i.split('/')[-1].split('.')[0].split('_')[1])
        detected_appended_pop_chunk = util.get_sc_state_from_sc1_position(my_chunk, config)

        print(f"Rank {rank} computed appended population")

        # re integrate according to exposure time and slew time to get 16 samples
        for jdx, detected_minimoon in detected_appended_pop_chunk.iterrows():

            # print(str(rank) + ': ' + str(detected_minimoon.name))
            file_path = config['minimoon_files_folder'] + detected_minimoon.name[1] + '.csv'
            orbit = pd.read_csv(file_path, sep=' ', header=0, names=config['minimoon_column_names'])

            # asteroid ##############
            asteroid_state_helio = orbit.loc[
                detected_minimoon['min_nonnegative'], ['Helio x', 'Helio y', 'Helio z', 'Helio vx', 'Helio vy',
                                                       'Helio vz']].values  # au and au/d
            asteroid_state_helio[:3] *= config['AU_TO_M'] / config['KM_TO_M']  # to match spice
            asteroid_state_helio[3:] *= (config['AU_TO_M'] / config['KM_TO_M'] / config['SECONDS_PER_DAY'])

            asteroid_epoch = orbit.loc[detected_minimoon['min_nonnegative'], 'Julian Date']

            # calc epochs
            num_frames = config['number_of_frames']
            start_time = asteroid_epoch
            step = config['time_between_frames'] / config['SECONDS_PER_DAY']
            total_observation_window = num_frames * step  # epoch is in jd
            epochs = np.arange(start_time, start_time + total_observation_window, step)

            # integrate s/c traj - output is km and km/s
            asteroid_integrated_states, asteroid_earth_states = nbody.integrate_n_body(asteroid_state_helio,
                                                                                       asteroid_epoch,
                                                                                       total_observation_window *
                                                                                       config['SECONDS_PER_DAY'],
                                                                                       config['time_between_frames'],
                                                                                       type="ASTEROID")  # integrator takes seconds

            # km and km/s
            asteroid_state = util.helio_eclip_to_sun_earth_corotating_batch_full(asteroid_integrated_states,
                                                                                 asteroid_earth_states)
            # directly from helio eclip j2000 to eme j2000

            ##########
            # spacecraft
            ###############
            spacecraft_state_geo = detected_minimoon[
                ['GEO_ECLIP_X_(km)', 'GEO_ECLIP_Y_(km)', 'GEO_ECLIP_Z_(km)', 'GEO_ECLIP_Vx_(km/s)',
                 'GEO_ECLIP_Vy_(km/s)',
                 'GEO_ECLIP_Vz_(km/s)']].to_numpy()
            spacecraft_epoch = detected_minimoon['sc_epoch']

            # convert to heliocentric for integration
            sun_geo_state = sp.spkgeo(10, sp.str2et(spacecraft_epoch), "ECLIPJ2000", 399)[0]
            spacecraft_state_helio = spacecraft_state_geo - sun_geo_state

            # integrate s/c traj
            integrated_states, earth_states = nbody.integrate_n_body(spacecraft_state_helio, spacecraft_epoch,
                                                                     total_observation_window * config[
                                                                         'SECONDS_PER_DAY'],
                                                                     config['time_between_frames'], type="SPACECRAFT")

            # in km, km/s
            spacecraft_state = util.helio_eclip_to_sun_earth_corotating_batch_full(integrated_states, earth_states)
            spacecraft_state[:2, :] *= -1

            # convert back to geo eclip based on asteroid epoch
            new_spacecraft_state_geo = util.sun_earth_corotating_to_geo_eclip_batch_full(spacecraft_state,
                                                                                         asteroid_earth_states)
            new_asteroid_state_geo = util.sun_earth_corotating_to_geo_eclip_batch_full(asteroid_state,
                                                                                       asteroid_earth_states)

            # convert to geo eme
            new_spacecraft_state_geo_eme = util.ecliptic_to_eme_batch(new_spacecraft_state_geo)
            new_asteroid_state_geo_eme = util.ecliptic_to_eme_batch(new_asteroid_state_geo)

            #############

            # calc ra and dec from helio
            x_rel = new_asteroid_state_geo_eme[0, :] - new_spacecraft_state_geo_eme[0, :]
            y_rel = new_asteroid_state_geo_eme[1, :] - new_spacecraft_state_geo_eme[1, :]
            z_rel = new_asteroid_state_geo_eme[2, :] - new_spacecraft_state_geo_eme[2, :]

            r_xy = np.sqrt(x_rel ** 2 + y_rel ** 2)
            r = np.sqrt(x_rel ** 2 + y_rel ** 2 + z_rel ** 2)

            sin_ra = y_rel / r_xy
            cos_ra = x_rel / r_xy
            sin_dec = z_rel / r

            # generate output file with epoch , ast xyz vxvyvz, detecting sc id xyz vxvyvz sinRA cosRA sinDec
            # file name: run-x_minimoon-y_sc-z_index-k.csv
            file_name = ('minimoon-' + str(detected_minimoon.name[1]) + '_sc-'
                         + str(detected_minimoon.name[2]) + '_index-' + str(int(detected_minimoon['min_nonnegative'])))
            file_path = (config['IOD_folder_path'] + '/' + file_name + '_' + file_i.split('/')[-1].split('.')[0] +
                         '.csv')

            # make dataframe
            data = np.array([epochs, new_asteroid_state_geo_eme[0, :], new_asteroid_state_geo_eme[1, :],
                             new_asteroid_state_geo_eme[2, :], new_asteroid_state_geo_eme[3, :], new_asteroid_state_geo_eme[4, :],
                             new_asteroid_state_geo_eme[5, :], new_spacecraft_state_geo_eme[0, :],
                             new_spacecraft_state_geo_eme[1, :],
                             new_spacecraft_state_geo_eme[2, :], new_spacecraft_state_geo_eme[3, :],
                             new_spacecraft_state_geo_eme[4, :],
                             new_spacecraft_state_geo_eme[5, :], sin_ra, cos_ra, sin_dec]).T

            df = pd.DataFrame(data, columns=config['IOD_data_columns_geo'])
            # Get desired format from config
            output_format = config['save_format']  # default to csv

            # Write based on format
            base_path, _ = os.path.splitext(file_path)

            if output_format in ['csv', 'both']:
                df.to_csv(base_path + '.csv', sep=',', header=True, index=False)

            if output_format in ['parquet', 'both']:
                df.to_parquet(base_path + '.parquet', index=False)

            vis = True
            if vis:
                #######################
                # for visualization
                ###################
                # get the original s/c orbit and get the original s/c starting points
                # create a formation object, it has s/c s randomly placed
                formation = Formation(config)

                # we already had a saved formation, saved according to the s/c 1 position, get the correspoding index in overall orbit file
                sc1_ini_index = formation.get_index_from_pos(detected_minimoon['spacecraft_1_ini_pos'])

                # re-initialize formation with this index
                formation.recall_formation(sc1_ini_index, config)

                # match the spacecraft trajectories to that of the asteroid in terms of length and sampling (asteroid sampled at one hour)
                formation.match_spacecraft_trajectory(int(detected_minimoon['total_length']), config)
                current_minimoon = Asteroid(detected_minimoon.name[1], 100, config)

                # visualize it all
                util.viz_geo_and_secr(spacecraft_state[:3, :], asteroid_state[:3, :], current_minimoon, formation,
                                      [sin_ra, cos_ra, sin_dec], detected_minimoon, new_asteroid_state_geo_eme,
                                      new_spacecraft_state_geo_eme, config)
                ################

    return


def run_sim_runnumbers_MPI_getIOD_data_bychunk_eme(config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Master (rank 0) gathers the list of all files ---
    if rank == 0:
        # all_files = util.get_all_files(config['visible_files_folder'], config['save_format'])
        all_files = util.get_all_files_run_number(config['visible_files_folder'], config['save_format'], config['run_number'])
        num_files = 0
    else:
        all_files = None

    # --- Broadcast total list to all ranks ---
    all_files = comm.bcast(all_files, root=0)

    # --- Distribute work: each rank gets a subset ---
    for i, file_i in enumerate(all_files):
        if rank == 0:
            print(f"Files completed: {num_files}")
            num_files += 1
            print(file_i.split('/')[-1].split('.')[0])

            run_data = util.read_master(file_i, config)

            # Create a mask to filter nonnegative values
            run_data["min_nonnegative"] = run_data["values"].apply(
                lambda x: min(x) if np.any(np.array(x) >= 0) else np.nan)

            # Find the spacecraft with the minimum value for each object_id
            detected_pop = run_data[~np.isnan(run_data["min_nonnegative"])]
            missed_pop = run_data[np.isnan(run_data["min_nonnegative"])]

            # Split detected_pop into chunks (one chunk per rank)
            num_rows = len(detected_pop)
            chunk_size = num_rows // size  # Base number of rows per rank
            remainder = num_rows % size  # Remainder to be distributed

            # Create the chunks
            chunks = []
            start_row = 0
            for i in range(size):
                # If there is a remainder, give one more row to the current rank
                end_row = start_row + chunk_size + (1 if i < remainder else 0)
                chunks.append(detected_pop.iloc[start_row:end_row])
                start_row = end_row

        else:
            chunks = None

        # --- Now scatter manually ---
        if rank == 0:
            # Send each chunk individually
            for dest in range(1, size):
                comm.send(chunks[dest], dest=dest, tag=77)
            my_chunk = chunks[0]
        else:
            # Receive my chunk
            my_chunk = comm.recv(source=0, tag=77)

        # Now my_chunk is a small DataFrame (only my portion)
        print(f"Rank {rank} got {len(my_chunk)} rows.")

        comm.barrier()

        config['num_spacecraft'] = int(file_i.split('/')[-1].split('.')[0].split('_')[1])
        detected_appended_pop_chunk = util.get_sc_state_from_sc1_position(my_chunk, config)

        print(f"Rank {rank} computed appended population")

        # re integrate according to exposure time and slew time to get 16 samples
        for jdx, detected_minimoon in detected_appended_pop_chunk.iterrows():

            # print(str(rank) + ': ' + str(detected_minimoon.name))
            file_path = config['minimoon_files_folder'] + detected_minimoon.name[1] + '.csv'
            orbit = pd.read_csv(file_path, sep=' ', header=0, names=config['minimoon_column_names'])


            # asteroid ##############
            asteroid_state_helio = orbit.loc[
                detected_minimoon['min_nonnegative'], ['Helio x', 'Helio y', 'Helio z', 'Helio vx', 'Helio vy',
                                                       'Helio vz']].values  # au and au/d
            asteroid_state_helio[:3] *= config['AU_TO_M'] / config['KM_TO_M']  # to match spice
            asteroid_state_helio[3:] *= (config['AU_TO_M'] / config['KM_TO_M'] / config['SECONDS_PER_DAY'])

            asteroid_epoch = orbit.loc[detected_minimoon['min_nonnegative'], 'Julian Date']

            # calc epochs
            num_frames = config['number_of_frames']
            start_time = asteroid_epoch
            step = config['time_between_frames'] / config['SECONDS_PER_DAY']
            total_observation_window = num_frames * step  # epoch is in jd
            epochs = start_time + step * np.arange(num_frames)

            # integrate s/c traj - output is km and km/s
            asteroid_integrated_states, asteroid_earth_states = nbody.integrate_n_body(asteroid_state_helio,
                                                                                       asteroid_epoch,
                                                                                       total_observation_window *
                                                                                       config['SECONDS_PER_DAY'],
                                                                                       config['time_between_frames'],
                                                                                       type="ASTEROID")  # integrator takes seconds

            # km and km/s
            asteroid_state = util.helio_eclip_to_sun_earth_corotating_batch_full(asteroid_integrated_states,
                                                                                 asteroid_earth_states)
            # directly from helio eclip j2000 to eme j2000

            ##########
            # spacecraft
            ###############
            spacecraft_state_geo = detected_minimoon[
                ['GEO_ECLIP_X_(km)', 'GEO_ECLIP_Y_(km)', 'GEO_ECLIP_Z_(km)', 'GEO_ECLIP_Vx_(km/s)',
                 'GEO_ECLIP_Vy_(km/s)',
                 'GEO_ECLIP_Vz_(km/s)']].to_numpy()
            spacecraft_epoch = detected_minimoon['sc_epoch']

            # convert to heliocentric for integration
            sun_geo_state = sp.spkgeo(10, sp.str2et(spacecraft_epoch), "ECLIPJ2000", 399)[0]
            spacecraft_state_helio = spacecraft_state_geo - sun_geo_state


            #####################################
            # Geometrically sound
            ####################################

            # integrate s/c traj
            integrated_states, earth_states = nbody.integrate_n_body(spacecraft_state_helio, spacecraft_epoch,
                                                                     total_observation_window * config[
                                                                         'SECONDS_PER_DAY'],
                                                                     config['time_between_frames'], type="SPACECRAFT")

            # in km, km/s
            spacecraft_state = util.helio_eclip_to_sun_earth_corotating_batch_full(integrated_states, earth_states)

            # convert back to geo eclip based on asteroid epoch
            new_spacecraft_state_geo = util.sun_earth_corotating_to_geo_eclip_batch_full(spacecraft_state,
                                                                                         asteroid_earth_states)
            new_asteroid_state_geo = util.sun_earth_corotating_to_geo_eclip_batch_full(asteroid_state,
                                                                                       asteroid_earth_states)

            # convert to geo eme
            new_spacecraft_state_geo_eme = util.ecliptic_to_eme_batch(new_spacecraft_state_geo)
            new_asteroid_state_geo_eme = util.ecliptic_to_eme_batch(new_asteroid_state_geo)

            # calc ra and dec from helio
            x_rel = new_asteroid_state_geo_eme[0, :] - new_spacecraft_state_geo_eme[0, :]
            y_rel = new_asteroid_state_geo_eme[1, :] - new_spacecraft_state_geo_eme[1, :]
            z_rel = new_asteroid_state_geo_eme[2, :] - new_spacecraft_state_geo_eme[2, :]

            r_xy = np.sqrt(x_rel ** 2 + y_rel ** 2)
            r = np.sqrt(x_rel ** 2 + y_rel ** 2 + z_rel ** 2)

            sin_ra = y_rel / r_xy
            cos_ra = x_rel / r_xy
            sin_dec = z_rel / r


            #################
            # physically sound
            #################

            # grab initial spacecraft state in SECR
            sc_secr_ini_state = spacecraft_state[:, 0]
            earth_helio_ini_state = asteroid_earth_states[:, 0]

            # convert to helio according to asteroid epoch system
            sc_helio_ini_state = util.sun_earth_corotating_to_helio_eclip_single(sc_secr_ini_state, earth_helio_ini_state)


            # integrate n-body
            sc_helio_states, asteroid_earth_states_2 = nbody.integrate_n_body(sc_helio_ini_state, asteroid_epoch,
                                   total_observation_window * config[
                                       'SECONDS_PER_DAY'],
                                   config['time_between_frames'], type="SPACECRAFT-ASTEROIDTIME")


            # conver to eme
            sc_eme_states = util.helio_eclip_to_geo_eme_batch(sc_helio_states, asteroid_earth_states)

            # calc ra and dec from helio
            x_rel_phys = new_asteroid_state_geo_eme[0, :] - sc_eme_states[0, :]
            y_rel_phys = new_asteroid_state_geo_eme[1, :] - sc_eme_states[1, :]
            z_rel_phys = new_asteroid_state_geo_eme[2, :] - sc_eme_states[2, :]

            r_xy_phys = np.sqrt(x_rel_phys ** 2 + y_rel_phys ** 2)
            r_phys = np.sqrt(x_rel_phys ** 2 + y_rel_phys ** 2 + z_rel_phys ** 2)

            sin_ra_phys = y_rel_phys / r_xy_phys
            cos_ra_phys = x_rel_phys / r_xy_phys
            sin_dec_phys = z_rel_phys / r_phys

            # make dataframe
            data = np.array([epochs, new_asteroid_state_geo_eme[0, :], new_asteroid_state_geo_eme[1, :],
                             new_asteroid_state_geo_eme[2, :], new_asteroid_state_geo_eme[3, :], new_asteroid_state_geo_eme[4, :],
                             new_asteroid_state_geo_eme[5, :], new_spacecraft_state_geo_eme[0, :],
                             new_spacecraft_state_geo_eme[1, :],
                             new_spacecraft_state_geo_eme[2, :], new_spacecraft_state_geo_eme[3, :],
                             new_spacecraft_state_geo_eme[4, :],
                             new_spacecraft_state_geo_eme[5, :], sin_ra, cos_ra, sin_dec, sc_eme_states[0, :],
                             sc_eme_states[1, :], sc_eme_states[2, :], sc_eme_states[3, :], sc_eme_states[4, :],
                             sc_eme_states[5, :], sin_ra_phys, cos_ra_phys, sin_dec_phys]).T


            # generate output file with epoch , ast xyz vxvyvz, detecting sc id xyz vxvyvz sinRA cosRA sinDec
            # file name: run-x_minimoon-y_sc-z_index-k.csv
            file_name = ('minimoon-' + str(detected_minimoon.name[1]) + '_sc-'
                         + str(detected_minimoon.name[2]) + '_index-' + str(int(detected_minimoon['min_nonnegative'])))
            file_path = (config['IOD_folder_path'] + '/' + file_name + '_' + file_i.split('/')[-1].split('.')[0] +
                         '.csv')

            df = pd.DataFrame(data, columns=config['IOD_data_columns_geo_and_phys'])
            # Get desired format from config
            output_format = config['save_format']  # default to csv

            # Write based on format
            base_path, _ = os.path.splitext(file_path)

            if output_format in ['csv', 'both']:
                df.to_csv(base_path + '.csv', sep=',', header=True, index=False)

            if output_format in ['parquet', 'both']:
                df.to_parquet(base_path + '.parquet', index=False)


            vis = False
            if vis:
                #######################
                # for visualization
                ###################
                # get the original s/c orbit and get the original s/c starting points
                # create a formation object, it has s/c s randomly placed
                formation = Formation(config)

                # we already had a saved formation, saved according to the s/c 1 position, get the correspoding index in overall orbit file
                sc1_ini_index = formation.get_index_from_pos(detected_minimoon['spacecraft_1_ini_pos'])

                # re-initialize formation with this index
                formation.recall_formation(sc1_ini_index, config)

                # match the spacecraft trajectories to that of the asteroid in terms of length and sampling (asteroid sampled at one hour)
                formation.match_spacecraft_trajectory(int(detected_minimoon['total_length']), config)
                current_minimoon = Asteroid(detected_minimoon.name[1], 100, config)

                # visualize it all
                util.viz_geo_and_secr(spacecraft_state[:3, :], asteroid_state[:3, :], current_minimoon, formation,
                                      [sin_ra, cos_ra, sin_dec], detected_minimoon, new_asteroid_state_geo_eme,
                                      new_spacecraft_state_geo_eme, sc_eme_states, config)
                ################

    return


def run_IOD_MPI(config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Master (rank 0) gathers the list of all files ---
    if rank == 0:
        all_files = util.get_all_files(config['IOD_folder_path'], config['save_format'])
        num_files = 0
    else:
        all_files = None

    # --- Broadcast total list to all ranks ---
    all_files = comm.bcast(all_files, root=0)

    # --- Round-robin distribution ---
    files_for_this_rank = all_files[rank::size]

    # --- Process each file assigned to this rank ---
    for file_path in files_for_this_rank:
        print(file_path.split('/')[-1])

        # file_path = 'output.csv'

        # read data
        iod_data = util.read_IOD_data_geo(file_path, config)

        # add noise in quadrature - assuming independent
        sigma_ra = np.sqrt(config['sigma_ra'] ** 2 + config['sigma_pointing'] ** 2) / config['MAS_TO_DEGREE']
        sigma_dec = np.sqrt(config['sigma_dec'] ** 2 + config['sigma_pointing'] ** 2) / config['MAS_TO_DEGREE']
        iod_data_w_noise = util.add_noise_to_angles(iod_data, sigma_ra, sigma_dec)

        # observation epochs
        obs_e = iod_data_w_noise['EPOCH(JDTDB)'].values
        # spacecraft position at observation
        spacecraft_pos = iod_data_w_noise.loc[:, ['SC_GEO_X(KM)_PHYS', 'SC_GEO_Y(KM)_PHYS', 'SC_GEO_Z(KM)_PHYS']].values
        # observations - without noise
        obs = iod_data.loc[:, ['SIN_RA_PHYS', 'COS_RA_PHYS', 'SIN_DEC_PHYS']].values
        # observations - wit noise
        # obs = iod_data_w_noise.loc[:, ['SIN_RA', 'COS_RA', 'SIN_DEC']].values

        # other relevant parameters
        delta = 1
        num_points = 500
        layer_ratios = [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)]
        # mean = obs_e[0] + (obs_e[-1] - obs_e[0]) / 2
        # std = (obs_e[-1] - obs_e[0]) * 4
        z_range = (-1, 1)
        method = "uniform"
        hidden_dim = 150

        ###################################
        # Non-linear least squares
        ###################################

        # generate collocation points
        # colloc_points = pielm_nlls.sample_time_points(method, obs_e, delta, num_points, layer_ratios=layer_ratios,
        #                                               config=config)
        # Step 1: Build a mask for which collocation points are in the observation epochs
        # obs_mask = np.isin(colloc_points, obs_e)

        # Step 2: Get the indices of observation epochs in colloc_points
        # obs_indices = np.where(obs_mask)[0]

        # normalize epochs (inputs)
        # epochs_nd_norm, c = pielm_nlls.epoch_normalization(colloc_points, z_range, config)

        # as a 2D tensor
        # epochs_nd_norm_reshaped_tensor = torch.tensor(epochs_nd_norm, dtype=torch.float32).unsqueeze(1)

        # train pielm
        # true = iod_data_w_noise.loc[:, ['GEO_X(KM)', 'GEO_Y(KM)', 'GEO_Z(KM)']].values
        # pielm_nlls.solve(true, epochs_nd_norm_reshaped_tensor, obs, obs_indices, spacecraft_pos,
        #                  colloc_points, hidden_dim, c, config)

        ########################################
        # Stochastic Gradient Desent Implementation
        ##########################################

        # generate collocation points
        colloc_points = pielm_sgd.sample_time_points(method, obs_e, delta, num_points, layer_ratios=layer_ratios, config=config)

        # Step 1: Build a mask for which collocation points are in the observation epochs
        obs_mask = np.isin(colloc_points, obs_e)

        # Step 2: Get the indices of observation epochs in colloc_points
        obs_indices = np.where(obs_mask)[0]

        # normalize epochs (inputs)
        epochs_nd_norm, c = pielm_sgd.epoch_normalization(colloc_points, z_range, config)

        # as a 2D tensor
        epochs_nd_norm_reshaped_tensor = torch.tensor(epochs_nd_norm, dtype=torch.float32).unsqueeze(1)


        # initial position guess - just initial observation to delta uniformly
        # integrate n-body
        ini_eme_state = iod_data_w_noise.loc[0, ["GEO_X(KM)", "GEO_Y(KM)", "GEO_Z(KM)", "GEO_VX(KM/S)", "GEO_VY(KM/S)",
                   "GEO_VZ(KM/S)"]].values
        ini_epoch = iod_data_w_noise.loc[0, "EPOCH(JDTDB)"] - delta
        ini_earth_helio = sp.spkgeo(399, sp.unitim(ini_epoch, 'JDTDB', 'ET'), "ECLIPJ2000", 10)[0]
        ini_eclip_state = util.eme_to_ecliptic_batch(ini_eme_state)
        ini_helio_state =  ini_eclip_state + ini_earth_helio

        # calc epochs
        step = 2 * delta / num_points
        total_observation_window = 2 * delta  # epoch is in jd

        ini_helio_states, ini_earth_states = nbody.integrate_n_body(ini_helio_state, ini_epoch,
                                                                          total_observation_window * config[
                                                                              'SECONDS_PER_DAY'],
                                                                          step * config[
                                                                              'SECONDS_PER_DAY'],
                                                                          type="ASTEROID")

        # conver to eme
        L = config['normalization_ratio'] * config['EARTH_RADIUS_KM']  # Length scale in km
        ini_eme_states = util.helio_eclip_to_geo_eme_batch(ini_helio_states, ini_earth_states)
        ini_eme_positions_nd = torch.tensor(ini_eme_states[:3, :], dtype=torch.float32) / L

        # declare pielm
        elm = ELM(hidden_dim, c_normalization=c, initial_nd_positions=ini_eme_positions_nd, z_s=epochs_nd_norm_reshaped_tensor)

        # train pielm
        res, positions, vel = pielm_sgd.train(elm, epochs_nd_norm_reshaped_tensor, obs, obs_indices, spacecraft_pos, obs_e, colloc_points, config)


        util.iod_viz(iod_data_w_noise, res, positions, vel, config)

    return


def run_IOD_testing(config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_runs = config['n_runs_test']

    # --- Round-robin assignment ---
    local_master = []

    for run_idx in range(n_runs):

        if run_idx % size == rank:

            # get case
            dynamics, orbit, observer, optimizer = config['dynamics'], config['orbit'], config['observer'], config['optimizer']
            if dynamics == '2BD' and orbit == 'GEO' and observer == 'GROUND' and optimizer == 'SGD':
                import PIELM_sgd_geo_earth as pielm_2gg
                parameters = {'NUMBER_OF_OBSERVATIONS': 260, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.0000000001 * u.day,
                              'TOTAL_POINTS': 300, 'SAMPLING_METHOD': "lhs",
                              'LAYER_RATIOS': [(0., 1 / 10000), (1 / 10000, 9999 / 10000), (9999 / 10000, 1.)],
                              'INPUT_RANGE': (-1, 1), 'WEIGHT_SCALE_FACTOR': 1e0,
                              'HIDDEN_DIMENSION': 20, 'NUMBER_OF_EPOCHS': 100000, 'LEARNING_RATE': 1e-2,
                              'PHYSICS_WEIGHT': 1e2, 'STEPSIZE': 1, 'NUMBER_OF_ITERATIONS': 50, 'TEMPERATURE': 1,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_NFEV': 100,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15.}
                config['lambda'] = parameters['PHYSICS_WEIGHT']
                data = pielm_2gg.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_2gg.run(
                    data, config, parameters)

            elif dynamics == '2BD' and orbit == 'GEO' and observer == 'GROUND' and optimizer == 'NLLS':
                import  PIELM_nlls_geo_earth as pielm_2ggn
                parameters = {'NUMBER_OF_OBSERVATIONS': 260, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.0000000001 * u.day,
                              'TOTAL_POINTS': 300, 'SAMPLING_METHOD': "gaussian",
                              'LAYER_RATIOS': [(0., 1 / 10000), (1 / 10000, 9999 / 10000), (9999 / 10000, 1.)],
                              'INPUT_RANGE': (-1, 1), 'WEIGHT_SCALE_FACTOR': 1.0e0,
                              'HIDDEN_DIMENSION': 20, 'NUMBER_OF_EPOCHS': 50000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': 1e0, 'STEPSIZE': 1, 'NUMBER_OF_ITERATIONS': 50, 'TEMPERATURE': 1,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_NFEV': 1000,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15.}
                config['lambda'] = parameters['PHYSICS_WEIGHT']
                data = pielm_2ggn.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_2ggn.run(
                    data, config, parameters)

            elif dynamics == '2BD' and orbit=='GEO' and observer == 'GROUND' and optimizer == 'CONSTRAINED_BASIN_HOPPING':
                import PIELM_basinhopping_w_range_geo_earth as pielm_2ggcb
                parameters = {'NUMBER_OF_OBSERVATIONS': 260, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.0000000001 * u.day,
                              'TOTAL_POINTS': 300, 'SAMPLING_METHOD': "gaussian",
                              'LAYER_RATIOS': [(0., 1 / 10000), (1 / 10000, 9999 / 10000), (9999 / 10000, 1.)],
                              'INPUT_RANGE': (-1, 1), 'WEIGHT_SCALE_FACTOR': 1e0,
                              'HIDDEN_DIMENSION': 20, 'NUMBER_OF_EPOCHS': 50000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': 1e2, 'LAMBDA_DIST': 1e-7, 'STEPSIZE': 10e-5,
                              'NUMBER_OF_ITERATIONS': 100, 'TEMPERATURE': 10e-8,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000, 'G_TOLERANCE': 1e-15, 'MAX_NFEV':100,
                              'A_PERT': 0., 'ECC_PERT': 0., 'INC_PERT': 0., 'RAAN_PERT': 0., 'ARGPER_PERT': 0.,
                              'ANOM_PERT': 0., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': 95 - run_idx,
                              'MIN_RHO': 1.0976e0, 'MAX_RHO': 2.35157e2, 'MIN_RHO_DOT': -1.2647e0,
                              'MAX_RHO_DOT': 1.2647e0, 'DELTA_RHO_STEP': 1.5679e0, 'DELTA_RHO_DOT_STEP': 1.265e-1,
                              'INITIAL_TRAJECTORIES': 1}
                config['lambda'] = parameters['PHYSICS_WEIGHT']
                config['run_idx'] = run_idx
                data = pielm_2ggcb.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_2ggcb.run(data, config, parameters)

            elif dynamics == 'CR3BP' and observer == 'GROUND' and optimizer == 'NLLS':
                import  PIELM_nlls_periodicorbits_earth_cr3bp as pielm_cgn
                parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 15000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': 1e-2, 'STEPSIZE': 10, 'NUMBER_OF_ITERATIONS': 500, 'TEMPERATURE': 1000,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000, 'WEIGHT_SCALE_FACTOR': 1e-6,
                              'G_TOLERANCE': 1e-15, 'MAX_NFEV': 1000,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_idx}
                if config['orbit'] == 'Horizontal Lyapunov Orbits':
                    data = pielm_cgn.generate_data(config, parameters)
                elif config['orbit'] == 'Halo Orbits':
                    parameters['ORBIT_TYPE'] = 'Halo Orbits'
                    data = pielm_cgn.generate_data(config, parameters)
                else:
                    parameters['ORBIT_TYPE'] = 'Vertical Lyapunov Orbits'
                    data = pielm_cgn.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_cgn.run(data, config, parameters)

            elif dynamics == 'CR3BP' and observer == 'GROUND' and optimizer == 'SGD':
                import  PIELM_sgd_periodicorbits_earth_cr3bp as pielm_cgs
                parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5, 'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "lhs",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 100000, 'LEARNING_RATE': 1e-2,
                              'PHYSICS_WEIGHT': 1e-4, 'STEPSIZE': 1, 'NUMBER_OF_ITERATIONS': 50, 'TEMPERATURE': 1,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'G_TOLERANCE': 1e-15, 'WEIGHT_SCALE_FACTOR': 1e-4,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_idx
                               }
                if config['orbit'] == 'Horizontal Lyapunov Orbits':
                    data = pielm_cgs.generate_data(config, parameters)
                elif config['orbit'] == 'Halo Orbits':
                    parameters['ORBIT_TYPE'] = 'Halo Orbits'
                    data = pielm_cgs.generate_data(config, parameters)
                else:
                    parameters['ORBIT_TYPE'] = 'Vertical Lyapunov Orbits'
                    data = pielm_cgs.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_cgs.run(data, config, parameters)

            elif dynamics == 'CR3BP' and observer == 'GROUND' and optimizer == 'CONSTRAINED_BASIN_HOPPING':
                import PIELM_basinhopping_w_range_cr3bp as pielm_cgcb
                parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 15000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': 1e1, 'LAMBDA_DIST':1e1, 'STEPSIZE': np.power(10, float(1)),
                              'NUMBER_OF_ITERATIONS': 100, 'TEMPERATURE': 1e-6, 'WEIGHT_SCALE_FACTOR': 1e-3,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000, 'TARGET_ACCEPT_RATE': 0.5, 'STEPWISE_FACTOR': 0.9,
                              'G_TOLERANCE': 1e-15, 'MAX_NFEV': 100, 'A_PERT': 1000, 'ECC_PERT': 0.2,
                              'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_idx,
                              'MIN_RHO': 1e-4, 'MAX_RHO': 1e-1, 'MIN_RHO_DOT': -1.01e0, 'MAX_RHO_DOT': 1.01e0,
                              'DELTA_RHO': 1e-2, 'DELTA_RHO_DOT': 0.067, 'INITIAL_TRAJECTORIES': 1}
                config['lambda'] = parameters['STEPSIZE']
                config['run_idx'] = run_idx
                if config['orbit'] == 'Horizontal Lyapunov Orbits':
                    data = pielm_cgcb.generate_data(config, parameters)
                elif config['orbit'] == 'Halo Orbits':
                    parameters['ORBIT_TYPE'] = 'Halo Orbits'
                    data = pielm_cgcb.generate_data(config, parameters)
                else:
                    parameters['ORBIT_TYPE'] = 'Vertical Lyapunov Orbits'
                    data = pielm_cgcb.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_cgcb.run(
                    data, config, parameters)

            elif dynamics == 'NBD' and observer == 'SPACE' and optimizer == 'NLLS':
                import PIELM_nlls_tbo_space_nbody as pielm_nstn
                parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "gaussian",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 15000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': 1e-3, 'STEPSIZE': 10, 'NUMBER_OF_ITERATIONS': 500, 'TEMPERATURE': 1000,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000, 'WEIGHT_SCALE_FACTOR': 1e-7,
                              'G_TOLERANCE': 1e-15, 'MAX_NFEV': 1000,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_idx}
                config['lambda'] = parameters['PHYSICS_WEIGHT']
                config['run_idx'] = run_idx
                data = pielm_nstn.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_nstn.run(data, config, parameters)

            elif dynamics == 'NBD' and observer == 'SPACE' and optimizer == 'SGD':
                import PIELM_sgd_tbo_space_nbody as pielm_nsts
                parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "lhs",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 100000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': 1e-11, 'STEPSIZE': 10, 'NUMBER_OF_ITERATIONS': 500, 'TEMPERATURE': 1000,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000, 'WEIGHT_SCALE_FACTOR': 1e-4,
                              'G_TOLERANCE': 1e-15, 'MAX_NFEV': 1000,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_idx}
                config['lambda'] = parameters['PHYSICS_WEIGHT']
                config['run_idx'] = run_idx
                data = pielm_nsts.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_nsts.run(data, config, parameters)


            elif dynamics == 'NBD' and observer == 'SPACE' and optimizer == 'CONSTRAINED_BASIN_HOPPING':
                import PIELM_basinhopping_w_range_nbody as pielm_ctsn
                parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "gaussian",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 15000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': 1e2, 'LAMBDA_DIST': 1e-2, 'WEIGHT_SCALE_FACTOR': 1e-2, 'STEPSIZE': 1e0,
                              'NUMBER_OF_ITERATIONS': 100, 'TEMPERATURE': 1e-6,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000, 'TARGET_ACCEPT_RATE': 0.5, 'STEPWISE_FACTOR': 0.9,
                              'G_TOLERANCE': 1e-15, 'MAX_NFEV':100,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_idx,
                              'MIN_RHO': 1e-4, 'MAX_RHO': 1e-1, 'MIN_RHO_DOT': -1.01e0, 'MAX_RHO_DOT': 1.01e0,
                              'DELTA_RHO': 1e-2, 'DELTA_RHO_DOT': 0.067, 'INITIAL_TRAJECTORIES':10}
                config['lambda'] = parameters['TIME_DELTA']
                config['run_idx'] = run_idx
                data = pielm_ctsn.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_ctsn.run(data, config, parameters)

            else:
                print("Error specifying")

            # c3bp uses non-dim time directly
            if dynamics == 'CR3BP':
                epochs_data = data[2]
            else:
                epochs_data = [time.tdb.jd for time in data[2]]

            # visualize
            data_for_df = {"EPOCH(JDTDB)": epochs_data, "GEO_X(KM)": data[3][:, 0],
                           "GEO_Y(KM)": data[3][:, 1], "GEO_Z(KM)": data[3][:, 2], "GEO_VX(KM/S)": data[4][:, 0],
                           "GEO_VY(KM/S)": data[4][:, 1], "GEO_VZ(KM/S)": data[4][:, 2],
                           "SC_GEO_X(KM)": data[1][:, 0],
                           "SC_GEO_Y(KM)": data[1][:, 1], "SC_GEO_Z(KM)": data[1][:, 2],
                           "SC_GEO_VX(KM/S)": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VY(KM/S)": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VZ(KM/S)": np.zeros_like(data[1][:, 1]), 'SIN_RA': data[0][0],
                           'COS_RA': data[0][1],
                           'SIN_DEC': data[0][2], "SC_GEO_X(KM)_PHYS": data[1][:, 0],
                           "SC_GEO_Y(KM)_PHYS": data[1][:, 1],
                           "SC_GEO_Z(KM)_PHYS": data[1][:, 2], "SC_GEO_VX(KM/S)_PHYS": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VY(KM/S)_PHYS": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VZ(KM/S)_PHYS": np.zeros_like(data[1][:, 1]), 'SIN_RA_PHYS': data[0][0],
                           'COS_RA_PHYS': data[0][1], 'SIN_DEC_PHYS': data[0][2]}

            file_used = data[9]
            file_name = config['dynamics'] + '_' + config['orbit'] + '_' + config['observer'] + '_' + \
                        config['optimizer'] + '_run_' + str(run_idx) + '.csv'

            os.makedirs(config['error_file_dir'], exist_ok=True)
            file_path = os.path.join(config['error_file_dir'], file_name)
            rmse_df = util.generate_iod_file(file_path, final_pos, final_vel, true_pos, true_vel, epochs)

            # Position RMSE (IOD)
            pos_rmse = np.sqrt(((rmse_df[["IOD_X", "IOD_Y", "IOD_Z"]].values -
                                 rmse_df[["TRUE_X", "TRUE_Y", "TRUE_Z"]].values) ** 2).mean())

            # Position RMSE (IOD_NLLS)
            pos_rmse_nlls = np.sqrt(((rmse_df[["IOD_X_NLLS", "IOD_Y_NLLS", "IOD_Z_NLLS"]].values -
                                      rmse_df[["TRUE_X", "TRUE_Y", "TRUE_Z"]].values) ** 2).mean())

            # Velocity RMSE (IOD)
            vel_rmse = np.sqrt(((rmse_df[["IOD_VX", "IOD_VY", "IOD_VZ"]].values -
                                 rmse_df[["TRUE_VX", "TRUE_VY", "TRUE_VZ"]].values) ** 2).mean())

            # Velocity RMSE (IOD_NLLS)
            vel_rmse_nlls = np.sqrt(((rmse_df[["IOD_VX_NLLS", "IOD_VY_NLLS", "IOD_VZ_NLLS"]].values -
                                      rmse_df[["TRUE_VX", "TRUE_VY", "TRUE_VZ"]].values) ** 2).mean())

            # Add to parameters
            parameters['POS_RMSE'] = pos_rmse
            parameters['POS_RMSE_NLLS'] = pos_rmse_nlls
            parameters['VEL_RMSE'] = vel_rmse
            parameters['VEL_RMSE_NLLS'] = vel_rmse_nlls
            parameters['COMPUTATION_TIME'] = comp_time
            parameters['FILE_USED'] = file_used
            parameters['SAVED_AS'] = file_name
            local_master.append(parameters)

            df = pd.DataFrame(data_for_df)
            # util.iod_viz(df, results, positions, velocities, nlls_start, config, rmse_df)

    # Convert local list to DataFrame
    df_local = pd.DataFrame(local_master)

    # Gather all DataFrames at rank 0
    dfs = comm.gather(df_local, root=0)

    if rank == 0:
        # Concatenate all into one DataFrame
        meta_file_name = config['dynamics'] + '_' + config['orbit'] + '_' + config['observer'] + '_' + \
                    config['optimizer'] + 'meta_data.csv'
        df_global = pd.concat(dfs, ignore_index=True)
        df_global.to_csv(meta_file_name, index=False)

    return


def run_IOD_hyperparameter_run_par_resumable(config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("Starting...")

    viz_flag = bool(config.get('visualization_flag', 0))

    # --- Hyperparam grids (don't materialize huge lists more than needed) ---
    p1 = config['physics_weight']
    p2 = config['c_wb']
    p3 = config.get('range_weight', [])
    optimizer = config['optimizer']

    if optimizer == 'CONSTRAINED_BASIN_HOPPING':
        combo_iter = itertools.product(p1, p2, p3)
    else:
        combo_iter = itertools.product(p1, p2)

    n_runs = int(config['n_runs_test'])

    # --- Helpers ----------------------------------------------------------------
    def task_stream():
        """Yield (combo, run_idx) without ever building a giant list."""
        idx = 0
        for combo in combo_iter:
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
            float(row.get("PHYSICS_WEIGHT", 0) or 0),
            float(row.get("WEIGHT_SCALE_FACTOR", 0) or 0),
            float(row.get("LAMBDA_DIST", 0) or 0),
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
        pw = float(combo[0])
        wsf = float(combo[1]) if len(combo) > 1 else 0.0
        ld = float(combo[2]) if len(combo) > 2 else 0.0
        return (pw, wsf, ld, int(run_idx))

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
            parameters = {
                'NUMBER_OF_OBSERVATIONS': 16,
                'TIME_DELTA': 0.5 * u.day,
                'TOTAL_POINTS': 100,
                'SAMPLING_METHOD': "uniform",
                'LAYER_RATIOS': [(0., 1/3), (1/3, 2/3), (2/3, 1.)],
                'INPUT_RANGE': (-1, 1),
                'HIDDEN_DIMENSION': 100,
                'PHYSICS_WEIGHT': combo[0],
                'LAMBDA_DIST': combo[2],
                'WEIGHT_SCALE_FACTOR': combo[1],
                'NUMBER_OF_ITERATIONS': 2,
                'TEMPERATURE': 1e-6,
                'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15,
                'MAX_FUNCTION_EVAL': 1000, 'MAX_ITERATiONS': 1000,
                'G_TOLERANCE': 1e-15, 'RUN_NUMBER': run_idx,
                'MIN_RHO': 0.0006684587122, 'MAX_RHO': 0.06684587122,
                'MIN_RHO_DOT': -0.05033557046, 'MAX_RHO_DOT': 0.05033557046,
                'DELTA_RHO': 0.00334229356, 'DELTA_RHO_DOT': 0.00335570469
            }
            if viz_flag:
                config['lambda'] = parameters['TIME_DELTA']
                config['run_idx'] = run_idx
            data = pielm_ctsn.generate_data(config, parameters)
            results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_ctsn.run(data, config, parameters)

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
                    row[k] = json.dumps(v if not isinstance(v, np.ndarray) else v.tolist())
                    continue

                # everything else as string
                row[k] = str(v)

            # Ensure RUN_NUMBER and RUN_IDX exist and are ints
            row["RUN_NUMBER"] = int(parameters.get("RUN_NUMBER", run_idx))

            # Attach extras/metrics (already numeric/strings)
            row.update(extras)
            return row

        def build_extras(pos_rmse, vel_rmse, comp_time, file_used, file_name):
            return {
                "POS_RMSE": float(pos_rmse),
                "VEL_RMSE": float(vel_rmse),
                "COMPUTATION_TIME_SEC": float(comp_time),
                "FILE_USED": str(file_used),
                "SAVED_AS": str(file_name),
            }

        extras = build_extras(pos_rmse, vel_rmse, comp_time, file_used, file_name)
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


def run_IOD_hyperparameter_run_par(config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    viz_flag = config['visualization_flag']

    # --- Read config-provided grids ---
    param1_values = config['physics_weight']
    param2_values = config['c_wb']
    param3_values = config['range_weight']

    # Build hyperparameter combos
    if config['optimizer'] == 'CONSTRAINED_BASIN_HOPPING':
        hyperparam_combos = list(itertools.product(param1_values, param2_values, param3_values))
    else:
        hyperparam_combos = list(itertools.product(param1_values, param2_values))

    n_runs = int(config['n_runs_test'])

    # === NEW: build the combined task list over (combo, run_idx) ===
    # Use product to cover all run repeats for each combo
    tasks = list(itertools.product(hyperparam_combos, range(n_runs)))

    # Round-robin assignment across ranks
    local_tasks = [t for i, t in enumerate(tasks) if i % size == rank]

    local_master = []

    # --- Work loop ---
    for (combo, run_idx) in local_tasks:
        if run_idx == n_runs - 1:
            print(f"[Rank {rank}] Running hyperparameter combo {combo}, run={run_idx + 1}")

        run_number_orb = 0.


        # get case
        dynamics, orbit, observer, optimizer = config['dynamics'], config['orbit'], config['observer'], config['optimizer']
        if dynamics == '2BD' and orbit == 'GEO' and observer == 'GROUND' and optimizer == 'SGD':
            import PIELM_sgd_geo_earth as pielm_2gg
            parameters = {'NUMBER_OF_OBSERVATIONS': 260, 'OBSERVATION_TIME_FRACTION': 0.5,
                          'TIME_DELTA': 0.0000000001 * u.day,
                          'TOTAL_POINTS': 300, 'SAMPLING_METHOD': "uniform",
                          'LAYER_RATIOS': [(0., 1 / 10000), (1 / 10000, 9999 / 10000), (9999 / 10000, 1.)],
                          'INPUT_RANGE': (-1, 1),
                          'HIDDEN_DIMENSION': 20, 'NUMBER_OF_EPOCHS': 100000, 'LEARNING_RATE': 1e-2,
                          'PHYSICS_WEIGHT': 1e2, 'STEPSIZE': 1, 'NUMBER_OF_ITERATIONS': 50, 'TEMPERATURE': 1,
                          'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_NFEV': 100,
                          'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                          'ANOM_PERT': 15.}
            config['lambda'] = parameters['PHYSICS_WEIGHT']
            data = pielm_2gg.generate_data(config, parameters)
            results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_2gg.run(
                data, config, parameters)

        elif dynamics == '2BD' and orbit == 'GEO' and observer == 'GROUND' and optimizer == 'NLLS':
            import  PIELM_nlls_geo_earth as pielm_2ggn
            parameters = {'NUMBER_OF_OBSERVATIONS': 260, 'OBSERVATION_TIME_FRACTION': 0.5,
                          'TIME_DELTA': 0.0000000001 * u.day,
                          'TOTAL_POINTS': 300, 'SAMPLING_METHOD': "uniform",
                          'LAYER_RATIOS': [(0., 1 / 10000), (1 / 10000, 9999 / 10000), (9999 / 10000, 1.)],
                          'INPUT_RANGE': (-1, 1),
                          'HIDDEN_DIMENSION': 20, 'NUMBER_OF_EPOCHS': 50000, 'LEARNING_RATE': 1e-1,
                          'PHYSICS_WEIGHT': 1e0, 'STEPSIZE': 1, 'NUMBER_OF_ITERATIONS': 50, 'TEMPERATURE': 1,
                          'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_NFEV': 100,
                          'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                          'ANOM_PERT': 15.}
            config['lambda'] = parameters['PHYSICS_WEIGHT']
            data = pielm_2ggn.generate_data(config, parameters)
            results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_2ggn.run(
                data, config, parameters)

        elif dynamics == '2BD' and orbit=='GEO' and observer == 'GROUND' and optimizer == 'CONSTRAINED_BASIN_HOPPING':
            import PIELM_basinhopping_w_range_geo_earth as pielm_2ggcb
            parameters = {'NUMBER_OF_OBSERVATIONS': 260, 'OBSERVATION_TIME_FRACTION': 0.5,
                          'TIME_DELTA': 0.0000000001 * u.day,
                          'TOTAL_POINTS': 300, 'SAMPLING_METHOD': "uniform",
                          'LAYER_RATIOS': [(0., 1 / 10000), (1 / 10000, 9999 / 10000), (9999 / 10000, 1.)],
                          'INPUT_RANGE': (-1, 1),
                          'HIDDEN_DIMENSION': 20, 'NUMBER_OF_EPOCHS': 50000, 'LEARNING_RATE': 1e-1,
                          'PHYSICS_WEIGHT': 1e2, 'LAMBDA_DIST': 1e-7, 'STEPSIZE': 10e-5,
                          'NUMBER_OF_ITERATIONS': 100, 'TEMPERATURE': 10e-8,
                          'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                          'MAX_ITERATiONS': 20000, 'G_TOLERANCE': 1e-15, 'MAX_NFEV':100,
                          'A_PERT': 0., 'ECC_PERT': 0., 'INC_PERT': 0., 'RAAN_PERT': 0., 'ARGPER_PERT': 0.,
                          'ANOM_PERT': 0., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': 95 - run_idx,
                          'MIN_RHO': 1.0976e0, 'MAX_RHO': 2.35157e2, 'MIN_RHO_DOT': -1.2647e0,
                          'MAX_RHO_DOT': 1.2647e0, 'DELTA_RHO_STEP': 1.5679e0, 'DELTA_RHO_DOT_STEP': 1.265e-1,
                          'INITIAL_TRAJECTORIES': 1}
            config['lambda'] = parameters['PHYSICS_WEIGHT']
            config['run_idx'] = run_idx
            data = pielm_2ggcb.generate_data(config, parameters)
            results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_2ggcb.run(data, config, parameters)

        elif dynamics == 'CR3BP' and observer == 'GROUND' and optimizer == 'NLLS':
            import  PIELM_nlls_periodicorbits_earth_cr3bp as pielm_cgn
            if run_idx % 3 == 0:
               run_number_orb = 95 - 8 * (run_idx % 10)
            parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                          'TIME_DELTA': 0.5 * u.day,
                          'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                          'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                          'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 15000, 'LEARNING_RATE': 1e-1,
                          'PHYSICS_WEIGHT': combo[0], 'STEPSIZE': 10, 'NUMBER_OF_ITERATIONS': 500, 'TEMPERATURE': 1000,
                          'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                          'MAX_ITERATiONS': 20000, 'WEIGHT_SCALE_FACTOR': combo[1],
                          'G_TOLERANCE': 1e-15, 'MAX_NFEV': 1000,
                          'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                          'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_number_orb}
            if run_idx % 3 == 0:
                data = pielm_cgn.generate_data(config, parameters)
            elif run_idx % 3 == 1:
                parameters['ORBIT_TYPE'] = 'Halo Orbits'
                data = pielm_cgn.generate_data(config, parameters)
            else:
                parameters['ORBIT_TYPE'] = 'Vertical Lyapunov Orbits'
                data = pielm_cgn.generate_data(config, parameters)
            results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_cgn.run(data, config, parameters)

        elif dynamics == 'CR3BP' and observer == 'GROUND' and optimizer == 'SGD':
            import  PIELM_sgd_periodicorbits_earth_cr3bp as pielm_cgs
            if run_idx % 3 == 0:
                run_number_orb = 95 - 8 * (run_idx % 10)
            parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5, 'TIME_DELTA': 0.5 * u.day,
                          'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                          'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                          'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 100000, 'LEARNING_RATE': 1e-2,
                          'PHYSICS_WEIGHT': combo[0], 'WEIGHT_SCALE_FACTOR': combo[1],
                          'STEPSIZE': 1, 'NUMBER_OF_ITERATIONS': 50, 'TEMPERATURE': 1,
                          'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'G_TOLERANCE': 1e-15,
                          'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                          'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_number_orb
                           }
            if run_idx % 3 == 0:
                data = pielm_cgs.generate_data(config, parameters)
            elif run_idx % 3 == 1:
                parameters['ORBIT_TYPE'] = 'Halo Orbits'
                data = pielm_cgs.generate_data(config, parameters)
            else:
                parameters['ORBIT_TYPE'] = 'Vertical Lyapunov Orbits'
                data = pielm_cgs.generate_data(config, parameters)
            results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_cgs.run(data, config, parameters)

        elif dynamics == 'CR3BP' and observer == 'GROUND' and optimizer == 'CONSTRAINED_BASIN_HOPPING':
            import PIELM_basinhopping_w_range_cr3bp as pielm_cgcb
            if run_idx % 3 == 0:
               run_number_orb = 95 - 8 * (run_idx % 10)
            parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                          'TIME_DELTA': 0.5 * u.day,
                          'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                          'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                          'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 15000, 'LEARNING_RATE': 1e-1,
                          'PHYSICS_WEIGHT': combo[0], 'LAMBDA_DIST':combo[2], 'STEPSIZE': np.power(10, float(1)),
                          'NUMBER_OF_ITERATIONS': 10, 'TEMPERATURE': np.power(10, float(-4)),
                          'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                          'MAX_ITERATiONS': 20000, 'TARGET_ACCEPT_RATE': 0.5, 'STEPWISE_FACTOR': 0.9,
                          'G_TOLERANCE': 1e-15, 'MAX_NFEV': 100, 'WEIGHT_SCALE_FACTOR': combo[1],
                          'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                          'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_number_orb,
                          'MIN_RHO': 1e-4, 'MAX_RHO': 1e-1, 'MIN_RHO_DOT': -1.01e0, 'MAX_RHO_DOT': 1.01e0,
                          'DELTA_RHO': 1e-2, 'DELTA_RHO_DOT': 0.067, 'INITIAL_TRAJECTORIES': 1}
            config['lambda'] = parameters['STEPSIZE']
            config['run_idx'] = run_idx
            if run_idx % 3 == 0:
                data = pielm_cgcb.generate_data(config, parameters)
            elif run_idx % 3 == 1:
                parameters['ORBIT_TYPE'] = 'Halo Orbits'
                data = pielm_cgcb.generate_data(config, parameters)
            else:
                parameters['ORBIT_TYPE'] = 'Vertical Lyapunov Orbits'
                data = pielm_cgcb.generate_data(config, parameters)
            results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_cgcb.run(
                data, config, parameters)

        elif dynamics == 'NBD' and observer == 'SPACE' and optimizer == 'NLLS':
            import PIELM_nlls_tbo_space_nbody as pielm_nstn
            parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                          'TIME_DELTA': 0.5 * u.day,
                          'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                          'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                          'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 15000, 'LEARNING_RATE': 1e-1,
                          'PHYSICS_WEIGHT': combo[0], 'WEIGHT_SCALE_FACTOR': combo[1],
                          'STEPSIZE': 10, 'NUMBER_OF_ITERATIONS': 500, 'TEMPERATURE': 1000,
                          'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                          'MAX_ITERATiONS': 20000, 'G_TOLERANCE': 1e-15, 'MAX_NFEV': 1000,
                          'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                          'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_idx}
            config['lambda'] = parameters['PHYSICS_WEIGHT']
            config['run_idx'] = run_idx
            data = pielm_nstn.generate_data(config, parameters)
            results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_nstn.run(data, config, parameters)

        elif dynamics == 'NBD' and observer == 'SPACE' and optimizer == 'SGD':
            import PIELM_sgd_tbo_space_nbody as pielm_nsts
            parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                          'TIME_DELTA': 0.5 * u.day,
                          'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                          'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                          'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 100000, 'LEARNING_RATE': 1e-2,
                          'PHYSICS_WEIGHT': combo[0], 'WEIGHT_SCALE_FACTOR': combo[1],
                          'STEPSIZE': 10, 'NUMBER_OF_ITERATIONS': 500, 'TEMPERATURE': 1000,
                          'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                          'MAX_ITERATiONS': 20000,
                          'G_TOLERANCE': 1e-15, 'MAX_NFEV': 1000,
                          'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                          'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_idx}
            config['lambda'] = parameters['PHYSICS_WEIGHT']
            config['run_idx'] = run_idx
            data = pielm_nsts.generate_data(config, parameters)
            results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_nsts.run(data, config, parameters)

        elif dynamics == 'NBD' and observer == 'SPACE' and optimizer == 'CONSTRAINED_BASIN_HOPPING':
            import PIELM_basinhopping_w_range_nbody as pielm_ctsn
            parameters = {'NUMBER_OF_OBSERVATIONS': 16, 'TIME_DELTA': 0.5 * u.day,
                          'TOTAL_POINTS': 500, 'SAMPLING_METHOD': "uniform",
                          'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                          'HIDDEN_DIMENSION': 500,
                          'PHYSICS_WEIGHT': combo[0], 'LAMBDA_DIST': combo[2], 'WEIGHT_SCALE_FACTOR': combo[1],
                          'NUMBER_OF_ITERATIONS': 500, 'TEMPERATURE': 1e-6,
                          'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 1000,
                          'MAX_ITERATiONS': 1000, 'G_TOLERANCE': 1e-15, 'RUN_NUMBER': run_idx,
                          'MIN_RHO': 0.0006684587122, 'MAX_RHO': 0.06684587122,
                          'MIN_RHO_DOT':-0.05033557046, 'MAX_RHO_DOT': 0.05033557046,
                          'DELTA_RHO': 0.00334229356, 'DELTA_RHO_DOT': 0.00335570469}
            config['lambda'] = parameters['TIME_DELTA']
            config['run_idx'] = run_idx
            data = pielm_ctsn.generate_data(config, parameters)
            results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_ctsn.run(data, config, parameters)

        else:
            print("Error specifying")

        def combo_to_string(combo):
            return "_" + "_".join(str(x) for x in combo)
        combo_str = combo_to_string(combo)
        file_used = data[9]
        file_name = config['dynamics'] + '_' + config['orbit'] + '_' + config['observer'] + '_' + \
                    config['optimizer'] + '_run_' + str(run_idx) + combo_str + '.csv'

        os.makedirs(config['error_file_dir'], exist_ok=True)
        file_path = os.path.join(config['error_file_dir'], file_name)
        rmse_df = util.generate_iod_file(file_path, final_pos, final_vel, true_pos, true_vel, epochs)

        # Position RMSE (IOD)
        pos_rmse = np.sqrt(((rmse_df[["IOD_X", "IOD_Y", "IOD_Z"]].values -
                             rmse_df[["TRUE_X", "TRUE_Y", "TRUE_Z"]].values) ** 2).mean())

        # Position RMSE (IOD_NLLS)
        pos_rmse_nlls = np.sqrt(((rmse_df[["IOD_X_NLLS", "IOD_Y_NLLS", "IOD_Z_NLLS"]].values -
                                  rmse_df[["TRUE_X", "TRUE_Y", "TRUE_Z"]].values) ** 2).mean())

        # Velocity RMSE (IOD)
        vel_rmse = np.sqrt(((rmse_df[["IOD_VX", "IOD_VY", "IOD_VZ"]].values -
                             rmse_df[["TRUE_VX", "TRUE_VY", "TRUE_VZ"]].values) ** 2).mean())

        # Velocity RMSE (IOD_NLLS)
        vel_rmse_nlls = np.sqrt(((rmse_df[["IOD_VX_NLLS", "IOD_VY_NLLS", "IOD_VZ_NLLS"]].values -
                                  rmse_df[["TRUE_VX", "TRUE_VY", "TRUE_VZ"]].values) ** 2).mean())

        # Add to parameters
        parameters['POS_RMSE'] = pos_rmse
        parameters['POS_RMSE_NLLS'] = pos_rmse_nlls
        parameters['VEL_RMSE'] = vel_rmse
        parameters['VEL_RMSE_NLLS'] = vel_rmse_nlls
        parameters['COMPUTATION_TIME'] = comp_time
        parameters['FILE_USED'] = file_used
        parameters['SAVED_AS'] = file_name
        local_master.append(parameters)

        if viz_flag:
            # c3bp uses non-dim time directly
            if dynamics == 'CR3BP':
                epochs_data = data[2]
            else:
                epochs_data = [time.tdb.jd for time in data[2]]

            # visualize
            data_for_df = {"EPOCH(JDTDB)": epochs_data, "GEO_X(KM)": data[3][:, 0],
                           "GEO_Y(KM)": data[3][:, 1], "GEO_Z(KM)": data[3][:, 2], "GEO_VX(KM/S)": data[4][:, 0],
                           "GEO_VY(KM/S)": data[4][:, 1], "GEO_VZ(KM/S)": data[4][:, 2],
                           "SC_GEO_X(KM)": data[1][:, 0],
                           "SC_GEO_Y(KM)": data[1][:, 1], "SC_GEO_Z(KM)": data[1][:, 2],
                           "SC_GEO_VX(KM/S)": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VY(KM/S)": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VZ(KM/S)": np.zeros_like(data[1][:, 1]), 'SIN_RA': data[0][0],
                           'COS_RA': data[0][1],
                           'SIN_DEC': data[0][2], "SC_GEO_X(KM)_PHYS": data[1][:, 0],
                           "SC_GEO_Y(KM)_PHYS": data[1][:, 1],
                           "SC_GEO_Z(KM)_PHYS": data[1][:, 2], "SC_GEO_VX(KM/S)_PHYS": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VY(KM/S)_PHYS": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VZ(KM/S)_PHYS": np.zeros_like(data[1][:, 1]), 'SIN_RA_PHYS': data[0][0],
                           'COS_RA_PHYS': data[0][1], 'SIN_DEC_PHYS': data[0][2]}

            df = pd.DataFrame(data_for_df)
            util.iod_viz(df, results, positions, velocities, nlls_start, config, rmse_df)

    # Convert local list to DataFrame
    df_local = pd.DataFrame(local_master)

    # Gather all DataFrames at rank 0
    dfs = comm.gather(df_local, root=0)

    if rank == 0:
        # Concatenate all into one DataFrame
        meta_file_name = config['dynamics'] + '_' + config['orbit'] + '_' + config['observer'] + '_' + \
                    config['optimizer'] + 'meta_data.csv'
        df_global = pd.concat(dfs, ignore_index=True)
        df_global.to_csv(meta_file_name, index=False)

    return


def run_IOD_hyperparameter(config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Example hyperparameters ---
    param1_values = config['physics_weight']
    param2_values = config['c_wb']
    param3_values = config['range_weight']

    # Create all hyperparameter combinations
    if config['optimizer'] == 'CONSTRAINED_BASIN_HOPPING':
        hyperparam_combos = list(itertools.product(param1_values, param2_values, param3_values))
    else:
        hyperparam_combos = list(itertools.product(param1_values, param2_values))    

    n_combos = len(hyperparam_combos)

    # Each rank gets a subset of hyperparameter combos (round-robin)
    local_combos = [combo for i, combo in enumerate(hyperparam_combos) if i % size == rank]

    # Number of repeated runs per combination
    n_runs = config['n_runs_test']

    local_master = []

    # --- Work loop ---
    for combo in local_combos:
        print(f"[Rank {rank}] Running hyperparameter combo {combo}")

        run_number_orb = 0.
        for run_idx in range(n_runs):

            # get case
            dynamics, orbit, observer, optimizer = config['dynamics'], config['orbit'], config['observer'], config['optimizer']
            if dynamics == '2BD' and orbit == 'GEO' and observer == 'GROUND' and optimizer == 'SGD':
                import PIELM_sgd_geo_earth as pielm_2gg
                parameters = {'NUMBER_OF_OBSERVATIONS': 260, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.0000000001 * u.day,
                              'TOTAL_POINTS': 300, 'SAMPLING_METHOD': "uniform",
                              'LAYER_RATIOS': [(0., 1 / 10000), (1 / 10000, 9999 / 10000), (9999 / 10000, 1.)],
                              'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 20, 'NUMBER_OF_EPOCHS': 100000, 'LEARNING_RATE': 1e-2,
                              'PHYSICS_WEIGHT': 1e2, 'STEPSIZE': 1, 'NUMBER_OF_ITERATIONS': 50, 'TEMPERATURE': 1,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_NFEV': 100,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15.}
                config['lambda'] = parameters['PHYSICS_WEIGHT']
                data = pielm_2gg.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_2gg.run(
                    data, config, parameters)

            elif dynamics == '2BD' and orbit == 'GEO' and observer == 'GROUND' and optimizer == 'NLLS':
                import  PIELM_nlls_geo_earth as pielm_2ggn
                parameters = {'NUMBER_OF_OBSERVATIONS': 260, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.0000000001 * u.day,
                              'TOTAL_POINTS': 300, 'SAMPLING_METHOD': "uniform",
                              'LAYER_RATIOS': [(0., 1 / 10000), (1 / 10000, 9999 / 10000), (9999 / 10000, 1.)],
                              'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 20, 'NUMBER_OF_EPOCHS': 50000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': 1e0, 'STEPSIZE': 1, 'NUMBER_OF_ITERATIONS': 50, 'TEMPERATURE': 1,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_NFEV': 100,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15.}
                config['lambda'] = parameters['PHYSICS_WEIGHT']
                data = pielm_2ggn.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_2ggn.run(
                    data, config, parameters)

            elif dynamics == '2BD' and orbit=='GEO' and observer == 'GROUND' and optimizer == 'CONSTRAINED_BASIN_HOPPING':
                import PIELM_basinhopping_w_range_geo_earth as pielm_2ggcb
                parameters = {'NUMBER_OF_OBSERVATIONS': 260, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.0000000001 * u.day,
                              'TOTAL_POINTS': 300, 'SAMPLING_METHOD': "uniform",
                              'LAYER_RATIOS': [(0., 1 / 10000), (1 / 10000, 9999 / 10000), (9999 / 10000, 1.)],
                              'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 20, 'NUMBER_OF_EPOCHS': 50000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': 1e2, 'LAMBDA_DIST': 1e-7, 'STEPSIZE': 10e-5,
                              'NUMBER_OF_ITERATIONS': 100, 'TEMPERATURE': 10e-8,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000, 'G_TOLERANCE': 1e-15, 'MAX_NFEV':100,
                              'A_PERT': 0., 'ECC_PERT': 0., 'INC_PERT': 0., 'RAAN_PERT': 0., 'ARGPER_PERT': 0.,
                              'ANOM_PERT': 0., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': 95 - run_idx,
                              'MIN_RHO': 1.0976e0, 'MAX_RHO': 2.35157e2, 'MIN_RHO_DOT': -1.2647e0,
                              'MAX_RHO_DOT': 1.2647e0, 'DELTA_RHO_STEP': 1.5679e0, 'DELTA_RHO_DOT_STEP': 1.265e-1,
                              'INITIAL_TRAJECTORIES': 1}
                config['lambda'] = parameters['PHYSICS_WEIGHT']
                config['run_idx'] = run_idx
                data = pielm_2ggcb.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_2ggcb.run(data, config, parameters)

            elif dynamics == 'CR3BP' and observer == 'GROUND' and optimizer == 'NLLS':
                import  PIELM_nlls_periodicorbits_earth_cr3bp as pielm_cgn
                if run_idx % 3 == 0:
                   run_number_orb = 95 - 8 * (run_idx % 10)
                parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 15000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': combo[0], 'STEPSIZE': 10, 'NUMBER_OF_ITERATIONS': 500, 'TEMPERATURE': 1000,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000, 'WEIGHT_SCALE_FACTOR': combo[1],
                              'G_TOLERANCE': 1e-15, 'MAX_NFEV': 1000,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_number_orb}
                if run_idx % 3 == 0:
                    data = pielm_cgn.generate_data(config, parameters)
                elif run_idx % 3 == 1:
                    parameters['ORBIT_TYPE'] = 'Halo Orbits'
                    data = pielm_cgn.generate_data(config, parameters)
                else:
                    parameters['ORBIT_TYPE'] = 'Vertical Lyapunov Orbits'
                    data = pielm_cgn.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_cgn.run(data, config, parameters)

            elif dynamics == 'CR3BP' and observer == 'GROUND' and optimizer == 'SGD':
                import  PIELM_sgd_periodicorbits_earth_cr3bp as pielm_cgs
                if run_idx % 3 == 0:
                    run_number_orb = 95 - 8 * (run_idx % 10)
                parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5, 'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 100000, 'LEARNING_RATE': 1e-2,
                              'PHYSICS_WEIGHT': combo[0], 'WEIGHT_SCALE_FACTOR': combo[1],
                              'STEPSIZE': 1, 'NUMBER_OF_ITERATIONS': 50, 'TEMPERATURE': 1,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'G_TOLERANCE': 1e-15,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_number_orb
                               }
                if run_idx % 3 == 0:
                    data = pielm_cgs.generate_data(config, parameters)
                elif run_idx % 3 == 1:
                    parameters['ORBIT_TYPE'] = 'Halo Orbits'
                    data = pielm_cgs.generate_data(config, parameters)
                else:
                    parameters['ORBIT_TYPE'] = 'Vertical Lyapunov Orbits'
                    data = pielm_cgs.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_cgs.run(data, config, parameters)

            elif dynamics == 'CR3BP' and observer == 'GROUND' and optimizer == 'CONSTRAINED_BASIN_HOPPING':
                import PIELM_basinhopping_w_range_cr3bp as pielm_cgcb
                if run_idx % 3 == 0:
                   run_number_orb = 95 - 8 * (run_idx % 10)
                parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 15000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': combo[0], 'LAMBDA_DIST':combo[2], 'STEPSIZE': np.power(10, float(1)),
                              'NUMBER_OF_ITERATIONS': 10, 'TEMPERATURE': np.power(10, float(-4)),
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000, 'TARGET_ACCEPT_RATE': 0.5, 'STEPWISE_FACTOR': 0.9,
                              'G_TOLERANCE': 1e-15, 'MAX_NFEV': 100, 'WEIGHT_SCALE_FACTOR': combo[1],
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_number_orb,
                              'MIN_RHO': 1e-4, 'MAX_RHO': 1e-1, 'MIN_RHO_DOT': -1.01e0, 'MAX_RHO_DOT': 1.01e0,
                              'DELTA_RHO': 1e-2, 'DELTA_RHO_DOT': 0.067, 'INITIAL_TRAJECTORIES': 1}
                config['lambda'] = parameters['STEPSIZE']
                config['run_idx'] = run_idx
                if run_idx % 3 == 0:
                    data = pielm_cgcb.generate_data(config, parameters)
                elif run_idx % 3 == 1:
                    parameters['ORBIT_TYPE'] = 'Halo Orbits'
                    data = pielm_cgcb.generate_data(config, parameters)
                else:
                    parameters['ORBIT_TYPE'] = 'Vertical Lyapunov Orbits'
                    data = pielm_cgcb.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_cgcb.run(
                    data, config, parameters)

            elif dynamics == 'NBD' and observer == 'SPACE' and optimizer == 'NLLS':
                import PIELM_nlls_tbo_space_nbody as pielm_nstn
                parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 15000, 'LEARNING_RATE': 1e-1,
                              'PHYSICS_WEIGHT': combo[0], 'WEIGHT_SCALE_FACTOR': combo[1],
                              'STEPSIZE': 10, 'NUMBER_OF_ITERATIONS': 500, 'TEMPERATURE': 1000,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000, 'G_TOLERANCE': 1e-15, 'MAX_NFEV': 1000,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_idx}
                config['lambda'] = parameters['PHYSICS_WEIGHT']
                config['run_idx'] = run_idx
                data = pielm_nstn.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_nstn.run(data, config, parameters)

            elif dynamics == 'NBD' and observer == 'SPACE' and optimizer == 'SGD':
                import PIELM_sgd_tbo_space_nbody as pielm_nsts
                parameters = {'NUMBER_OF_OBSERVATIONS': 10, 'OBSERVATION_TIME_FRACTION': 0.5,
                              'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 100, 'SAMPLING_METHOD': "uniform",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 100, 'NUMBER_OF_EPOCHS': 100000, 'LEARNING_RATE': 1e-2,
                              'PHYSICS_WEIGHT': combo[0], 'WEIGHT_SCALE_FACTOR': combo[1],
                              'STEPSIZE': 10, 'NUMBER_OF_ITERATIONS': 500, 'TEMPERATURE': 1000,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000,
                              'G_TOLERANCE': 1e-15, 'MAX_NFEV': 1000,
                              'A_PERT': 1000, 'ECC_PERT': 0.2, 'INC_PERT': 15., 'RAAN_PERT': 15., 'ARGPER_PERT': 15.,
                              'ANOM_PERT': 15., 'ORBIT_TYPE': 'Horizontal Lyapunov Orbits', 'RUN_NUMBER': run_idx}
                config['lambda'] = parameters['PHYSICS_WEIGHT']
                config['run_idx'] = run_idx
                data = pielm_nsts.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_nsts.run(data, config, parameters)

            elif dynamics == 'NBD' and observer == 'SPACE' and optimizer == 'CONSTRAINED_BASIN_HOPPING':
                import PIELM_basinhopping_w_range_nbody as pielm_ctsn
                parameters = {'NUMBER_OF_OBSERVATIONS': 16, 'TIME_DELTA': 0.5 * u.day,
                              'TOTAL_POINTS': 500, 'SAMPLING_METHOD': "uniform",
                              'LAYER_RATIOS': [(0., 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.)], 'INPUT_RANGE': (-1, 1),
                              'HIDDEN_DIMENSION': 500,
                              'PHYSICS_WEIGHT': combo[0], 'LAMBDA_DIST': combo[2], 'WEIGHT_SCALE_FACTOR': combo[1],
                              'NUMBER_OF_ITERATIONS': 500, 'TEMPERATURE': 1e-6,
                              'X_TOLERANCE': 1e-15, 'F_TOLERANCE': 1e-15, 'MAX_FUNCTION_EVAL': 20000,
                              'MAX_ITERATiONS': 20000, 'G_TOLERANCE': 1e-15, 'RUN_NUMBER': run_idx,
                              'MIN_RHO': 0.0006684587122, 'MAX_RHO': 0.06684587122,
                              'MIN_RHO_DOT':-0.05033557046, 'MAX_RHO_DOT': 0.05033557046,
                              'DELTA_RHO': 0.00334229356, 'DELTA_RHO_DOT': 0.00335570469}
                config['lambda'] = parameters['TIME_DELTA']
                config['run_idx'] = run_idx
                data = pielm_ctsn.generate_data(config, parameters)
                results, positions, velocities, nlls_start, final_pos, final_vel, true_pos, true_vel, epochs, comp_time = pielm_ctsn.run(data, config, parameters)

            else:
                print("Error specifying")

            # c3bp uses non-dim time directly
            if dynamics == 'CR3BP':
                epochs_data = data[2]
            else:
                epochs_data = [time.tdb.jd for time in data[2]]

            # visualize
            data_for_df = {"EPOCH(JDTDB)": epochs_data, "GEO_X(KM)": data[3][:, 0],
                           "GEO_Y(KM)": data[3][:, 1], "GEO_Z(KM)": data[3][:, 2], "GEO_VX(KM/S)": data[4][:, 0],
                           "GEO_VY(KM/S)": data[4][:, 1], "GEO_VZ(KM/S)": data[4][:, 2],
                           "SC_GEO_X(KM)": data[1][:, 0],
                           "SC_GEO_Y(KM)": data[1][:, 1], "SC_GEO_Z(KM)": data[1][:, 2],
                           "SC_GEO_VX(KM/S)": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VY(KM/S)": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VZ(KM/S)": np.zeros_like(data[1][:, 1]), 'SIN_RA': data[0][0],
                           'COS_RA': data[0][1],
                           'SIN_DEC': data[0][2], "SC_GEO_X(KM)_PHYS": data[1][:, 0],
                           "SC_GEO_Y(KM)_PHYS": data[1][:, 1],
                           "SC_GEO_Z(KM)_PHYS": data[1][:, 2], "SC_GEO_VX(KM/S)_PHYS": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VY(KM/S)_PHYS": np.zeros_like(data[1][:, 1]),
                           "SC_GEO_VZ(KM/S)_PHYS": np.zeros_like(data[1][:, 1]), 'SIN_RA_PHYS': data[0][0],
                           'COS_RA_PHYS': data[0][1], 'SIN_DEC_PHYS': data[0][2]}

            def combo_to_string(combo):
                return "_" + "_".join(str(x) for x in combo)
            combo_str = combo_to_string(combo)
            file_used = data[9]
            file_name = config['dynamics'] + '_' + config['orbit'] + '_' + config['observer'] + '_' + \
                        config['optimizer'] + '_run_' + str(run_idx) + combo_str + '.csv'

            os.makedirs(config['error_file_dir'], exist_ok=True)
            file_path = os.path.join(config['error_file_dir'], file_name)
            rmse_df = util.generate_iod_file(file_path, final_pos, final_vel, true_pos, true_vel, epochs)

            # Position RMSE (IOD)
            pos_rmse = np.sqrt(((rmse_df[["IOD_X", "IOD_Y", "IOD_Z"]].values -
                                 rmse_df[["TRUE_X", "TRUE_Y", "TRUE_Z"]].values) ** 2).mean())

            # Position RMSE (IOD_NLLS)
            pos_rmse_nlls = np.sqrt(((rmse_df[["IOD_X_NLLS", "IOD_Y_NLLS", "IOD_Z_NLLS"]].values -
                                      rmse_df[["TRUE_X", "TRUE_Y", "TRUE_Z"]].values) ** 2).mean())

            # Velocity RMSE (IOD)
            vel_rmse = np.sqrt(((rmse_df[["IOD_VX", "IOD_VY", "IOD_VZ"]].values -
                                 rmse_df[["TRUE_VX", "TRUE_VY", "TRUE_VZ"]].values) ** 2).mean())

            # Velocity RMSE (IOD_NLLS)
            vel_rmse_nlls = np.sqrt(((rmse_df[["IOD_VX_NLLS", "IOD_VY_NLLS", "IOD_VZ_NLLS"]].values -
                                      rmse_df[["TRUE_VX", "TRUE_VY", "TRUE_VZ"]].values) ** 2).mean())

            # Add to parameters
            parameters['POS_RMSE'] = pos_rmse
            parameters['POS_RMSE_NLLS'] = pos_rmse_nlls
            parameters['VEL_RMSE'] = vel_rmse
            parameters['VEL_RMSE_NLLS'] = vel_rmse_nlls
            parameters['COMPUTATION_TIME'] = comp_time
            parameters['FILE_USED'] = file_used
            parameters['SAVED_AS'] = file_name
            local_master.append(parameters)

            df = pd.DataFrame(data_for_df)
            # util.iod_viz(df, results, positions, velocities, nlls_start, config, rmse_df)

    # Convert local list to DataFrame
    df_local = pd.DataFrame(local_master)

    # Gather all DataFrames at rank 0
    dfs = comm.gather(df_local, root=0)

    if rank == 0:
        # Concatenate all into one DataFrame
        meta_file_name = config['dynamics'] + '_' + config['orbit'] + '_' + config['observer'] + '_' + \
                    config['optimizer'] + 'meta_data.csv'
        df_global = pd.concat(dfs, ignore_index=True)
        df_global.to_csv(meta_file_name, index=False)

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
run_IOD_hyperparameter_run_par_resumable(config)  # par over combos and runs, visualization flag, can continue from where you left off

###################################
# Single results file implementation
####################################
# a_sc_res = run_sim_runnumbers_MPI(master, config)
# if rank == 0:
#     a_res = a_sc_res[0]
#     a_scs = a_sc_res[1]
#     flat_data = []
#     for idx, process in enumerate(a_res):
#         for jdx, run in enumerate(process):
#             for kdx, entry in enumerate(run):
#                 index = tuple(entry[0])  # (run_number, object_id, spacecraft_number)
#                 values = entry[1]  # List of values over time
#                 flat_data.append((*index, values, a_scs[idx][jdx]))

# Convert to DataFrame
# df = pd.DataFrame(flat_data, columns=["run_number", "object_id", "spacecraft_number", "values", "spacecraft_1_ini_pos"])

# Set MultiIndex
# df.set_index(["run_number", "object_id", "spacecraft_number"], inplace=True)

# df.to_csv(config['output_df_file_name'], sep=',', header=True, index=True)


##############################
# Run parallel for number of runs using multiprocessor
##############################
# run_sim_runnumbers_partial = partial(run_sim_runnumbers, minimoon_master=master, config=config)
# run_numbers = np.arange(1, config['number_of_runs'] + 1, 1)
# pool = multiprocessing.Pool(processes=config['number_of_process'])
# results = pool.map(run_sim_runnumbers_partial, run_numbers)

# Flatten the data into lists of tuples
# flat_data = []
# for run in results:
#     for entry in run:
#         index = tuple(entry[0])  # (run_number, object_id, spacecraft_number)
#         values = entry[1]  # List of values over time
#         flat_data.append((*index, values))

# Convert to DataFrame
# df = pd.DataFrame(flat_data, columns=["run_number", "object_id", "spacecraft_number", "values"])

# Set MultiIndex
# df.set_index(["run_number", "object_id", "spacecraft_number"], inplace=True)


#######################################
# iterate over minimoons in parallel for a single run using multiprocessor - partially implemented
########################################

# run_sim_minimoons_partial = partial(run_sim_minimoons, minimoon_master=master, config=config)

# parallel implementation
# pool = multiprocessing.Pool(processes=1)
# results = pool.map(run_sim_minimoons_partial, master['Object id'])
# pool.close()
