import torch
import yaml
import sys
import os
import pandas as pd
from Asteroid import Asteroid
from Formation import Formation
import numpy as np
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import ast
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import spiceypy as sp
import utilities as util
import n_body_integrator as nbody
import argparse
import PIELM_sgd as pielm
from PIELM_sgd import ELM

# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
sp.furnsh("de430.bsp")
sp.furnsh('naif0012.tls')


def run_sim_viz_minimoons(object_id, minimoon_master, config):
    # declare asteroid
    current_minimoon_master = minimoon_master[minimoon_master['Object id'] == object_id]
    current_minimoon = Asteroid(object_id, current_minimoon_master['Min_SunEarthL1_V_index'], config)

    # declare formation
    formation = Formation(config)

    # determine when an initial detection will be made
    sc_visible = util.viz(current_minimoon, formation, config)

    return sc_visible


def run_sim_runnumbers_new(run_number, minimoon_master, config):
    df_buffer = pd.DataFrame(columns=["run_number", "object_id", "spacecraft_number", "values", "total_length", "spacecraft_1_ini_pos"])
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

            df_buffer = pd.DataFrame(columns=["run_number", "object_id", "spacecraft_number", "values", "total_length", "spacecraft_1_ini_pos"])
            part_number += 1

    return


def run_sim_runnumbers(run_number, minimoon_master, config):
    # declare asteroid
    memeory_offload = []
    for idx, master_i in minimoon_master.iterrows():
        flat_data = []

        current_minimoon = Asteroid(master_i['Object id'], master_i['Min_SunEarthL1_V_index'], config)

        # declare formation
        formation = Formation(config)

        # asteroid position
        asteroid_pos = current_minimoon.orbit.loc[:, ['Synodic x', 'Synodic y', 'Synodic z']].values
        earth_pos = np.zeros_like(asteroid_pos)

        moon_pos = current_minimoon.orbit.loc[:, ['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']].values

        # get spacecraft positions over trajectory
        formation.match_spacecraft_trajectory(len(asteroid_pos[:, 0]), config)

        for jdx, spacecraft in enumerate(formation.spacecraft):

            print(run_number, current_minimoon.id, jdx + 1)

            sc_pos = spacecraft.matched_trajectory
            # find when the asteroid is in fov and not ocluded by earth or moon
            visible = spacecraft.asteroid_in_fov_batch(asteroid_pos, sc_pos, earth_pos, moon_pos, config)
            flat_data.append((*[run_number, current_minimoon.id, jdx + 1], tuple(visible),
                              tuple(formation.spacecraft[0].ini_position)))

            if idx != 0:
                memeory_offload.append((*[run_number, current_minimoon.id, jdx + 1], tuple(visible),
                                        tuple(formation.spacecraft[0].ini_position)))

        # save file
        if idx == 0:
            # Convert to DataFrame
            df = pd.DataFrame(flat_data, columns=["run_number", "object_id", "spacecraft_number", "values",
                                                  "spacecraft_1_ini_pos"])

            # Set MultiIndex
            df.set_index(["run_number", "object_id", "spacecraft_number"], inplace=True)

            df.to_csv(config['output_df_file_name'] + '_run_' + str(run_number) + '.csv', sep=',', header=True,
                      index=True)

        elif idx > 0 and (idx % 5 == 0 or idx % len(minimoon_master['Object id']) > 0):
            # Convert to DataFrame
            df = pd.DataFrame(memeory_offload, columns=["run_number", "object_id", "spacecraft_number", "values",
                                                        "spacecraft_1_ini_pos"])

            # Set MultiIndex
            df.set_index(["run_number", "object_id", "spacecraft_number"], inplace=True)

            df.to_csv(config['output_df_file_name'] + '_run_' + str(run_number) + '.csv', mode='a', index=True,
                      header=False)

            memeory_offload = []
        else:
            pass

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
        run_sim_runnumbers_new(idx + 1, minimoon_master, config)

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


def run_sim_runnumbers_MPI_getIOD_data(config):
    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Master (rank 0) gathers the list of all files ---
    if rank == 0:
        # all_files = util.get_all_files(config['visible_files_folder'], config['save_format'])
        all_files = util.get_all_files_run_number(config['visible_files_folder'], config['save_format'], config['run_number'])
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

            # spacecraft ###############
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
                    new_asteroid_state_helio[2, :], new_asteroid_state_helio[3, :], new_asteroid_state_helio[4, :],
                    new_asteroid_state_helio[5, :], new_spacecraft_state_helio[0, :], new_spacecraft_state_helio[1, :],
                    new_spacecraft_state_helio[2, :], new_spacecraft_state_helio[3, :], new_spacecraft_state_helio[4, :],
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
                util.viz(spacecraft_state[:3, :], asteroid_state[:3, :], current_minimoon, formation, [sin_ra, cos_ra, sin_dec], config)
                ################

    return


def run_sim_runnumbers_MPI_getIOD_data_new(config):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        all_files_per_folder = util.get_files_per_folder(config['visible_files_folder'], config['save_format'])

        # Flatten and tag each file with folder index for balance tracking
        tagged_files = []
        for folder_idx, folder_files in enumerate(all_files_per_folder):
            for f in folder_files:
                tagged_files.append((folder_idx, f))

        # Round-robin assignment
        assignments = [[] for _ in range(size)]
        for i, (_, f) in enumerate(tagged_files):
            assignments[i % size].append(f)
    else:
        assignments = None

        # Scatter assignments
    local_files = comm.scatter(assignments, root=0)

    # --- Distribute work: each rank gets a subset ---
    for i, file_i in enumerate(local_files):
        print(f"Rank {rank}: " + file_i.split('/')[-1].split('.')[0])

        run_data = util.read_master(file_i, config)
        # Create a mask to filter nonnegative values
        run_data["min_nonnegative"] = run_data["values"].apply(
            lambda x: min(x) if np.any(np.array(x) >= 0) else np.nan)

        # Find the spacecraft with the minimum value for each object_id
        detected_pop = run_data[~np.isnan(run_data["min_nonnegative"])]

        config['num_spacecraft'] = int(file_i.split('/')[-1].split('.')[0].split('_')[1])
        detected_appended_pop = util.get_sc_state_from_sc1_position(detected_pop, config)

        print(f"Rank {rank} computed appended population")

        # re integrate according to exposure time and slew time to get 16 samples
        for jdx, detected_minimoon in detected_appended_pop.iterrows():

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

            # spacecraft ###############
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
                    new_asteroid_state_helio[2, :], new_asteroid_state_helio[3, :], new_asteroid_state_helio[4, :],
                    new_asteroid_state_helio[5, :], new_spacecraft_state_helio[0, :], new_spacecraft_state_helio[1, :],
                    new_spacecraft_state_helio[2, :], new_spacecraft_state_helio[3, :], new_spacecraft_state_helio[4, :],
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
                util.viz(spacecraft_state[:3, :], asteroid_state[:3, :], current_minimoon, formation, [sin_ra, cos_ra, sin_dec], config)
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

        # read data
        iod_data = util.read_IOD_data(file_path, config)

        # add noise in quadrature - assuming independent
        sigma_ra = np.sqrt(config['sigma_ra'] ** 2 + config['sigma_pointing'] ** 2) / config['MAS_TO_DEGREE']
        sigma_dec = np.sqrt(config['sigma_dec'] ** 2 + config['sigma_pointing'] ** 2) / config['MAS_TO_DEGREE']
        iod_data_w_noise = util.add_noise_to_angles(iod_data, sigma_ra, sigma_dec)

        # observation epochs
        obs_e = iod_data_w_noise['EPOCH(JDTDB)'].values
        # spacecraft position at observation
        spacecraft_pos = iod_data_w_noise.loc[:, ['SC_HELIO_X(AU)', 'SC_HELIO_Y(AU)', 'SC_HELIO_Z(AU)']].values
        # observations - without noise
        obs = iod_data.loc[:, ['SIN_RA', 'COS_RA', 'SIN_DEC']].values
        # observations - wit noise
        # obs = iod_data_w_noise.loc[:, ['SIN_RA', 'COS_RA', 'SIN_DEC']].values


        # other relevant parameters
        delta = 10
        num_points = 100
        layer_ratios = [(0., 1/3), (1/3, 2/3), (2/3, 1.)]
        # mean = obs_e[0] + (obs_e[-1] - obs_e[0]) / 2
        # std = (obs_e[-1] - obs_e[0]) * 4
        z_range = (-1, 1)
        method = "lhs"
        hidden_dim = 200

        # generate collocation points
        colloc_points = pielm.sample_time_points(method, obs_e, delta, num_points, layer_ratios=layer_ratios, config=config)

        # Step 1: Build a mask for which collocation points are in the observation epochs
        obs_mask = np.isin(colloc_points, obs_e)

        # Step 2: Get the indices of observation epochs in colloc_points
        obs_indices = np.where(obs_mask)[0]

        # normalize epochs (inputs)
        epochs_nd_norm, c = pielm.epoch_normalization(colloc_points, z_range, config)


        # as a 2D tensor
        epochs_nd_norm_reshaped_tensor = torch.tensor(epochs_nd_norm, dtype=torch.float32).unsqueeze(1)

        # declare pielm
        elm = ELM(hidden_dim, c_normalization=c)

        # train pielm
        pielm.train(elm, epochs_nd_norm_reshaped_tensor, obs, obs_indices, spacecraft_pos, obs_e, colloc_points, config)

        # make data

        raise NotImplementedError

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

# run_sim_runnumbers_MPI_getIOD_data(config)

###################################
# Run IOD simulation in parallel
###################################

run_IOD_MPI(config)



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
