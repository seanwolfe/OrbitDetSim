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


def run_sim_viz_minimoons(object_id, minimoon_master, config):

    # declare asteroid
    current_minimoon_master = minimoon_master[minimoon_master['Object id'] == object_id]
    current_minimoon = Asteroid(object_id, current_minimoon_master['Min_SunEarthL1_V_index'], config)

    # declare formation
    formation = Formation(config)

    # determine when an initial detection will be made
    sc_visible = util.viz(current_minimoon, formation, config)

    return sc_visible


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
            flat_data.append((*[run_number, current_minimoon.id, jdx + 1], tuple(visible), tuple(formation.spacecraft[0].ini_position)))

            if idx != 0:
                memeory_offload.append((*[run_number, current_minimoon.id, jdx + 1], tuple(visible), tuple(formation.spacecraft[0].ini_position)))


        # save file
        if idx == 0:
            # Convert to DataFrame
            df = pd.DataFrame(flat_data, columns=["run_number", "object_id", "spacecraft_number", "values",
                                                  "spacecraft_1_ini_pos"])

            # Set MultiIndex
            df.set_index(["run_number", "object_id", "spacecraft_number"], inplace=True)

            df.to_csv(config['output_df_file_name'] + '_run_' + str(run_number) + '.csv', sep=',', header=True, index=True)

        elif idx > 0 and (idx % 5 == 0 or idx % len(minimoon_master['Object id']) > 0):
            # Convert to DataFrame
            df = pd.DataFrame(memeory_offload, columns=["run_number", "object_id", "spacecraft_number", "values",
                                                  "spacecraft_1_ini_pos"])

            # Set MultiIndex
            df.set_index(["run_number", "object_id", "spacecraft_number"], inplace=True)

            df.to_csv(config['output_df_file_name'] + '_run_' + str(run_number) + '.csv', mode='a', index=True, header=False)

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
        run_sim_runnumbers(idx + 1, minimoon_master, config)

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


def run_sim_runnumbers_MPI_getIOD_data(minimoon_master, config):

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
        # read file of visible
        folder = config['visible_files_folder'] + '_' + str(config['num_spacecraft'])
        file_name = ('spacecraft_' + str(config['num_spacecraft']) + '_runs_' + str(config['number_of_runs']) +
                     '_run_' + str(idx + 1) + '.csv')
        full_path = folder + '/' + file_name
        run_data = util.read_master(full_path, config)

        # Create a mask to filter nonnegative values
        run_data["min_nonnegative"] = run_data["values"].apply(lambda x: min([y for y in x if y >= 0]) if np.any(np.array(x) >= 0) else np.nan)

        # Find the spacecraft with the minimum value for each object_id
        detected_pop = run_data[~np.isnan(run_data["min_nonnegative"])]
        missed_pop = run_data[np.isnan(run_data["min_nonnegative"])]

        detected_appended_pop = util.get_sc_state_from_sc1_position(detected_pop, config)

        # re integrate according to exposure time and slew time to get 16 samples
        for jdx, detected_minimoon in detected_appended_pop.iterrows():
            print(detected_minimoon.name)
            file_path = config['minimoon_files_folder'] + detected_minimoon.name[1] + '.csv'
            orbit = pd.read_csv(file_path, sep=' ', header=0, names=config['minimoon_column_names'])

            # Add an asteroid (near Earth)
            asteroid_state = orbit.loc[
                detected_minimoon['min_nonnegative'], ['Helio x', 'Helio y', 'Helio z', 'Helio vx', 'Helio vy',
                                      'Helio vz']].values  # au and au/d
            asteroid_state[:3] *= config['AU_TO_M'] / config['KM_TO_M']  # to match spice
            asteroid_state[3:] *= (config['AU_TO_M'] / config['KM_TO_M'] / config['SECONDS_PER_DAY'])

            spacecraft_state = detected_minimoon[['HELIO_X_(km)', 'HELIO_Y_(km)', 'HELIO_Z_(km)', 'HELIO_Vx_(km/s)', 'HELIO_Vy_(km/s)',
                  'HELIO_Vz_(km/s)']].to_numpy()

            epoch = orbit.loc[detected_minimoon['min_nonnegative'], 'Julian Date']

            object_states =  [asteroid_state, spacecraft_state]

            integrated_states = nbody.integrate_n_body(object_states, epoch, 10 * config['SECONDS_PER_DAY'], 3600)

            ####################################
            # Comparison
            ####################################

            asteroid_x = integrated_states[0, 0, :]
            asteroid_y = integrated_states[0, 1, :]
            asteroid_z = integrated_states[0, 2, :]
            asteroid_vx = integrated_states[0, 3, :]
            asteroid_vy = integrated_states[0, 4, :]
            asteroid_vz = integrated_states[0, 5, :]

            # Plot results
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot(asteroid_x / config['AU_TO_M'], asteroid_y / config['AU_TO_M'], asteroid_z / config['AU_TO_M'],
                    'r-',
                    label="Integrated", linewidth=3, zorder=5)
            ax.scatter(asteroid_x[0] / config['AU_TO_M'], asteroid_y[0] / config['AU_TO_M'],
                       asteroid_z[0] / config['AU_TO_M'],
                       'g',
                       label="Start", s=10, zorder=15)
            ax.plot(orbit['Helio x'], orbit['Helio y'], orbit['Helio z'], 'b', label="Openorb", linewidth=1, zorder=10)
            ax.set_xlabel("X Position (au)")
            ax.set_ylabel("Y Position (au)")
            ax.set_zlabel("Z Position (au)")
            plt.legend()

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(projection='3d')
            ax2.plot(asteroid_vx / config['AU_TO_M'] * config['SECONDS_PER_DAY'],
                     asteroid_vy / config['AU_TO_M'] * config['SECONDS_PER_DAY'],
                     asteroid_vz / config['AU_TO_M'] * config['SECONDS_PER_DAY'], 'r-', label="Integrated", linewidth=3,
                     zorder=5)
            ax2.plot(orbit['Helio vx'], orbit['Helio vy'], orbit['Helio vz'], 'b', label="Openorb vel", linewidth=1,
                     zorder=10)
            ax2.set_xlabel("X Velocity (au/d)")
            ax2.set_ylabel("Y Velocity (au/d)")
            ax2.set_zlabel("Z Velocity (au/d)")
            plt.legend()

            plt.show()



        # calc ra and dec

        # generate output file with epoch , ast xyz vxvyvz, detecting sc id xyz vxvyvz RA Dec
        # file name: run-x_minimoon-id-y_sc-id-z_index_k.csv

    return


###########################
# run sim
##########################

# Load YAML config file
with open("orbit_det_configuration.yaml", "r") as file:
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

run_sim_runnumbers_MPI_getIOD_data(master, config)

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





