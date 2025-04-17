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

            # asteroid ##############
            asteroid_state = orbit.loc[
                detected_minimoon['min_nonnegative'], ['Helio x', 'Helio y', 'Helio z', 'Helio vx', 'Helio vy',
                                      'Helio vz']].values  # au and au/d
            asteroid_state[:3] *= config['AU_TO_M'] / config['KM_TO_M']  # to match spice
            asteroid_state[3:] *= (config['AU_TO_M'] / config['KM_TO_M'] / config['SECONDS_PER_DAY'])

            asteroid_epoch = orbit.loc[detected_minimoon['min_nonnegative'], 'Julian Date']

            # integrate s/c traj
            asteroid_integrated_states, asteroid_earth_states = nbody.integrate_n_body(asteroid_state, asteroid_epoch,
                                                                     1 * config['SECONDS_PER_DAY'], 3600,
                                                                     type="ASTEROID")

            ##########

            # spacecraft ###############
            spacecraft_state_geo = detected_minimoon[['GEO_ECLIP_X_(km)', 'GEO_ECLIP_Y_(km)', 'GEO_ECLIP_Z_(km)', 'GEO_ECLIP_Vx_(km/s)', 'GEO_ECLIP_Vy_(km/s)',
                  'GEO_ECLIP_Vz_(km/s)']].to_numpy()
            spacecraft_epoch = detected_minimoon['sc_epoch']

            # convert to heliocentric for integration
            sun_geo_state = sp.spkgeo(10, sp.str2et(spacecraft_epoch), "ECLIPJ2000", 399)[0]
            spacecraft_state_helio = spacecraft_state_geo - sun_geo_state

            # integrate s/c traj
            integrated_states, earth_states = nbody.integrate_n_body(spacecraft_state_helio, spacecraft_epoch, 1 * config['SECONDS_PER_DAY'], 3600, type="SPACECRAFT")

            integrated_states_aud = util.ms_to_aud(integrated_states)
            earth_states_aud = util.ms_to_aud(earth_states)

            # convert integrated s/c traj to sun-earth co in au and day
            spacecraft_state = util.eclip_to_sun_earth_corotating_batch_n_body_integrator_output(integrated_states_aud, -earth_states_aud)

            ###
            # does not seem right for velocity
            ###
            spacecraft_state2 = util.helio_eclip_to_sun_earth_corotating_batch_full(integrated_states_aud, earth_states_aud)

            from scipy.signal import savgol_filter
            deriv = savgol_filter(spacecraft_state[0], window_length=11, polyorder=3, deriv=1, delta= 3600 / 86400, axis=1)
            print(spacecraft_state)
            print(deriv)
            print(spacecraft_state2)


            # convert back to helio based on asteroid epoch

            #############

            # calc ra and dec from helio


            ####
            # comparison
            ####

            # convert asteroid state to sun-earth-co

            # calc ra and dec from sun-earth-co



            # spacecraft_state[3:] /= (config['AU_TO_M'] / config['KM_TO_M'] / config['SECONDS_PER_DAY'])
            # earth_helio_state[:3] /= (config['AU_TO_M'] / config['KM_TO_M'])  # to match spice
            # earth_helio_state[3:] /= (config['AU_TO_M'] / config['KM_TO_M'] / config['SECONDS_PER_DAY'])

            #######################
            # for visualization
            ###################
            # get the original s/c orbit and get the original s/c starting points
            # create a formation object, it has s/c s randomly placed
            # formation = Formation(config)

            # we already had a saved formation, saved according to the s/c 1 position, get the correspoding index in overall orbit file
            # sc1_ini_index = formation.get_index_from_pos(detected_minimoon['spacecraft_1_ini_pos'])

            # re-initialize formation with this index
            # formation.recall_formation(sc1_ini_index, config)

            # match the spacecraft trajectories to that of the asteroid in terms of length and sampling (asteroid sampled at one hour)
            # formation.match_spacecraft_trajectory(len(detected_minimoon['values']), config)
            # current_minimoon = Asteroid(detected_minimoon.name[1], 100, config)

            # visualize it all
            # util.viz(spacecraft_pos, current_minimoon, formation, config)
            ################



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





