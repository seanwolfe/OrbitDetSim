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
import spiceypy as spice
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import argparse


# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
sp.furnsh("../de430.bsp")
sp.furnsh('../naif0012.tls')


def integrate_n_body_multi(object_states, epoch, end_time, time_interval):

    # Argument parser to get the config file path
    parser = argparse.ArgumentParser(description="Run the spacecraft simulation")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    epoch_et = spice.unitim(epoch, 'JDTDB', 'ET')  # initial epoch

    # Function to get state vectors (position, velocity) in km & km/s
    def get_state(body, reference=10):
        state, _ = spice.spkgeo(body, epoch_et, "ECLIPJ2000", reference)
        return np.array(state)

    # Get Sun & planets' initial states
    bodies = [10, 1, 2, 399, 4, 5, 6, 7, 8,
              301]  # ["SUN", "MERCURY", "VENUS", "EARTH", "JUPITER", "SATURN", "URANUS", "NEPTUNE", "MOON"]
    planet_states = {body: get_state(body) for body in bodies}

    # Combine all bodies into a state vector
    initial_states = np.vstack([planet_states[body] for body in bodies] + object_states)
    initial_positions = initial_states[:, :3]
    initial_velocities = initial_states[:, 3:]

    # Convert km, km/s to meters, meters/s
    initial_positions *= config['KM_TO_M']
    initial_velocities *= config['KM_TO_M']

    # Flatten initial state vector (for integration)
    y0 = np.hstack([initial_positions.flatten(), initial_velocities.flatten()])

    # Define masses (in kg) for Sun, planets, and Moon
    G = config['GRAVITATIONAL_CONSTANT']  # m^3 kg^-1 s^-2
    masses = {
        10: config['SUN_MASS'], 1: config['MERCURY_MASS'], 2: config['VENUS_MASS'],
        399: config['EARTH_MASS'], 4: config['MARS_MASS'], 5: config['JUPITER_MASS'], 6: config['SATURN_MASS'],
        7: config['URANUS_MASS'], 8: config['NEPTUNE_MASS'], 301: config['MOON_MASS'],
        "ASTEROID": config['asteroid_mass'], "SPACECRAFT": config['spacecraft_mass']  # Arbitrary mass
    }
    mass_array = np.array([masses[body] for body in bodies] + [masses["ASTEROID"]] + [masses["SPACECRAFT"] for i in object_states[1:]])

    start_time = 0
    t_span = (start_time, end_time)  # Start at t=0, end at t=900s
    t_eval = np.linspace(start_time, end_time, time_interval)  # 30s intervals

    # Define N-body equations of motion
    def nbody_derivatives(t, y):
        n = len(mass_array)
        positions = y[:3 * n].reshape((n, 3))
        velocities = y[3 * n:].reshape((n, 3))
        accelerations = np.zeros((n, 3))

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    accelerations[i] += G * mass_array[j] * r_vec / r_mag ** 3

        return np.hstack([velocities.flatten(), accelerations.flatten()])

    # Solve the N-body problem
    sol = solve_ivp(nbody_derivatives, t_span, y0, method="RK45", t_eval=t_eval)

    # Extract asteroid's trajectory
    n_bodies = len(mass_array)
    n_spacecraft = len(object_states) - 1
    asteroid_idx = n_bodies - n_spacecraft - 1  # asteroid index
    sc_idx = n_bodies - 1


    earth_positions = sol.y[9:12, :]
    earth_velocities = sol.y[3 * (n_bodies + 3), :]
    object_positions = sol.y[3 * asteroid_idx:3 * (sc_idx + 1), :]
    object_velocities = sol.y[2 * 3 * (asteroid_idx + 1): 2 * 3 * (sc_idx + 1), :]

    M = object_positions.shape[0] // 3  # Number of objects
    N = object_positions.shape[1]  # Number of time steps

    # Reshape to (M, 3, N), where 3 corresponds to x, y, z
    object_pos_reshaped = object_positions.reshape(M, 3, N)  # (M, 3, N)
    object_vel_reshaped = object_velocities.reshape(M, 3, N)  # (M, 3, N)

    return np.hstack((object_pos_reshaped, object_vel_reshaped)), np.vstack((earth_positions, earth_velocities))


def run_sim_runnumbers_MPI_getIOD_data_byfile_helio(config):
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


def run_sim_viz_minimoons(object_id, minimoon_master, config):
    # declare asteroid
    current_minimoon_master = minimoon_master[minimoon_master['Object id'] == object_id]
    current_minimoon = Asteroid(object_id, current_minimoon_master['Min_SunEarthL1_V_index'], config)

    # declare formation
    formation = Formation(config)

    # determine when an initial detection will be made
    sc_visible = util.viz(current_minimoon, formation, config)

    return sc_visible