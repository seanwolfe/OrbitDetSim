import yaml
import sys
import os
sys.path.append(os.path.expanduser('~/Documents/sean/minimoon_integrations'))
from MM_Parser import MmParser
from Asteroid import Asteroid
from Formation import Formation
import numpy as np
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def calc_start_index(minimoon, sc_formation, configs):

    # asteroid position
    asteroid_pos = minimoon.orbit.loc[:, ['Synodic x', 'Synodic y', 'Synodic z']].values
    earth_pos = np.zeros_like(asteroid_pos)
    print(minimoon.id)
    moon_pos = minimoon.orbit.loc[:, ['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']].values

    # get spacecraft positions over trajectory
    sc_formation.match_spacecraft_trajectory(len(asteroid_pos[:, 0]), configs)

    # start index is first instance asteroid is FOV of a sc, without occlusion from Earth or moon
    # it is the index in the minimoon trajectory corresponding to this
    for i, spacecraft in enumerate(sc_formation.spacecraft):

        sc_pos = spacecraft.matched_trajectory

        #####
        # test to see if all trajectories look fine - and they do
        ####
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(sc_pos[:, 0], sc_pos[:, 1], sc_pos[:, 2], label='SC')
        ax.scatter(sc_pos[0, 0], sc_pos[0, 1], sc_pos[0, 2], s=20)
        ax.plot(moon_pos[:, 0], moon_pos[:, 1], moon_pos[:, 2], label='Moon')
        ax.plot(asteroid_pos[:, 0], asteroid_pos[:, 1], asteroid_pos[:, 2], label='Asteroid')
        ax.scatter(0.009, 0, 0, label='L_1', s=20)
        # Create a sphere (Earth model)
        theta = np.linspace(0, np.pi, 30)  # Latitude
        phi = np.linspace(0, 2 * np.pi, 60)  # Longitude
        theta, phi = np.meshgrid(theta, phi)
        #
        # Earth radius (approx. in arbitrary units)
        R = 6378  # Normalize radius

        # Convert spherical to Cartesian coordinates
        x = R * np.sin(theta) * np.cos(phi) / (configs['AU_TO_M'] / 1000)  # km
        y = R * np.sin(theta) * np.sin(phi) / (configs['AU_TO_M'] / 1000)
        z = R * np.cos(theta) / (configs['AU_TO_M'] / 1000)

        # Plot wireframe Earth
        ax.plot_wireframe(x, y, z, color="blue", linewidth=0.5, alpha=0.7)

        ax.set_xlabel('X (au)')
        ax.set_ylabel('Y (au)')
        ax.set_zlabel('Z (au)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
        ax.legend()
        ax.set_aspect('equal')
        plt.show()


        # find when the asteroid is in fov and not ocluded by earth or moon
        # visible = spacecraft.asteroid_in_fov_batch(asteroid_pos, sc_pos, earth_pos, moon_pos, configs)


def calc_end_index(minimoon, sc_formaiton, configs):
    # start index is first instance asteroid is FOV of a sc, without occlusion from Earth or moon
    # it is the first time
    raise NotImplementedError


def eclip_to_sun_earth_corotating_batch(minimoon_df):
    """
    Converts a batch of positions and velocities from the heliocentric ECLIPJ2000 frame
    to the Sun-Earth co-rotating frame.

    Parameters:
    - positions_eclip (numpy array): Nx3 array of positions in ECLIPJ2000 (AU).
    - et_times (numpy array): N-element array of ephemeris times.

    Returns:
    - positions_corotating (numpy array): Nx3 array of positions in Sun-Earth co-rotating frame (AU).
    """

    moon_pos = minimoon_df.loc[:, ["Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)"]].to_numpy()
    earth_positions = minimoon_df.loc[:, ["Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)"]].to_numpy()

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angles = np.arctan2(earth_positions[:, 1], earth_positions[:, 0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angles = np.cos(-angles)
    sin_angles = np.sin(-angles)

    # Construct rotation matrices (shape: Nx3x3)
    rotation_matrices = np.zeros((len(angles), 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # No rotation in the Z direction

    # Apply the rotation to transform positions
    positions_corotating = np.einsum("nij,nj->ni", rotation_matrices, moon_pos - earth_positions)

    old_file_convention = positions_corotating.copy()
    old_file_convention[:, :2] *= -1

    minimoon_df[['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']] = old_file_convention
    file_path = '/media/aeromec/Seagate Desktop Drive/minimoon_files_oorb/' + str(minimoon_df['Object id'].iloc[0]) + '.csv'
    minimoon_df.to_csv(file_path, sep=' ', header=True, index=False)

    return positions_corotating


def run_sim(object_id, minimoon_master, config):

    # declare asteroid
    current_minimoon_master = minimoon_master[minimoon_master['Object id'] == object_id]
    current_minimoon = Asteroid(object_id, current_minimoon_master['Min_SunEarthL1_V_index'], config)

    # declare formation
    formation = Formation(config)

    # determine when an initial detection will be made
    start_index = calc_start_index(current_minimoon, formation, config)

    # determine when formation will stop detecting on initial window end index
    # end_index = calc_end_index(current_minimoon, formation, config)

    # reintegrate from start time to end time at desired time interval

    # transform to sun-earth co-rotating (position and velocity)

    # calc ra and dec

    # save file





###########################
# run sim
##########################

# Load YAML config file
with open("orbit_det_configuration.yaml", "r") as file:
    config = yaml.safe_load(file)

# open minimoon_master (assuming master and minimoon files have l1 apparant magnitude over traj and min index in master)
# create parser
mm_parser = MmParser("", "", "")

# get the master file - you need a list of initial orbits to integrate with openorb (pyorb)
master = mm_parser.parse_master_new_new_new(config['minimoon_master_file_path'])

run_sim_partial = partial(run_sim, minimoon_master=master, config=config)

#######################################
# iterate over minimoons in parallel
########################################

# parallel implementation
pool = multiprocessing.Pool()
results = pool.map(run_sim_partial, master['Object id'])
pool.close()





