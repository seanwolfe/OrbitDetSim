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
import spiceypy as spice

# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')

def viz(objects_pos, minimoon, sc_formation, configs):
    # asteroid position
    asteroid_pos = minimoon.orbit.loc[:, ['Synodic x', 'Synodic y', 'Synodic z']].values
    earth_pos = np.zeros_like(asteroid_pos)
    print(minimoon.id)
    moon_pos = minimoon.orbit.loc[:, ['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']].values

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(moon_pos[:, 0], moon_pos[:, 1], moon_pos[:, 2], label='Moon')
    ax.plot(asteroid_pos[:, 0], asteroid_pos[:, 1], asteroid_pos[:, 2], label='Asteroid', color='green')
    ax.scatter(0.009, 0, 0, label='L_1', s=20)
    # Create a sphere (Earth model)
    theta = np.linspace(0, np.pi, 30)  # Latitude
    phi = np.linspace(0, 2 * np.pi, 60)  # Longitude
    theta, phi = np.meshgrid(theta, phi)

    # Earth radius (approx. in arbitrary units)
    R = 6378  # Normalize radius

    # Convert spherical to Cartesian coordinates
    x = R * np.sin(theta) * np.cos(phi) / (configs['AU_TO_M'] / 1000)  # km
    y = R * np.sin(theta) * np.sin(phi) / (configs['AU_TO_M'] / 1000)
    z = R * np.cos(theta) / (configs['AU_TO_M'] / 1000)

    # Plot wireframe Earth
    ax.plot_wireframe(x, y, z, color="blue", linewidth=0.5, alpha=0.7)

    # start index is first instance asteroid is FOV of a sc, without occlusion from Earth or moon
    # it is the index in the minimoon trajectory corresponding to this
    sc_visible = []
    colors = ['red', 'blue', 'orange', 'green', 'grey', 'brown', 'black', 'purple', 'yellow', 'pink']
    for i, spacecraft in enumerate(sc_formation.spacecraft):

        sc_pos = spacecraft.matched_trajectory

        # find when the asteroid is in fov and not ocluded by earth or moon
        visible = spacecraft.asteroid_in_fov_batch(asteroid_pos, sc_pos, earth_pos, moon_pos, configs)
        sc_visible.append(visible)

        ######################
        # For generation of spacecrafr fov with asteroid figure
        ###########################################
        is_visible = visible[visible!=-1]

        if len(is_visible) == 0:
            pass
        else:
            # for indi in is_visible:
            test_i = int(is_visible[0])
            ax.scatter(*minimoon.get_asteroid_pos(test_i), s=20, color='green')
            fov_corners = plot_fov_projection(spacecraft, minimoon, test_i)
            spacecraft_pos = spacecraft.get_spacecraft_pos(test_i)
            # Plot dotted lines from spacecraft to FOV corners
            for corner in fov_corners:
                ax.plot([spacecraft_pos[0], corner[0]],
                        [spacecraft_pos[1], corner[1]],
                        [spacecraft_pos[2], corner[2]], 'k--', alpha=0.5)

            # Draw FOV projection as a polygon
            fov_poly = Poly3DCollection([fov_corners], color='cyan', alpha=0.3, edgecolor='k')
            ax.add_collection3d(fov_poly)

            for j, spacecraft_j in enumerate(sc_formation.spacecraft):
                spacecraft_pos_j = spacecraft_j.get_spacecraft_pos(test_i)
                sc_pos_j = spacecraft_j.matched_trajectory

                ax.plot(sc_pos_j[:test_i, 0], sc_pos_j[:test_i, 1], sc_pos_j[:test_i, 2], color=colors[j], zorder=15)
                ax.scatter(*spacecraft_j.get_spacecraft_pos(0), s=20, color=colors[j], label='Initial pos sc' + str(j),
                           zorder=20, marker='^')
                ax.scatter(*spacecraft_pos_j, s=20, color=colors[j], label='Detection instant sc ' + str(j), zorder=20)

                if j == 0:
                    object_pos = objects_pos[j]
                    ax.scatter(object_pos[0, 0], object_pos[1, 0], object_pos[2, 0], color=colors[j + 2], s=30, label='Integration start sc ' + str(j), zorder=19)
                    ax.plot(object_pos[0, :], object_pos[1, :], object_pos[2, :], color=colors[j + 2], linewidth=3, label='Integrated traj sc ' + str(j), zorder=14)



    ax.plot(sc_pos[:, 0], sc_pos[:, 1], sc_pos[:, 2], color='pink', label='Halo Orbit', zorder=5)
    ax.set_xlabel('X (au)')
    ax.set_ylabel('Y (au)')
    ax.set_zlabel('Z (au)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.legend()
    ax.set_aspect('equal')
    plt.show()

    return sc_visible


# Define a converter function
def str_to_tuple(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


def read_master(file_path, config):
    columns_to_convert = config['visible_file_columns']
    return pd.read_csv(file_path, sep=',', converters={col: str_to_tuple for col in columns_to_convert},
                       index_col=config['index_columns'])


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
    file_path = '/media/aeromec/Seagate Desktop Drive/minimoon_files_oorb/' + str(
        minimoon_df['Object id'].iloc[0]) + '.csv'
    minimoon_df.to_csv(file_path, sep=' ', header=True, index=False)

    return positions_corotating


def eclip_to_sun_earth_corotating_batch_n_body_integrator_output(states, earth_states):
    """
    Converts a batch of positions and velocities from the heliocentric ECLIPJ2000 frame
    to the Sun-Earth co-rotating frame.

    Parameters:
    - positions_eclip (numpy array): Nx3 array of positions in ECLIPJ2000 (AU).
    - et_times (numpy array): N-element array of ephemeris times.

    Returns:
    - positions_corotating (numpy array): Nx3 array of positions in Sun-Earth co-rotating frame (AU).
    """

    earth_positions = earth_states[:3, :].T

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

    num_objects = len(states)

    positions_corotating = []
    for i in range(0, num_objects):

        object_i_pos = states[i, :3, :].T  + earth_positions

        rotation_matrices = np.asarray(rotation_matrices, dtype=np.float64)
        relative_positions = np.asarray(object_i_pos, dtype=np.float64)

        # Apply the rotation to transform positions
        position_corotating = np.einsum("nij,nj->ni", rotation_matrices, relative_positions)
        positions_corotating.append(position_corotating.T)


    return positions_corotating


def plot_fov_projection(spacecraft, asteroid, index):
    """
    Visualizes the spacecraft's field of view (FOV) projection along the boresight at the asteroid's distance.

    Parameters:
        spacecraft: the spacecraft object
        asteroid: the asteroid object
            """
    # Convert FOV from degrees to radians
    fov_rad = np.radians(np.sqrt(spacecraft.fov))

    spacecraft_pos = spacecraft.get_spacecraft_pos(index)
    asteroid_pos = asteroid.get_asteroid_pos(index)

    # Compute distance to asteroid
    sc_to_ast = np.array(asteroid_pos) - np.array(spacecraft_pos)
    ast_distance = np.linalg.norm(sc_to_ast)

    # Find the FOV projection center (along boresight at asteroid's distance)
    fov_center = np.array(spacecraft_pos) + ast_distance * spacecraft.boresight

    # Define perpendicular vectors for FOV plane (orthogonal to boresight)
    up = np.array([0, 0, 1]) if abs(spacecraft.boresight[2]) < 0.9 else np.array([1, 0, 0])  # Avoid collinear vector
    right = np.cross(spacecraft.boresight, up)
    new_right = right / np.linalg.norm(right)
    up = np.cross(new_right, spacecraft.boresight)  # Recompute true "up" vector

    # Compute FOV half-width at this distance
    fov_half_width = np.tan(fov_rad / 2) * ast_distance

    # Compute the 4 corners of the FOV projection at the asteroid's distance
    # order is [-1,-1], [1, -1], [-1, 1], [1, 1]
    fov_corners = []
    for dy in [-1, 1]:
        for dx in [-1, 1]:
            corner = fov_center + dx * fov_half_width * right + dy * fov_half_width * up
            fov_corners.append(corner)

    fov_corners[1], fov_corners[2], fov_corners[3] = fov_corners[2], fov_corners[3], fov_corners[1]
    fov_corners.append(fov_corners[0])

    return fov_corners


def parse_master_new_new_new(file_path):
    """
    function for obtaning master mimimoon data file, with parameters
    'Object id', 'H', 'D', 'Capture Date', 'Helio x at Capture', 'Helio y at Capture', 'Helio z at Capture',
    'Helio vx at Capture', 'Helio vy at Capture', 'Helio vz at Capture', 'Helio q at Capture', 'Helio e at Capture',
    'Helio i at Capture', 'Helio Omega at Capture', 'Helio omega at Capture', 'Helio M at Capture',
    'Geo x at Capture', 'Geo y at Capture', 'Geo z at Capture', 'Geo vx at Capture', 'Geo vy at Capture',
    'Geo vz at Capture', 'Geo q at Capture', 'Geo e at Capture', 'Geo i at Capture', 'Geo Omega at Capture',
    'Geo omega at Capture', 'Geo M at Capture', 'Moon (Helio) x at Capture', 'Moon (Helio) y at Capture',
    'Moon (Helio) z at Capture', 'Moon (Helio) vx at Capture', 'Moon (Helio) vy at Capture',
    'Moon (Helio) vz at Capture', 'Capture Duration', 'Spec. En. Duration', '3 Hill Duration', 'Number of Rev',
    '1 Hill Duration', 'Min. Distance', 'Release Date', 'Helio x at Release', 'Helio y at Release',
    'Helio z at Release', 'Helio vx at Release', 'Helio vy at Release', 'Helio vz at Release', 'Helio q at Release',
    'Helio e at Release', 'Helio i at Release', 'Helio Omega at Release', 'Helio omega at Release',
    'Helio M at Release', 'Geo x at Release', 'Geo y at Release', 'Geo z at Release', 'Geo vx at Release',
    'Geo vy at Release', 'Geo vz at Release', 'Geo q at Release', 'Geo e at Release', 'Geo i at Release',
    'Geo Omega at Release', 'Geo omega at Release', 'Geo M at Release', 'Moon (Helio) x at Release',
     'Moon (Helio) y at Release', 'Moon (Helio) z at Release', 'Moon (Helio) vx at Release',
     'Moon (Helio) vy at Release', 'Moon (Helio) vz at Release', 'Retrograde', 'Became Minimoon', 'Max. Distance',
     'Capture Index', 'Release Index', 'X at Earth Hill', 'Y at Earth Hill', 'Z at Earth Hill', 'Taxonomy', 'STC'
     "EMS Duration", "Periapsides in EMS", "Periapsides in 3 Hill", "Periapsides in 2 Hill", "Periapsides in 1 Hill",
    "STC Start", "STC Start Index", "STC End", "STC End Index", "Helio x at EMS", "Helio y at EMS", "Helio z at EMS",
     "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS", "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
    "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)", "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
     "Moon x at EMS (Helio)", "Moon y at EMS (Helio)", "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
      "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", 'Entry Date to EMS', 'Entry to EMS Index',
     'Exit Date to EMS', 'Exit Index to EMS' "Dimensional Jacobi" "Non-Dimensional Jacobi" Alpha_I Beta_I Theta_M
     "Minimum Energy", "Peri-EM-L2", "Average Geo z", "Average Geo vz", "Winding Difference"
    :return:
    """
    master_data = pd.read_csv(file_path, sep=",", header=0, names=['Object id', 'H', 'D', 'Capture Date',
                                                                   'Helio x at Capture', 'Helio y at Capture',
                                                                   'Helio z at Capture', 'Helio vx at Capture',
                                                                   'Helio vy at Capture', 'Helio vz at Capture',
                                                                   'Helio q at Capture', 'Helio e at Capture',
                                                                   'Helio i at Capture', 'Helio Omega at Capture',
                                                                   'Helio omega at Capture', 'Helio M at Capture',
                                                                   'Geo x at Capture', 'Geo y at Capture',
                                                                   'Geo z at Capture', 'Geo vx at Capture',
                                                                   'Geo vy at Capture', 'Geo vz at Capture',
                                                                   'Geo q at Capture', 'Geo e at Capture',
                                                                   'Geo i at Capture', 'Geo Omega at Capture',
                                                                   'Geo omega at Capture', 'Geo M at Capture',
                                                                   'Moon (Helio) x at Capture',
                                                                   'Moon (Helio) y at Capture',
                                                                   'Moon (Helio) z at Capture',
                                                                   'Moon (Helio) vx at Capture',
                                                                   'Moon (Helio) vy at Capture',
                                                                   'Moon (Helio) vz at Capture',
                                                                   'Capture Duration', 'Spec. En. Duration',
                                                                   '3 Hill Duration', 'Number of Rev',
                                                                   '1 Hill Duration', 'Min. Distance',
                                                                   'Release Date', 'Helio x at Release',
                                                                   'Helio y at Release', 'Helio z at Release',
                                                                   'Helio vx at Release', 'Helio vy at Release',
                                                                   'Helio vz at Release', 'Helio q at Release',
                                                                   'Helio e at Release', 'Helio i at Release',
                                                                   'Helio Omega at Release',
                                                                   'Helio omega at Release',
                                                                   'Helio M at Release', 'Geo x at Release',
                                                                   'Geo y at Release', 'Geo z at Release',
                                                                   'Geo vx at Release', 'Geo vy at Release',
                                                                   'Geo vz at Release', 'Geo q at Release',
                                                                   'Geo e at Release', 'Geo i at Release',
                                                                   'Geo Omega at Release',
                                                                   'Geo omega at Release', 'Geo M at Release',
                                                                   'Moon (Helio) x at Release',
                                                                   'Moon (Helio) y at Release',
                                                                   'Moon (Helio) z at Release',
                                                                   'Moon (Helio) vx at Release',
                                                                   'Moon (Helio) vy at Release',
                                                                   'Moon (Helio) vz at Release', 'Retrograde',
                                                                   'Became Minimoon', 'Max. Distance',
                                                                   'Capture Index',
                                                                   'Release Index', 'X at Earth Hill',
                                                                   'Y at Earth Hill',
                                                                   'Z at Earth Hill', 'Taxonomy', 'STC',
                                                                   "EMS Duration",
                                                                   "Periapsides in EMS", "Periapsides in 3 Hill",
                                                                   "Periapsides in 2 Hill", "Periapsides in 1 Hill",
                                                                   "STC Start", "STC Start Index", "STC End",
                                                                   "STC End Index",
                                                                   "Helio x at EMS", "Helio y at EMS",
                                                                   "Helio z at EMS",
                                                                   "Helio vx at EMS", "Helio vy at EMS",
                                                                   "Helio vz at EMS",
                                                                   "Earth x at EMS (Helio)",
                                                                   "Earth y at EMS (Helio)",
                                                                   "Earth z at EMS (Helio)",
                                                                   "Earth vx at EMS (Helio)",
                                                                   "Earth vy at EMS (Helio)",
                                                                   "Earth vz at EMS (Helio)",
                                                                   "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
                                                                   "Moon z at EMS (Helio)",
                                                                   "Moon vx at EMS (Helio)",
                                                                   "Moon vy at EMS (Helio)",
                                                                   "Moon vz at EMS (Helio)",
                                                                   'Entry Date to EMS', 'Entry to EMS Index',
                                                                   'Exit Date to EMS', 'Exit Index to EMS',
                                                                   "Dimensional Jacobi", "Non-Dimensional Jacobi",
                                                                   'Alpha_I',
                                                                   'Beta_I', 'Theta_M', "Minimum Energy",
                                                                   "Peri-EM-L2", "Average Geo z", "Average Geo vz",
                                                                   "Winding Difference", "Min_SunEarthL1_V",
                                                                   "Min_SunEarthL1_V_index"])

    return master_data


def get_sc_state_from_sc1_position(detected_pop, config):
    closest_indices = []
    scs_helio = []
    sc_epochs = []
    for kdx, detection in detected_pop.iterrows():
        # create a formation object, it has s/c s randomly placed
        formation = Formation(config)

        # we already had a saved formation, saved according to the s/c 1 position, get the correspoding index in overall orbit file
        sc1_ini_index = formation.get_index_from_pos(detection['spacecraft_1_ini_pos'])

        # re-initialize formation with this index
        formation.recall_formation(sc1_ini_index, config)

        # match the spacecraft trajectories to that of the asteroid in terms of length and sampling (asteroid sampled at one hour)
        formation.match_spacecraft_trajectory(len(detection['values']), config)

        # the spacecraft that detected the asteroid
        detecting_spacecraft = formation.spacecraft[detection.name[2] - 1]  # spacecraft id start from 1
        # detecting_spacecraft = formation.spacecraft[0]  # spacecraft id start from 1

        # the position of the detecting spacecraft at the detection instant
        desired_sc_pos = detecting_spacecraft.matched_trajectory[int(detection['min_nonnegative']), :] * (
                config['AU_TO_M'] / 1000)  # now in km sun-earth-syn

        # desired_sc_pos = detecting_spacecraft.matched_trajectory[0, :] * (
        #         config['AU_TO_M'] / 1000)

        # match this position to the overall orbit file
        possible_positions = formation.orbit.loc[:, ['SUN_EARTH_CO_X_(km)',
                                                     'SUN_EARTH_CO_Y_(km)',
                                                     'SUN_EARTH_CO_Z_(km)']]
        distances = np.linalg.norm(possible_positions - desired_sc_pos, axis=1)
        closest_position_index = np.argmin(distances)

        # use the index at the match to query spacecraft state vector
        geo_eme_state = formation.orbit.loc[
            formation.orbit.index[closest_position_index], ["GEO_EME_X_(km)", "GEO_EME_Y_(km)", "GEO_EME_Z_(km)",
                                                            "GEO_EME_Vx_(km/s)", "GEO_EME_Vy_(km/s)",
                                                            "GEO_EME_Vz_(km/s)"]].to_numpy()


        # get the earth's state vector at detection instant
        sc_time = formation.orbit.loc[formation.orbit.index[closest_position_index], "Time"]

        # get the detecting spacecraft state in geo eclip frame
        geo_eclip_state = eme_to_ecliptic_batch(geo_eme_state)
        scs_helio.append(geo_eclip_state)
        closest_indices.append(closest_position_index)
        sc_epochs.append(sc_time.strftime("%Y-%m-%d %H:%M:%S"))

    detected_pop.loc[:, 'detecting_sc_lpf_orbit_index'] = closest_indices
    detected_pop.loc[:, 'sc_epoch'] = sc_epochs
    detected_pop.loc[:, ['GEO_ECLIP_X_(km)', 'GEO_ECLIP_Y_(km)', 'GEO_ECLIP_Z_(km)', 'GEO_ECLIP_Vx_(km/s)', 'GEO_ECLIP_Vy_(km/s)',
                  'GEO_ECLIP_Vz_(km/s)']] = np.array(scs_helio)

    return  detected_pop


def helio_eclip_from_geo_eme(eme_vectors, earth_helio_state):
    ###########################
    # convert geo eme to geo elcip
    ###########################

    # Obliquity of the ecliptic at J2000 (in degrees)
    epsilon = 23.439281  # Mean obliquity of the ecliptic at J2000 epoch

    # Convert epsilon to radians
    epsilon_rad = np.radians(epsilon)

    # Rotation matrix for transformation about the x-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(epsilon_rad), np.sin(epsilon_rad)],
        [0, -np.sin(epsilon_rad), np.cos(epsilon_rad)]
    ])

    # Apply the rotation to each vector using matrix multiplication
    ecliptic_positions = np.dot(eme_vectors[:3], rotation_matrix.T)
    ecliptic_velocities = np.dot(eme_vectors[3:], rotation_matrix.T)

    ##############################
    # convert geo eclip to helio
    ##########################
    helio_eclip_position = ecliptic_positions + earth_helio_state[:3]
    helio_eclip_velocities = ecliptic_velocities + earth_helio_state[3:]

    return np.hstack((helio_eclip_position, helio_eclip_velocities))


def eme_to_ecliptic_batch(state_vectors_eme):
    """
    Transforms a batch of full state vectors from EME J2000 to Ecliptic J2000.

    Parameters:
    - state_vectors_eme (numpy array): Nx6 array representing N state vectors
      in EME J2000 (each row: [x, y, z, vx, vy, vz]).

    Returns:
    - numpy array: Nx6 array representing N state vectors in Ecliptic J2000.
    """
    # Obliquity of the ecliptic at J2000 (in degrees)
    epsilon = 23.439281  # Mean obliquity of the ecliptic at J2000 epoch
    epsilon_rad = np.radians(epsilon)

    # Rotation matrix about the x-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(epsilon_rad), np.sin(epsilon_rad)],
        [0, -np.sin(epsilon_rad), np.cos(epsilon_rad)]
    ])

    # Split into position and velocity
    positions = state_vectors_eme[0:3]
    velocities = state_vectors_eme[3:6]

    # Rotate both
    pos_ecliptic = rotation_matrix @ positions
    vel_ecliptic = velocities @ rotation_matrix.T

    # Concatenate position and velocity back
    state_vectors_ecliptic = np.hstack((pos_ecliptic, vel_ecliptic))

    return state_vectors_ecliptic