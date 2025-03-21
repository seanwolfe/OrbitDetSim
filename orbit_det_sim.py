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


# Define a converter function
def str_to_tuple(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


def calc_start_index(minimoon, sc_formation, configs):

    # asteroid position
    asteroid_pos = minimoon.orbit.loc[:, ['Synodic x', 'Synodic y', 'Synodic z']].values
    earth_pos = np.zeros_like(asteroid_pos)
    print(minimoon.id)
    moon_pos = minimoon.orbit.loc[:, ['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']].values

    # get spacecraft positions over trajectory
    sc_formation.match_spacecraft_trajectory(len(asteroid_pos[:, 0]), configs)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(moon_pos[:, 0], moon_pos[:, 1], moon_pos[:, 2], label='Moon')
    ax.plot(asteroid_pos[:, 0], asteroid_pos[:, 1], asteroid_pos[:, 2], label='Asteroid')
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
        is_visible = visible[~np.isnan(visible)]

        if len(is_visible) == 0:
            pass
        else:
            # for indi in is_visible:
            test_i = int(is_visible[0])
            print(test_i)

            spacecraft_pos = spacecraft.get_spacecraft_pos(test_i)
            #####
            # test to see if all trajectories look fine - and they do
            ####

            ax.plot(sc_pos[:test_i, 0], sc_pos[:test_i, 1], sc_pos[:test_i, 2], color=colors[i], zorder=15)
            ax.scatter(*spacecraft.get_spacecraft_pos(0), s=20, color=colors[i], label='Initial pos sc' + str(i), zorder=20, marker='^')
            ax.scatter(*spacecraft_pos, s=20, color=colors[i], label='Detection instant sc '+ str(i), zorder=20)
            ax.scatter(*minimoon.get_asteroid_pos(test_i), s=20, color='green')

            fov_corners = plot_fov_projection(spacecraft, minimoon, test_i)

            # Plot dotted lines from spacecraft to FOV corners
            for corner in fov_corners:
                ax.plot([spacecraft_pos[0], corner[0]],
                        [spacecraft_pos[1], corner[1]],
                        [spacecraft_pos[2], corner[2]], 'k--', alpha=0.5)

            # Draw FOV projection as a polygon
            fov_poly = Poly3DCollection([fov_corners], color='cyan', alpha=0.3, edgecolor='k')
            ax.add_collection3d(fov_poly)

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



def run_sim_minimoons(object_id, minimoon_master, config):

    # declare asteroid
    current_minimoon_master = minimoon_master[minimoon_master['Object id'] == object_id]
    current_minimoon = Asteroid(object_id, current_minimoon_master['Min_SunEarthL1_V_index'], config)

    # declare formation
    formation = Formation(config)


    # determine when an initial detection will be made
    sc_visible = calc_start_index(current_minimoon, formation, config)

    # determine when formation will stop detecting on initial window end index
    # end_index = calc_end_index(current_minimoon, formation, config)

    # reintegrate from start time to end time at desired time interval

    # transform to sun-earth co-rotating (position and velocity)

    # calc ra and dec

    # save file

    return sc_visible


def run_sim_runnumbers(run_number, minimoon_master, config):

    visibles = []

    # declare asteroid
    for idx, master_i in minimoon_master.iterrows():

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
            visibles.append([[run_number, current_minimoon.id, jdx + 1], visible])


    return visibles, formation.spacecraft

@staticmethod
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
        result, scs = run_sim_runnumbers(idx + 1, minimoon_master, config)

        flat_data = []
        for jdx, run in enumerate(result):
            index = tuple(run[0])  # (run_number, object_id, spacecraft_number)
            values = run[1]  # List of values over time
            flat_data.append((*index, values, scs[0].ini_position))

        # Convert to DataFrame
        df = pd.DataFrame(flat_data, columns=["run_number", "object_id", "spacecraft_number", "values", "spacecraft_1_ini_pos"])

        # Set MultiIndex
        df.set_index(["run_number", "object_id", "spacecraft_number"], inplace=True)

        df.to_csv(config['output_df_file_name'] + '_run_' + str(idx + 1) + '.csv', sep=',', header=True, index=True)

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


###########################
# run sim
##########################

# Load YAML config file
with open("orbit_det_configuration.yaml", "r") as file:
    config = yaml.safe_load(file)

# get the master file
master = parse_master_new_new_new(config['minimoon_master_file_path'])


###################################
# Run parallel for number of runs using MPI
####################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

run_sim_runnumbers_MPI(master, config)

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

run_sim_minimoons_partial = partial(run_sim_minimoons, minimoon_master=master, config=config)

# parallel implementation
# pool = multiprocessing.Pool(processes=1)
# results = pool.map(run_sim_minimoons_partial, master['Object id'])
# pool.close()





