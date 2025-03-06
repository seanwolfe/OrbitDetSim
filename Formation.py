import pandas as pd
import numpy as np
from Spacecraft import Spacecraft
import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import MaxNLocator

class Formation:
    def __init__(self, configs):
        prelim_orbit = pd.read_csv(configs['orbit_file_path'], sep=',', header=0, names=configs['orbit_column_names'])
        self.orbit = prelim_orbit.iloc[configs['quasi_halo_start']:configs['quasi_halo_end']]
        self.num_spacecraft = configs['num_spacecraft']
        self.sim_steps = None  # this comes from the asteroid trajectory
        self.spacecraft = None
        self.initial_formation(configs)

    def initial_formation(self, configs):
        ######
        # determine the initial position indexes of the spacecraft
        ######
        # find the total number of steps in the quasi-halo
        quasi_steps = configs['quasi_halo_one_period_end'] - configs['quasi_halo_start']

        # divide by number of s/c
        sc_start_range = int(quasi_steps / self.num_spacecraft)

        # randomly pick an index for the first spacecraft
        scs_start = [np.random.randint(0, sc_start_range)]

        # assign remaining s/c indices by adding random number times zone length times ith spacecraft in formation
        for i in range(1, self.num_spacecraft):
            scs_start.append(i * sc_start_range + scs_start[0])

        ########
        # set the initial positions of the spacecraft
        ########
        scs_ini_pos = [np.array([self.orbit['SUN_EARTH_CO_X_(km)'].iloc[sc_start],
                                 self.orbit['SUN_EARTH_CO_Y_(km)'].iloc[sc_start],
                                 self.orbit['SUN_EARTH_CO_Z_(km)'].iloc[sc_start]]) for
                       i, sc_start in enumerate(scs_start)]

        #######
        # declare the spacecraft and assign them to the formation
        ########
        self.spacecraft = [Spacecraft(ini_pos, scs_start[i], configs) for i, ini_pos in enumerate(scs_ini_pos)]

        return

    def match_spacecraft_trajectory(self, asteroid_length, configs):
        """
        Resamples and aligns each spacecraft trajectory to match the asteroid trajectory's
        one-hour intervals and start time.

        Parameters:
            asteroid_times (pd.Series): Timestamps of the asteroid trajectory.
            configs (dict): Configuration dictionary with conversion factors.

        Returns:
            None (modifies each spacecraft's matched_trajectory in place).
        """

        for i, spacecraft in enumerate(self.spacecraft):
            # Convert spacecraft timestamps to pandas datetime format

            self.orbit['Time'] = pd.to_datetime(self.orbit['Time'])
            self.orbit = self.orbit.drop_duplicates(subset=['Time'])  # there are duplicates in the lisa pathfinder orbit file apparently

            # Resample spacecraft data at hourly intervals (matching asteroid)
            spacecraft_resampled = self.orbit.set_index('Time').resample('1H').nearest().reset_index()

            # Keep only relevant position columns
            spacecraft_pos = spacecraft_resampled.loc[:, ['SUN_EARTH_CO_X_(km)',
                                                          'SUN_EARTH_CO_Y_(km)',
                                                          'SUN_EARTH_CO_Z_(km)']].to_numpy()

            sc_length = len(spacecraft_pos)
            start_index = spacecraft.pos_index  # Initial position index

            # Create the trajectory starting at the correct index
            ordered_traj = np.vstack([spacecraft_pos[start_index:], spacecraft_pos[:start_index]])

            if sc_length >= asteroid_length:
                # Trim if spacecraft trajectory is longer
                adjusted_traj = ordered_traj[:asteroid_length]
            else:
                # Wrap around if spacecraft trajectory is shorter
                repeats = asteroid_length // sc_length
                remainder = asteroid_length % sc_length

                adjusted_traj = np.vstack([
                    np.tile(ordered_traj, (repeats, 1)),  # Full cycles
                    ordered_traj[:remainder]  # Remaining part
                ])

            # Convert to AU

            self.spacecraft[i].matched_trajectory = adjusted_traj / (configs['AU_TO_M'] / 1000)

        return


    def update_formation(self):
        raise NotImplementedError


#################
# test to see if formation is in good spots
########################

"""
# Load YAML config file
with open("orbit_det_configuration.yaml", "r") as file:
    config = yaml.safe_load(file)

num_tests = 10

for i in range(0, num_tests):
    formation_i = Formation(config)
    formation_i.initial_formation(config)

    kmtoau = 6.68459e-9
    start = config['quasi_halo_start']
    end = config['quasi_halo_end']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(formation_i.orbit["MOON_SUN_EARTH_CO_X_(km)"].iloc[start:end] * kmtoau,
            formation_i.orbit["MOON_SUN_EARTH_CO_Y_(km)"].iloc[start:end] * kmtoau,
            formation_i.orbit["MOON_SUN_EARTH_CO_Z_(km)"].iloc[start:end] * kmtoau, label='Moon')
    ax.plot(formation_i.orbit["SUN_EARTH_CO_X_(km)"].iloc[start:end] * kmtoau,
            formation_i.orbit["SUN_EARTH_CO_Y_(km)"].iloc[start:end] * kmtoau,
            formation_i.orbit["SUN_EARTH_CO_Z_(km)"].iloc[start:end] * kmtoau)
    ax.scatter(0.009, 0, 0, label='L_1', s=20)

    for i, sc in enumerate(formation_i.spacecraft):
        ax.scatter(sc.position[0] * kmtoau, sc.position[1] * kmtoau, sc.position[2] * kmtoau, s=20)

    # Create a sphere (Earth model)
    theta = np.linspace(0, np.pi, 30)  # Latitude
    phi = np.linspace(0, 2 * np.pi, 60)  # Longitude
    theta, phi = np.meshgrid(theta, phi)

    # Earth radius (approx. in arbitrary units)
    R = 6378  # Normalize radius

    # Convert spherical to Cartesian coordinates
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    # Plot wireframe Earth
    ax.plot_wireframe(x * kmtoau, y * kmtoau, z * kmtoau, color="blue", linewidth=0.5, alpha=0.7)

    ax.set_xlabel('X (au)')
    ax.set_ylabel('Y (au)')
    ax.set_zlabel('Z (au)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.legend()
    plt.show()
"""