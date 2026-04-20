import pandas as pd
import numpy as np
from Spacecraft import Spacecraft

class Formation:
    def __init__(self, configs):
        prelim_orbit = pd.read_csv(configs['orbit_file_path'], sep=',', header=0, names=configs['orbit_column_names'])
        self.orbit = prelim_orbit.iloc[configs['quasi_halo_start']:configs['quasi_halo_end']]
        self.num_spacecraft = configs['num_spacecraft']
        self.sim_steps = None  # this comes from the asteroid trajectory
        self.spacecraft = None
        self.currently_detecting = None
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
                       j, sc_start in enumerate(scs_start)]

        #######
        # declare the spacecraft and assign them to the formation
        ########
        self.spacecraft = [Spacecraft(ini_pos, scs_start[k], configs) for k, ini_pos in enumerate(scs_ini_pos)]

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
            start_index = spacecraft.ini_pos_index  # Initial position index

            self.orbit['Time'] = pd.to_datetime(self.orbit['Time'])
            self.orbit = self.orbit.drop_duplicates(subset=['Time'])  # there are duplicates in the lisa pathfinder orbit file apparently

            # Assuming self.orbit['Time'] is already in datetime format
            original_timestamp = self.orbit.iloc[start_index]['Time']

            # Resample spacecraft data at hourly intervals (matching asteroid)
            spacecraft_resampled = self.orbit.set_index('Time').resample('h').nearest().reset_index()

            # Find the closest timestamp in the resampled data
            try:
                new_index = spacecraft_resampled[spacecraft_resampled['Time'] == original_timestamp].index[0]
            except IndexError:
                # If the exact timestamp is not found, use nearest method
                # Get the nearest index based on the original timestamp
                nearest_timestamp = spacecraft_resampled['Time'].iloc[
                    (spacecraft_resampled['Time'] - original_timestamp).abs().argmin()]
                new_index = spacecraft_resampled[spacecraft_resampled['Time'] == nearest_timestamp].index[0]

            # Keep only relevant position columns
            spacecraft_pos = spacecraft_resampled.loc[:, ['SUN_EARTH_CO_X_(km)',
                                                          'SUN_EARTH_CO_Y_(km)',
                                                          'SUN_EARTH_CO_Z_(km)']].to_numpy()

            sc_length = len(spacecraft_pos)

            # Create the trajectory starting at the correct index
            ordered_traj = np.vstack([spacecraft_pos[new_index:], spacecraft_pos[:new_index]])

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

    def match_spacecraft_trajectory_full(self, asteroid_length, configs):
        """
        Resamples and aligns each spacecraft trajectory to match the asteroid trajectory's
        one-hour intervals and start time.

        TIME BEHAVIOR (as requested):
          - We resample onto an hourly grid using nearest-neighbor selection.
          - Instead of keeping the hourly grid times, we keep the ORIGINAL epoch of the
            row that was selected as nearest (by carrying a ROW_ID through resampling).

        Keeps ALL orbit columns.

        Unit conversion:
          - columns containing '(km)'   (but not '(km/s)') -> AU
          - columns containing '(km/s)'                    -> AU/day
          - Time stays datetime (original epochs)
        """

        AU_km = configs["AU_TO_M"] / 1000.0
        SEC_PER_DAY = 86400.0

        if "Time" not in self.orbit.columns:
            raise ValueError("self.orbit must contain a 'Time' column")

        # ---- Clean orbit once ----
        orbit = self.orbit.copy()
        orbit["Time"] = pd.to_datetime(orbit["Time"])
        orbit = orbit.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)

        # Add a stable row id so we can recover the exact original epoch after resampling
        orbit["ROW_ID"] = np.arange(len(orbit), dtype=np.int64)

        # Cache a ROW_ID -> original Time mapping
        rowid_to_time = orbit.set_index("ROW_ID")["Time"]

        # All data columns except Time (we will handle Time explicitly)
        data_cols = [c for c in orbit.columns if c not in ("Time",)]
        # (data_cols includes ROW_ID; we will drop it at the end)

        # ---- Hourly resample using nearest original row ----
        # This returns a row whose values come from the nearest original timestamp;
        # crucially, ROW_ID comes along for the ride.
        spacecraft_resampled = (
            orbit.set_index("Time")
            .resample("h")
            .nearest()
            .reset_index()
        )

        # Replace resampled Time (hourly grid) with the ORIGINAL epoch of the selected row
        if "ROW_ID" not in spacecraft_resampled.columns:
            raise RuntimeError("ROW_ID missing after resample; cannot recover original epochs.")

        spacecraft_resampled["Time"] = spacecraft_resampled["ROW_ID"].map(rowid_to_time)

        # Optional: if nearest selection causes duplicates in Time, keep them (your choice).
        # If you want to drop duplicates after mapping, uncomment:
        # spacecraft_resampled = spacecraft_resampled.drop_duplicates(subset=["Time"]).reset_index(drop=True)

        # Columns we will carry into matched trajectory (everything except ROW_ID)
        out_cols = [c for c in spacecraft_resampled.columns if c != "ROW_ID"]
        numeric_cols = [c for c in out_cols if c != "Time"]  # convert these as needed

        for i, spacecraft in enumerate(self.spacecraft):
            start_index = spacecraft.ini_pos_index
            original_timestamp = orbit.iloc[start_index]["Time"]

            # Find the index in the resampled table whose ORIGINAL epoch matches the spacecraft start,
            # otherwise snap to nearest by time.
            idx_matches = spacecraft_resampled.index[spacecraft_resampled["Time"] == original_timestamp]
            if len(idx_matches) > 0:
                new_index = int(idx_matches[0])
            else:
                new_index = int((spacecraft_resampled["Time"] - original_timestamp).abs().argmin())

            sc_length = len(spacecraft_resampled)

            # Rotate the entire table (Time + all columns) together
            ordered_df = pd.concat(
                [spacecraft_resampled.iloc[new_index:], spacecraft_resampled.iloc[:new_index]],
                ignore_index=True,
            )

            # Keep only desired output columns (drops ROW_ID)
            ordered_df = ordered_df.loc[:, out_cols]

            # Trim or repeat to match asteroid_length
            if sc_length >= asteroid_length:
                adjusted_df = ordered_df.iloc[:asteroid_length].copy()
            else:
                repeats = asteroid_length // sc_length
                remainder = asteroid_length % sc_length
                adjusted_df = pd.concat(
                    [ordered_df] * repeats + [ordered_df.iloc[:remainder]],
                    ignore_index=True,
                )

            # --- Unit conversion ---
            # Velocities: km/s -> AU/day
            vel_cols = [c for c in numeric_cols if "(km/s)" in c]
            if vel_cols:
                adjusted_df[vel_cols] = adjusted_df[vel_cols] * (SEC_PER_DAY / AU_km)

            # Positions: km -> AU  (exclude km/s)
            pos_cols = [c for c in numeric_cols if "(km)" in c and "(km/s)" not in c]
            if pos_cols:
                adjusted_df[pos_cols] = adjusted_df[pos_cols] / AU_km

            self.spacecraft[i].matched_trajectory_full = adjusted_df

        # Write back cleaned orbit if you want side effects to persist
        self.orbit = orbit.drop(columns=["ROW_ID"])

        return

    def get_index_from_pos(self, position):
        possible_positions = self.orbit.loc[:, ['SUN_EARTH_CO_X_(km)', 'SUN_EARTH_CO_Y_(km)', 'SUN_EARTH_CO_Z_(km)']]
        distances = np.linalg.norm(possible_positions - position, axis=1)
        closest_position_idx = np.argmin(distances)
        return closest_position_idx

    def recall_formation(self, sc1_ini_index, config):

        # find the total number of steps in the quasi-halo
        quasi_steps = config['quasi_halo_one_period_end'] - config['quasi_halo_start']

        # divide by number of s/c
        sc_start_range = int(quasi_steps / self.num_spacecraft)

        scs_start = [sc1_ini_index]

        # assign remaining s/c indices by adding random number times zone length times ith spacecraft in formation
        for i in range(1, self.num_spacecraft):
            scs_start.append(i * sc_start_range + scs_start[0])

        ########
        # set the initial positions of the spacecraft
        ########
        scs_ini_pos = [np.array([self.orbit['SUN_EARTH_CO_X_(km)'].iloc[sc_start],
                                 self.orbit['SUN_EARTH_CO_Y_(km)'].iloc[sc_start],
                                 self.orbit['SUN_EARTH_CO_Z_(km)'].iloc[sc_start]]) for
                       j, sc_start in enumerate(scs_start)]

        #######
        # declare the spacecraft and assign them to the formation
        ########
        self.spacecraft = [Spacecraft(ini_pos, scs_start[k], config) for k, ini_pos in enumerate(scs_ini_pos)]

        return

    def get_spacecraft_states(self, sc_ids=None):
        """
        Return spacecraft states in EME frame.

        Parameters
        ----------
        sc_ids : None, int, sequence of int, or boolean array
            - None: return all spacecraft states
            - int: return state of one spacecraft (shape: (6,))
            - sequence of int: return selected spacecraft states (N,6)
            - boolean array: mask over spacecraft list

        Returns
        -------
        np.ndarray
            Spacecraft state(s) in EME frame.
        """
        all_sc = self.spacecraft

        # Stack all states once
        states = np.vstack([sc.curr_state_eme for sc in all_sc])  # (M,6)

        if sc_ids is None:
            return states

        # Single spacecraft
        if isinstance(sc_ids, int):
            return states[sc_ids]

        # List / tuple / ndarray / boolean mask
        return states[sc_ids]

    def get_spacecraft_pointings(self, sc_ids=None):
        """
        Return spacecraft boresight vectors.

        Parameters
        ----------
        sc_ids : None, int, sequence of int, or boolean array
            - None: return all boresights (M,3)
            - int: return one boresight (3,)
            - sequence / mask: return selected boresights (N,3)

        Returns
        -------
        np.ndarray
            Spacecraft boresight vector(s).
        """
        # Stack once: (M,3)
        boresights = np.vstack([sc.boresight for sc in self.spacecraft])

        if sc_ids is None:
            return boresights

        if isinstance(sc_ids, int):
            return boresights[sc_ids]

        return boresights[sc_ids]

    def set_spacecraft_states(self, states_eme, sc_ids=None, *, set_epochs_from_row=None):
        """
        Set curr_state_eme for spacecraft (optionally a subset).

        Parameters
        ----------
        states_eme : np.ndarray
            If sc_ids is None: shape (M,6)
            If sc_ids is not None: shape (N,6) matching selected spacecraft count
        sc_ids : None, int, sequence of int, or boolean mask
            Which spacecraft to set.
        set_epochs_from_row : pandas.Series or None
            If provided, also sets sc.curr_sc_epoch from row["EPOCH_SC_i(jdtdb)"].
        """
        all_sc = self.spacecraft
        M = len(all_sc)

        if sc_ids is None:
            ids = np.arange(M)
        elif isinstance(sc_ids, int):
            ids = np.array([sc_ids], dtype=int)
            states_eme = np.atleast_2d(states_eme)
        else:
            ids = np.asarray(sc_ids)
            states_eme = np.asarray(states_eme)

        if states_eme.shape != (len(ids), 6):
            raise ValueError(f"states_eme must have shape ({len(ids)}, 6), got {states_eme.shape}")

        for j, sc_idx in enumerate(ids):
            sc = all_sc[int(sc_idx)]
            sc.curr_state_eme = states_eme[j, :]

            if set_epochs_from_row is not None:
                col = f"EPOCH_SC_{int(sc_idx) + 1}(jdtdb)"
                if col not in set_epochs_from_row.index:
                    raise KeyError(
                        f"Missing '{col}' in row. Available EPOCH_SC_* cols: "
                        f"{[c for c in set_epochs_from_row.index if str(c).startswith('EPOCH_SC_')]}"
                    )

                epoch_val = set_epochs_from_row[col]
                if pd.isna(epoch_val):
                    raise ValueError(f"'{col}' is NaN for spacecraft {int(sc_idx) + 1}")

                sc.curr_sc_epoch = float(epoch_val)

    def set_spacecraft_pointings(self, boresights_eme, sc_ids=None, *, normalize=True):
        """
        Set spacecraft boresight vectors.

        Parameters
        ----------
        boresights_eme : np.ndarray
            If sc_ids is None: shape (M,3)
            If sc_ids is not None: shape (N,3)
        sc_ids : None, int, sequence of int, or boolean mask
            Which spacecraft to set.
        normalize : bool
            If True, normalize boresights to unit vectors.
        """
        all_sc = self.spacecraft
        M = len(all_sc)

        if sc_ids is None:
            ids = np.arange(M)
        elif isinstance(sc_ids, int):
            ids = np.array([sc_ids], dtype=int)
            boresights_eme = np.atleast_2d(boresights_eme)
        else:
            ids = np.asarray(sc_ids)
            boresights_eme = np.asarray(boresights_eme)

        if boresights_eme.shape != (len(ids), 3):
            raise ValueError(f"boresights_eme must have shape ({len(ids)}, 3), got {boresights_eme.shape}")

        if normalize:
            norms = np.linalg.norm(boresights_eme, axis=1, keepdims=True)
            boresights_eme = boresights_eme / np.clip(norms, 1e-12, None)

        for j, sc_idx in enumerate(ids):
            all_sc[int(sc_idx)].boresight = boresights_eme[j, :]

    def detect(self, asteroid_state, epoch, n_body_prop, configs):
        """
        Generate perfect and noisy RA/Dec measurements for each spacecraft over
        a measurement window.

        Parameters
        ----------
        asteroid_state : array_like, shape (6,)
            Asteroid EME/J2000 state at initial epoch [km, km/s].
        epoch : float
            Initial epoch in JDTDB.
        n_body_prop : NBodyPropagator
            Propagator instance.
        configs : dict
            Configuration dict containing at least:
              - number_of_frames
              - time_between_frames   [s]
              - SECONDS_PER_DAY
              - sigma_ra              [mas]
              - sigma_dec             [mas]
              - sigma_pointing        [mas]

        Returns
        -------
        perfect_meas : ndarray, shape (M, N, 2)
            Perfect [RA, Dec] measurements in radians.
        noisy_meas : ndarray, shape (M, N, 2)
            Noisy [RA, Dec] measurements in radians.
        sc_states : ndarray, shape (M, N, 6)
            Propagated spacecraft states [km, km/s].
        ast_states : ndarray, shape (N, 6)
            Propagated asteroid states [km, km/s].
        epochs : ndarray, shape (N,)
            Epoch sequence used for measurements [JDTDB].
        detection_results : list of dict
            Initial per-spacecraft detection results.
        """

        def mas_to_rad(x_mas):
            """Convert milliarcseconds to radians."""
            return np.asarray(x_mas, dtype=float) * np.pi / (180.0 * 3600.0 * 1000.0)

        asteroid_state = np.asarray(asteroid_state, dtype=float).reshape(6,)

        # ---------------------------------------------------------
        # Initial detection check at the initial epoch
        # ---------------------------------------------------------
        detection_results = [
            sc.asteroid_in_fov_single_epoch(asteroid_state[:3], epoch, configs)
            for sc in self.spacecraft
        ]

        M = len(self.spacecraft)

        # ---------------------------------------------------------
        # Build spacecraft initial state array (M,6)
        # ---------------------------------------------------------
        sc_initial_states = np.zeros((M, 6), dtype=float)
        for i, sc in enumerate(self.spacecraft):
            sc_initial_states[i, :] = np.asarray(sc.curr_state_eme, dtype=float).reshape(6,)

        # ---------------------------------------------------------
        # Measurement sequence
        # ---------------------------------------------------------
        num_frames = int(configs["number_of_frames"])
        step_days = float(configs["time_between_frames"]) / float(configs["SECONDS_PER_DAY"])
        epochs = epoch + step_days * np.arange(num_frames, dtype=float)

        # ---------------------------------------------------------
        # Propagate asteroid and spacecraft over the same epoch grid
        # ---------------------------------------------------------
        # asteroid: (N,6)
        ast_states = n_body_prop.propagate(asteroid_state, epoch, epochs)

        # spacecraft: (N,M,6) from your propagator
        sc_states_time_major = n_body_prop.propagate_multiple_objects(
            sc_initial_states, epoch, epochs
        )

        # reorder to (M,N,6) for convenience
        sc_states = np.transpose(sc_states_time_major, (1, 0, 2))

        N = num_frames

        # ---------------------------------------------------------
        # Allocate outputs
        # ---------------------------------------------------------
        perfect_meas = np.full((M, N, 2), np.nan, dtype=float)
        noisy_meas = np.full((M, N, 2), np.nan, dtype=float)

        # ---------------------------------------------------------
        # Effective noise sigmas (all inputs in mas)
        # ---------------------------------------------------------
        sigma_ra_rad = mas_to_rad(configs.get("sigma_ra", 0.0))
        sigma_dec_rad = mas_to_rad(configs.get("sigma_dec", 0.0))
        sigma_pointing_rad = mas_to_rad(configs.get("sigma_pointing", 0.0))

        sigma_ra_eff = np.sqrt(sigma_ra_rad**2 + sigma_pointing_rad**2)
        sigma_dec_eff = np.sqrt(sigma_dec_rad**2 + sigma_pointing_rad**2)

        eps = 1e-12

        # ---------------------------------------------------------
        # Per-spacecraft measurement generation
        # ---------------------------------------------------------
        for i in range(M):
            detected = bool(detection_results[i].get("detected", False))

            # If not detected at initial epoch, keep all-NaN for this spacecraft
            if not detected:
                continue

            # Relative position asteroid - spacecraft over all frames
            x_rel = ast_states[:, 0] - sc_states[i, :, 0]
            y_rel = ast_states[:, 1] - sc_states[i, :, 1]
            z_rel = ast_states[:, 2] - sc_states[i, :, 2]

            r_xy = np.hypot(x_rel, y_rel)
            r = np.sqrt(r_xy**2 + z_rel**2)

            # Perfect RA/Dec
            ra = np.arctan2(y_rel, x_rel)
            dec = np.arcsin(np.clip(z_rel / np.maximum(r, eps), -1.0, 1.0))

            perfect_meas[i, :, 0] = ra
            perfect_meas[i, :, 1] = dec

            # Add Gaussian noise directly in RA/Dec
            ra_noisy = ra + np.random.normal(loc=0.0, scale=sigma_ra_eff, size=N)
            dec_noisy = dec + np.random.normal(loc=0.0, scale=sigma_dec_eff, size=N)

            # Wrap RA to [-pi, pi)
            ra_noisy = np.arctan2(np.sin(ra_noisy), np.cos(ra_noisy))

            noisy_meas[i, :, 0] = ra_noisy
            noisy_meas[i, :, 1] = dec_noisy

        return perfect_meas, noisy_meas, sc_states, ast_states, epochs, detection_results







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