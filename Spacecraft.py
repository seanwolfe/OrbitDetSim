
import numpy as np

class Spacecraft:


    def __init__(self, ini_pos, ini_pos_index, configs, current_state_eme=None, current_spacecraftepoch=None, current_boresight=None):
        self.ini_position = ini_pos  # initial position of the spacecraft in the quasi-halo orbit
        self.ini_pos_index = ini_pos_index  # initial position index in the quasi-halo orbit csv
        self.velocity = None
        self.position = None
        self.boresight = np.array([-1, 0, 0]) if current_boresight is None else current_boresight
        self.pixel_scale = configs['pixel_scale']
        self.fov = configs['fov']
        self.number_of_pixels = configs['number_of_pixels']
        self.reaction_wheel_torque = configs['reaction_wheel_torque']
        self.reaction_wheel_momentum = configs['reaction_wheel_momentum']
        self.mass = configs['mass']
        self.length = configs['length']
        self.telescope_diameter = configs['telescope_diameter']
        self.telescope_mass = configs['telescope_mass']
        self.telescope_offset = configs['telescope_offset']
        self.sigma_ra = configs['sigma_ra']
        self.sigma_dec = configs['sigma_dec']
        self.sigma_pointing = configs['sigma_pointing']
        self.matched_trajectory = None  # this contains an array of the trajectory of the sc that has same length as the asteroid traj in question
        self.matched_trajectory_full = None
        self.curr_state_eme = current_state_eme
        self.curr_sc_epoch = current_spacecraftepoch
        return


    def set_state(self, position, velocity):
        self.position = position
        self.velocity = velocity
        return

    def get_spacecraft_pos(self, index):
        return self.matched_trajectory[index, :]


    def get_attitude(self):
        raise NotImplementedError


    def is_occluded_batch(self, spacecraft_pos, asteroid_pos, earth_pos, moon_pos, configs):
        """Vectorized occlusion check for all time steps."""
        def check_occlusion(body_pos, body_radius):
            sc_to_ast = asteroid_pos - spacecraft_pos  # (N,3)
            sc_to_body = body_pos - spacecraft_pos  # (N,3)

            proj_length = np.einsum('ij,ij->i', sc_to_body, sc_to_ast) / np.linalg.norm(sc_to_ast, axis=1)
            proj_point = spacecraft_pos + (sc_to_ast / np.linalg.norm(sc_to_ast, axis=1)[:, None]) * proj_length[:, None]

            min_distance = np.linalg.norm(proj_point - body_pos, axis=1)
            occluded = (min_distance < body_radius) & (proj_length > 0) & (proj_length < np.linalg.norm(sc_to_ast, axis=1))
            return occluded

        return (check_occlusion(earth_pos, configs['EARTH_RADIUS_KM'] * configs['KM_TO_M'] / configs['AU_TO_M'])
                | check_occlusion(moon_pos, configs['MOON_RADIUS_KM'] * configs['KM_TO_M']  / configs['AU_TO_M']))

    def asteroid_in_fov_batch_old(self,
                              asteroid_trajectory,  # (N,3) in same frame as s/c & boresight
                              spacecraft_position,  # (N,3)
                              earth_position,  # (N,3)
                              moon_position,  # (N,3)
                              configs):
        """
        Determine when the asteroid is inside the spacecraft's conical FOV,
        accounting for Earth/Moon occlusion.

        Returns:
            result (N,) array:
                index where visible, -1 where not visible
        """

        positions = np.asarray(asteroid_trajectory, dtype=float)
        sc_pos = np.asarray(spacecraft_position, dtype=float)

        # ---- Correct half-angle from FOV area ----
        def fov_deg2_to_half_angle_rad(FOV_deg2):
            """
            Convert sky area FOV (deg^2) to cone half-angle (radians)
            using spherical cap geometry.
            """
            return np.arccos(
                1.0 - (FOV_deg2 / (180.0 / np.pi) ** 2) / (2.0 * np.pi)
            )
        theta_h = fov_deg2_to_half_angle_rad(self.fov)
        cos_theta_h = np.cos(theta_h)

        # ---- Normalize boresight once ----
        b = np.asarray(self.boresight, dtype=float)
        b = b / (np.linalg.norm(b) + 1e-15)

        # ---- Relative vectors from s/c to asteroid ----
        rel_pos = positions - sc_pos  # (N,3)
        rel_norm = np.linalg.norm(rel_pos, axis=1)  # (N,)

        # ---- Use cosine test instead of angle ----
        vhat = rel_pos / (rel_norm[:, None] + 1e-15)
        cos_angles = vhat @ b  # (N,)

        # Inside conical FOV
        in_fov = cos_angles >= cos_theta_h

        # ---- Occlusion check (unchanged) ----
        occluded = self.is_occluded_batch(
            spacecraft_position,
            positions,
            earth_position,
            moon_position,
            configs
        )

        visible = in_fov & (~occluded)

        # ---- Output format exactly like your original ----
        result = np.full(positions.shape[0], -1.0, dtype=float)
        visible_indices = np.where(visible)[0]
        result[visible_indices] = visible_indices

        return result


    def asteroid_in_fov_batch(self, asteroid_trajectory, spacecraft_position,
                              earth_position, moon_position, configs):
        """
        Determine when the asteroid is in the field of view, considering:
          1) conical FOV test
          2) Earth/Moon occlusion (existing is_occluded_batch)
          3) EMS exclusion sphere occlusion (new), with angular margin alpha_s_deg

        Parameters (as in your original):
            asteroid_trajectory (N,3) in AU
            spacecraft_position (N,3) in AU
            earth_position (N,3) in AU
            moon_position (N,3) in AU
            configs: YAML config dict containing:
                p_em: [x,y,z] in km
                R_em: radius in km
                alpha_s_deg: margin in degrees

        Returns:
            result_base (N,) float array: index where visible, -1 where not visible
            result_ems_filtered (N,) float array: same, but also filtered by EMS exclusion
        """
        positions = np.asarray(asteroid_trajectory, dtype=float)  # (N,3) AU
        sc_pos = np.asarray(spacecraft_position, dtype=float)  # (N,3) AU

        N = positions.shape[0]

        AU_KM = 149_597_870.7
        def fov_deg2_to_half_angle_rad(FOV_deg2):
            """
            Convert sky area FOV (deg^2) to cone half-angle (radians)
            using spherical cap geometry.
            """
            return np.arccos(
                1.0 - (FOV_deg2 / (180.0 / np.pi) ** 2) / (2.0 * np.pi)
            )
        theta_h = fov_deg2_to_half_angle_rad(self.fov)
        cos_theta_h = np.cos(theta_h)

        # ---- Normalize boresight (assumed in same frame) ----
        b = np.asarray(self.boresight, dtype=float).reshape(3, )
        b = b / (np.linalg.norm(b) + 1e-15)

        # ---- Relative LOS spacecraft -> asteroid ----
        rel_pos = positions - sc_pos  # (N,3)
        rel_norm = np.linalg.norm(rel_pos, axis=1)  # (N,)
        vhat = rel_pos / (rel_norm[:, None] + 1e-15)  # (N,3)

        # ---- Conical FOV test via cosine threshold ----
        cos_angles = vhat @ b  # (N,)
        in_fov = cos_angles >= cos_theta_h

        # ---- Existing occlusion (Earth/Moon) ----
        occluded_em = self.is_occluded_batch(sc_pos, positions, earth_position, moon_position, configs)

        # Base visibility (your current logic)
        visible_base = in_fov & (~occluded_em)

        # Build base result array
        result_base = np.full(N, -1.0, dtype=float)
        base_idx = np.where(visible_base)[0]
        result_base[base_idx] = base_idx

        # ---------------------------------------------------------
        # NEW: EMS exclusion sphere occlusion
        # ---------------------------------------------------------
        # Read EMS config
        p_em_km = np.asarray(configs.get("p_em", [0.0, 0.0, 0.0]), dtype=float).reshape(3, )
        R_em_km = float(configs.get("R_em", 0.0))
        alpha_s_deg = float(configs.get("alpha_s_deg", 0.0))

        # If R_em <= 0, treat as disabled (no extra filtering)
        if R_em_km <= 0.0:
            result_ems_filtered = result_base.copy()
            return result_base, result_ems_filtered

        # Convert EMS center/radius to AU (inputs are AU, config is km)
        p_em = p_em_km / AU_KM
        R_em = R_em_km / AU_KM
        alpha_s = np.deg2rad(alpha_s_deg)

        # Vector spacecraft -> EMS center
        c_vec = p_em[None, :] - sc_pos  # (N,3)
        c_dist = np.linalg.norm(c_vec, axis=1)  # (N,)
        c_hat = c_vec / (c_dist[:, None] + 1e-15)  # (N,3)

        # Separation angle between LOS-to-asteroid and LOS-to-EMS-center
        dot_uc = np.einsum("ij,ij->i", vhat, c_hat)  # (N,)
        dot_uc = np.clip(dot_uc, -1.0, 1.0)
        sep = np.arccos(dot_uc)  # (N,)

        # Apparent angular radius of EMS sphere as seen from spacecraft
        # If spacecraft is inside the sphere (c_dist < R_em), treat as fully occluded.
        inside_sphere = c_dist <= R_em

        # arcsin argument must be <= 1
        arg = np.zeros_like(c_dist)
        valid = c_dist > 1e-15
        arg[valid] = np.clip(R_em / c_dist[valid], 0.0, 1.0)
        beta = np.arcsin(arg)  # (N,)

        # Occluded by EMS if LOS passes within (beta + alpha_s) of the EMS center direction
        occluded_ems = inside_sphere | (sep <= (beta + alpha_s))

        # Final visibility with EMS exclusion applied
        visible_ems_filtered = visible_base & (~occluded_ems)

        result_ems_filtered = np.full(N, -1.0, dtype=float)
        idx2 = np.where(visible_ems_filtered)[0]
        result_ems_filtered[idx2] = idx2

        return result_base, result_ems_filtered


