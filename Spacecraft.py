
import numpy as np
import spiceypy as spice
# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')

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

    def asteroid_in_fov_single_epoch(self, asteroid_position_km, jdtdb, configs):
        """
        Single-epoch asteroid detectability check in geocentric EME/J2000 coordinates.

        Detection requires:
          1) asteroid lies inside the instrument conical FOV
          2) line of sight is not occluded by Earth or Moon
          3) line of sight does not pass through the EMS exclusion sphere

        Parameters
        ----------
        asteroid_position_km : array-like, shape (3,)
            Asteroid position in geocentric EMEJ2000/J2000 coordinates, km.
        jdtdb : float
            Epoch in Julian Date TDB.
        configs : dict
            Config dictionary. Expected keys:
                EARTH_RADIUS_KM : float
                MOON_RADIUS_KM  : float
            Optional EMS keys:
                p_em : [x, y, z] in km, in same geocentric frame
                R_em : float, radius in km
                alpha_s_deg : float, angular safety margin in degrees

        Returns
        -------
        out : dict
            Dictionary with detailed detection flags:
                detected : bool
                in_fov : bool
                occluded_earth : bool
                occluded_moon : bool
                occluded_em : bool
                occluded_ems : bool
                visible_base : bool
                visible_ems_filtered : bool
                cos_angle : float
                cos_theta_h : float
                sep_ems_rad : float or None
                beta_ems_rad : float or None
        """

        def fov_deg2_to_half_angle_rad(FOV_deg2):
            """
            Convert sky area FOV (deg^2) to cone half-angle (radians)
            using spherical cap geometry.
            """
            return np.arccos(
                1.0 - (FOV_deg2 / (180.0 / np.pi) ** 2) / (2.0 * np.pi)
            )

        def check_occlusion_single(spacecraft_pos, asteroid_pos, body_pos, body_radius):
            """
            Single-epoch version of your old batch occlusion logic.
            Returns True if the body occludes the LOS from spacecraft to asteroid.
            """
            sc_to_ast = asteroid_pos - spacecraft_pos
            sc_to_body = body_pos - spacecraft_pos

            sc_to_ast_norm = np.linalg.norm(sc_to_ast)
            if sc_to_ast_norm <= 1e-15:
                return False

            proj_length = np.dot(sc_to_body, sc_to_ast) / sc_to_ast_norm
            proj_point = spacecraft_pos + (sc_to_ast / sc_to_ast_norm) * proj_length

            min_distance = np.linalg.norm(proj_point - body_pos)
            occluded = (
                    (min_distance < body_radius)
                    and (proj_length > 0.0)
                    and (proj_length < sc_to_ast_norm)
            )
            return bool(occluded)

        # -----------------------------
        # inputs
        # -----------------------------
        asteroid_position_km = np.asarray(asteroid_position_km, dtype=float).reshape(3, )
        sc_pos_km = np.asarray(self.curr_state_eme[:3], dtype=float).reshape(3, )

        # -----------------------------
        # Earth / Moon positions
        # geocentric frame:
        #   Earth = origin
        #   Moon queried relative to Earth
        # -----------------------------
        et = spice.unitim(jdtdb, 'JDTDB', 'ET')

        earth_pos_km = np.zeros(3, dtype=float)

        moon_state_km, _ = spice.spkezr("MOON", et, "J2000", "NONE", "EARTH")
        moon_pos_km = np.asarray(moon_state_km[:3], dtype=float)

        earth_radius_km = float(configs["EARTH_RADIUS_KM"])
        moon_radius_km = float(configs["MOON_RADIUS_KM"])

        # -----------------------------
        # FOV test
        # -----------------------------
        theta_h = fov_deg2_to_half_angle_rad(self.fov)
        cos_theta_h = np.cos(theta_h)

        b = np.asarray(self.boresight, dtype=float).reshape(3, )
        print("single check")
        print(b)
        print(asteroid_position_km)
        print(jdtdb)
        print(sc_pos_km)
        b_norm = np.linalg.norm(b)
        if b_norm <= 1e-15:
            raise ValueError("self.boresight has zero norm.")
        b = b / b_norm

        rel_pos = asteroid_position_km - sc_pos_km
        rel_norm = np.linalg.norm(rel_pos)

        if rel_norm <= 1e-15:
            return {
                "detected": False,
                "in_fov": False,
                "occluded_earth": False,
                "occluded_moon": False,
                "occluded_em": False,
                "occluded_ems": False,
                "visible_base": False,
                "visible_ems_filtered": False,
                "cos_angle": None,
                "cos_theta_h": float(cos_theta_h),
                "sep_ems_rad": None,
                "beta_ems_rad": None,
            }

        vhat = rel_pos / rel_norm
        cos_angle = float(np.dot(vhat, b))
        in_fov = bool(cos_angle >= cos_theta_h)

        # -----------------------------
        # Earth / Moon occlusion
        # -----------------------------
        occluded_earth = check_occlusion_single(
            sc_pos_km, asteroid_position_km, earth_pos_km, earth_radius_km
        )
        occluded_moon = check_occlusion_single(
            sc_pos_km, asteroid_position_km, moon_pos_km, moon_radius_km
        )
        occluded_em = bool(occluded_earth or occluded_moon)

        visible_base = bool(in_fov and (not occluded_em))

        # -----------------------------
        # EMS exclusion sphere
        # -----------------------------
        p_em_km = np.asarray(configs.get("p_em", [0.0, 0.0, 0.0]), dtype=float).reshape(3, )
        R_em_km = float(configs.get("R_em", 0.0))
        alpha_s_deg = float(configs.get("alpha_s_deg", 0.0))

        occluded_ems = False
        sep_ems_rad = None
        beta_ems_rad = None

        if R_em_km > 0.0:
            alpha_s = np.deg2rad(alpha_s_deg)

            c_vec = p_em_km - sc_pos_km
            c_dist = np.linalg.norm(c_vec)

            # If spacecraft is inside the EMS sphere, treat as blocked
            if c_dist <= R_em_km:
                occluded_ems = True
                sep_ems_rad = None
                beta_ems_rad = None
            elif c_dist > 1e-15:
                c_hat = c_vec / c_dist
                dot_uc = np.clip(np.dot(vhat, c_hat), -1.0, 1.0)
                sep_ems_rad = float(np.arccos(dot_uc))
                beta_ems_rad = float(np.arcsin(np.clip(R_em_km / c_dist, 0.0, 1.0)))

                occluded_ems = bool(sep_ems_rad <= (beta_ems_rad + alpha_s))

        visible_ems_filtered = bool(visible_base and (not occluded_ems))

        return {
            "detected": visible_ems_filtered,
            "in_fov": in_fov,
            "occluded_earth": occluded_earth,
            "occluded_moon": occluded_moon,
            "occluded_em": occluded_em,
            "occluded_ems": occluded_ems,
            "visible_base": visible_base,
            "visible_ems_filtered": visible_ems_filtered,
            "cos_angle": float(cos_angle),
            "cos_theta_h": float(cos_theta_h),
            "sep_ems_rad": sep_ems_rad,
            "beta_ems_rad": beta_ems_rad,
        }

    def asteroid_in_fov_batch_km_geocentric(
            self,
            asteroid_trajectory_km,
            spacecraft_trajectory_km,
            spacecraft_boresight_eme,
            jdtdb_list,
            configs,
    ):
        """
        Determine when the asteroid is in the field of view, in geocentric EME/J2000 km coordinates,
        considering:
          1) conical FOV test using time-varying spacecraft boresight
          2) Earth/Moon occlusion
          3) EMS exclusion sphere occlusion with angular margin alpha_s_deg

        Parameters
        ----------
        asteroid_trajectory_km : array-like, shape (N,3)
            Asteroid positions in geocentric EMEJ2000/J2000 coordinates, km.
        spacecraft_trajectory_km : array-like, shape (N,3)
            Spacecraft positions in geocentric EMEJ2000/J2000 coordinates, km.
        spacecraft_boresight_eme : array-like, shape (N,3) or (3,)
            Spacecraft boresight vectors in geocentric EMEJ2000/J2000 coordinates.
            If shape is (3,), the same boresight is used for all epochs.
        jdtdb_list : array-like, shape (N,)
            Epochs in Julian Date TDB, one per sample.
        configs : dict
            Config dictionary. Expected keys:
                EARTH_RADIUS_KM : float
                MOON_RADIUS_KM  : float
            Optional EMS keys:
                p_em : [x, y, z] in km, same geocentric frame
                R_em : float, radius in km
                alpha_s_deg : float, angular safety margin in degrees

        Returns
        -------
        result_base : ndarray, shape (N,)
            Float array with index where visible under FOV + Earth/Moon rules, -1 otherwise.
        result_ems_filtered : ndarray, shape (N,)
            Float array with index where visible after EMS filtering, -1 otherwise.
        """

        def fov_deg2_to_half_angle_rad(FOV_deg2):
            """
            Convert sky area FOV (deg^2) to cone half-angle (radians)
            using spherical cap geometry.
            """
            return np.arccos(
                1.0 - (FOV_deg2 / (180.0 / np.pi) ** 2) / (2.0 * np.pi)
            )

        def check_occlusion_batch(spacecraft_pos, asteroid_pos, body_pos, body_radius):
            """
            Vectorized LOS occlusion test for a spherical body.

            Parameters
            ----------
            spacecraft_pos : (N,3)
            asteroid_pos   : (N,3)
            body_pos       : (N,3)
            body_radius    : float

            Returns
            -------
            occluded : (N,) bool
            """
            sc_to_ast = asteroid_pos - spacecraft_pos
            sc_to_body = body_pos - spacecraft_pos

            sc_to_ast_norm = np.linalg.norm(sc_to_ast, axis=1)
            valid = sc_to_ast_norm > 1e-15

            proj_length = np.zeros_like(sc_to_ast_norm)
            proj_length[valid] = (
                    np.einsum("ij,ij->i", sc_to_body[valid], sc_to_ast[valid])
                    / sc_to_ast_norm[valid]
            )

            proj_point = spacecraft_pos.copy()
            proj_point[valid] = (
                    spacecraft_pos[valid]
                    + (sc_to_ast[valid] / sc_to_ast_norm[valid, None]) * proj_length[valid, None]
            )

            min_distance = np.linalg.norm(proj_point - body_pos, axis=1)

            occluded = (
                    valid
                    & (min_distance < body_radius)
                    & (proj_length > 0.0)
                    & (proj_length < sc_to_ast_norm)
            )
            return occluded

        # -----------------------------
        # inputs
        # -----------------------------
        positions = np.asarray(asteroid_trajectory_km, dtype=float)
        sc_pos = np.asarray(spacecraft_trajectory_km, dtype=float)
        boresight = np.asarray(spacecraft_boresight_eme, dtype=float)
        jdtdb = np.asarray(jdtdb_list, dtype=float).reshape(-1)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"asteroid_trajectory_km must have shape (N,3), got {positions.shape}")
        if sc_pos.ndim != 2 or sc_pos.shape[1] != 3:
            raise ValueError(f"spacecraft_trajectory_km must have shape (N,3), got {sc_pos.shape}")
        if jdtdb.ndim != 1:
            raise ValueError(f"jdtdb_list must have shape (N,), got {jdtdb.shape}")

        N = positions.shape[0]
        if sc_pos.shape[0] != N or jdtdb.shape[0] != N:
            raise ValueError(
                f"Inconsistent lengths: asteroid={positions.shape[0]}, "
                f"spacecraft={sc_pos.shape[0]}, jdtdb={jdtdb.shape[0]}"
            )

        # Boresight can be either (3,) or (N,3)
        if boresight.ndim == 1:
            if boresight.shape[0] != 3:
                raise ValueError(
                    f"spacecraft_boresight_eme must have shape (3,) or (N,3), got {boresight.shape}"
                )
            boresight = np.broadcast_to(boresight.reshape(1, 3), (N, 3)).copy()
        elif boresight.ndim == 2:
            if boresight.shape != (N, 3):
                raise ValueError(
                    f"spacecraft_boresight_eme must have shape (3,) or ({N},3), got {boresight.shape}"
                )
        else:
            raise ValueError(
                f"spacecraft_boresight_eme must have shape (3,) or (N,3), got {boresight.shape}"
            )

        # Normalize boresights
        boresight_norm = np.linalg.norm(boresight, axis=1)
        if np.any(boresight_norm <= 1e-15):
            bad_idx = np.where(boresight_norm <= 1e-15)[0]
            raise ValueError(f"spacecraft_boresight_eme has zero-norm vectors at indices {bad_idx.tolist()}")

        b_hat = boresight / boresight_norm[:, None]

        # -----------------------------
        # Earth / Moon positions
        # geocentric frame:
        #   Earth = origin
        #   Moon queried relative to Earth
        # -----------------------------
        earth_pos = np.zeros((N, 3), dtype=float)

        et_list = []
        moon_pos = np.empty((N, 3), dtype=float)

        for i, jd in enumerate(jdtdb):
            et = spice.unitim(jd, 'JDTDB', 'ET')
            et_list.append(et)
            moon_state_km, _ = spice.spkezr("MOON", et, "J2000", "NONE", "EARTH")
            moon_pos[i, :] = np.asarray(moon_state_km[:3], dtype=float)

        earth_radius_km = float(configs["EARTH_RADIUS_KM"])
        moon_radius_km = float(configs["MOON_RADIUS_KM"])

        # -----------------------------
        # FOV test
        # -----------------------------
        theta_h = fov_deg2_to_half_angle_rad(self.fov)
        cos_theta_h = np.cos(theta_h)

        rel_pos = positions - sc_pos
        rel_norm = np.linalg.norm(rel_pos, axis=1)

        valid_rel = rel_norm > 1e-15
        vhat = np.zeros_like(rel_pos)
        vhat[valid_rel] = rel_pos[valid_rel] / rel_norm[valid_rel, None]

        cos_angles = np.full(N, -np.inf, dtype=float)
        cos_angles[valid_rel] = np.einsum("ij,ij->i", vhat[valid_rel], b_hat[valid_rel])
        in_fov = valid_rel & (cos_angles >= cos_theta_h)

        # -----------------------------
        # Earth / Moon occlusion
        # -----------------------------
        occluded_earth = check_occlusion_batch(sc_pos, positions, earth_pos, earth_radius_km)
        occluded_moon = check_occlusion_batch(sc_pos, positions, moon_pos, moon_radius_km)
        occluded_em = occluded_earth | occluded_moon

        visible_base = in_fov & (~occluded_em)

        result_base = np.full(N, -1.0, dtype=float)
        base_idx = np.where(visible_base)[0]
        result_base[base_idx] = base_idx

        # -----------------------------
        # EMS exclusion sphere
        # -----------------------------
        p_em_km = np.asarray(configs.get("p_em", [0.0, 0.0, 0.0]), dtype=float).reshape(3, )
        R_em_km = float(configs.get("R_em", 0.0))
        alpha_s_deg = float(configs.get("alpha_s_deg", 0.0))

        if R_em_km <= 0.0:
            result_ems_filtered = result_base.copy()
            return result_base, result_ems_filtered

        alpha_s = np.deg2rad(alpha_s_deg)

        c_vec = p_em_km[None, :] - sc_pos
        c_dist = np.linalg.norm(c_vec, axis=1)

        valid_c = c_dist > 1e-15
        c_hat = np.zeros_like(c_vec)
        c_hat[valid_c] = c_vec[valid_c] / c_dist[valid_c, None]

        dot_uc = np.zeros(N, dtype=float)
        both_valid = valid_rel & valid_c
        dot_uc[both_valid] = np.einsum("ij,ij->i", vhat[both_valid], c_hat[both_valid])
        dot_uc = np.clip(dot_uc, -1.0, 1.0)

        sep = np.full(N, np.inf, dtype=float)
        sep[both_valid] = np.arccos(dot_uc[both_valid])

        inside_sphere = c_dist <= R_em_km

        beta = np.zeros(N, dtype=float)
        beta[valid_c] = np.arcsin(np.clip(R_em_km / c_dist[valid_c], 0.0, 1.0))

        occluded_ems = inside_sphere | (sep <= (beta + alpha_s))

        visible_ems_filtered = visible_base & (~occluded_ems)

        result_ems_filtered = np.full(N, -1.0, dtype=float)
        idx2 = np.where(visible_ems_filtered)[0]
        result_ems_filtered[idx2] = idx2

        return result_base, result_ems_filtered


