import numpy as np

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
MAS_TO_RAD    = ARCSEC_TO_RAD / 1000.0

# ---------------------------------
# Setup Scenario from IOD for my framework
# ----------------------------------
def od_setup_from_iod(config, row, *, util, sp, eps=1e-12):
    """
    Build OD/UKF initial conditions + per-row context from a MASTER row.

    Returns
    -------
    setup : dict with keys
      - M
      - sc_detecting_id
      - epochs: dict of ae_jdtdb, ae_et
      - frames: dict containing various states/vectors in EME/SECR/...
      - iod: dict containing IOD states (eme, secr, etc) and clamping info
      - x0_eme_kms : (6,) initial asteroid state in EME (km, km/s)
      - P0_eme     : (6,6) initial covariance in EME Cartesian (km^2, (km/s)^2)
    """

    M = int(config["num_spacecraft"])

    # Detecting spacecraft id (convert 1-based -> 0-based)
    sc_detecting_id = int(row["DETECTING_SC_ID"]) - 1
    if sc_detecting_id < 0 or sc_detecting_id >= M:
        raise ValueError(f"DETECTING_SC_ID out of range: {row['DETECTING_SC_ID']} for M={M}")

    # -----------------------------
    # Parse asteroid heliocentric state at AE (ECLIPJ2000)
    # -----------------------------
    ast_helio_ae_kms = util.parse_vec_cell(row["HELIO_AST(kms)"])

    # AE epoch and Earth heliocentric state at AE
    ae_jdtdb = row["EPOCH_AST(jdtdb)"]
    ae_et = sp.unitim(ae_jdtdb, "JDTDB", "ET")
    reference_body = 10  # Sun
    body = 399  # Earth
    earth_helio_ae_kms, _ = sp.spkgeo(body, ae_et, "ECLIPJ2000", reference_body)

    # Convert asteroid heliocentric -> GEO EME at AE (for UKF)
    ast_eme_ae_kms = util.helio_eclip_to_geo_eme_generic(
        ast_helio_ae_kms, earth_helio_ae_kms, layout="batch"
    )
    # Also GEO SECR at AE (for interpretation/visualization)
    ast_secr_ae_kms = util.helio_eclip_to_geo_secr_generic(
        ast_helio_ae_kms, earth_helio_ae_kms, layout="batch"
    )

    # -----------------------------
    # Spacecraft heliocentric states at each spacecraft epoch SE (ECLIPJ2000)
    # and corresponding Earth heliocentric at each SE epoch
    # -----------------------------
    sc_helio_se_kms = np.zeros((M, 6), dtype=float)
    earth_helio_se_kms = np.zeros((M, 6), dtype=float)

    for i in range(M):
        sc_str = f"HELIO_SC_{i + 1}(kms)"
        sc_helio_se_kms[i, :] = util.parse_vec_cell(row[sc_str])

        se_str = f"EPOCH_SC_{i + 1}(jdtdb)"
        se_jdtdb = row[se_str]
        se_et = sp.unitim(se_jdtdb, "JDTDB", "ET")

        earth_helio_se_i, _ = sp.spkgeo(body, se_et, "ECLIPJ2000", reference_body)
        earth_helio_se_kms[i, :] = earth_helio_se_i

    # Spacecraft boresight in GEO SECR (cartesian unit-ish vectors, presumably)
    sc_pointing_sunearth_cartesian = np.zeros((M, 3), dtype=float)
    for i in range(M):
        sc_point_str = f"BORESIGHT_SC_{i + 1}_GEO_SECR"
        sc_pointing_sunearth_cartesian[i, :] = util.parse_vec_cell(row[sc_point_str])

    # -----------------------------
    # Convert spacecraft heliocentric-> GEO SECR at their SE epochs
    # -----------------------------
    sc_secr_se_kms = np.zeros((M, 6), dtype=float)
    for i in range(M):
        sc_secr_se_kms[i, :] = util.helio_eclip_to_geo_secr_generic(
            sc_helio_se_kms[i, :],
            earth_helio_se_kms[i, :],
            layout="batch",
            obj_hint="(batch, 6)"
        )

    # -----------------------------
    # Convert spacecraft GEO SECR -> GEO ECLIP at AE epoch, then -> GEO EME
    # (this is your "interpret everything at AE" choice)
    # -----------------------------
    sc_geoeclip_ae_kms = util.geo_secr_to_geo_eclip_generic(
        sc_secr_se_kms, earth_helio_ae_kms, layout="batch"
    )
    sc_eme_ae_kms = util.geo_eclip_to_geo_eme_generic(sc_geoeclip_ae_kms, layout="batch")

    # Convert pointing from SECR -> GEO ECLIP -> GEO EME (at AE)
    sc_pointing_geoeclip_cartesian = util.geo_secr_to_geo_eclip_generic(
        sc_pointing_sunearth_cartesian,
        earth_helio_ae_kms,
        layout="batch",
        obj_hint="(batch, 3)"
    )
    sc_pointing_eme_cartesian = util.geo_eclip_to_geo_eme_generic(
        sc_pointing_geoeclip_cartesian,
        layout="batch",
        hint="(batch, 3)"
    )

    # Optional: projected angles (CCW from +x) in SECR & EME
    sc_pointing_secr_angle_rad = util.proj_angle_xy_from_plus_x_ccw(sc_pointing_sunearth_cartesian)
    sc_pointing_eme_angle_rad = util.proj_angle_xy_from_plus_x_ccw(sc_pointing_eme_cartesian)

    # -----------------------------
    # IOD solution + frame conversions
    # -----------------------------
    ast_iod_eme_ae_kms = util.parse_vec_cell(row["IOD_FINAL_STATE"])

    # EME -> GEO ECLIP -> HELIO ECLIP -> GEO SECR (at AE)
    ast_iod_geoeclip_ae_kms = util.geo_eme_to_geo_eclip_generic(ast_iod_eme_ae_kms)
    ast_iod_helioeclip_ae_kms = ast_iod_geoeclip_ae_kms + earth_helio_ae_kms
    ast_iod_secr_ae_kms = util.helio_eclip_to_geo_secr_generic(
        ast_iod_helioeclip_ae_kms, earth_helio_ae_kms, layout="batch"
    )

    # -----------------------------
    # Clamp IOD if out of FOV (SECR + EME)
    # -----------------------------
    # ---- Correct half-angle from FOV area ----
    theta_h_rad = util.fov_deg2_to_half_angle_rad(config["fov"])

    # --- SECR clamp ---
    sc_pos_secr = sc_secr_se_kms[sc_detecting_id, :3]
    u_bore_secr = sc_pointing_sunearth_cartesian[sc_detecting_id, :]
    iod_pos_secr = ast_iod_secr_ae_kms[:3].copy()

    iod_pos_secr_clamped, inside_secr, info_secr = util.clamp_point_into_fov_cone(
        iod_pos_secr,
        sc_pos_xyz=sc_pos_secr,
        boresight_u_xyz=u_bore_secr,
        theta_h_rad=theta_h_rad,
    )
    if not inside_secr:
        ast_iod_secr_ae_kms[:3] = iod_pos_secr_clamped

    # --- EME clamp ---
    sc_pos_eme = sc_eme_ae_kms[sc_detecting_id, :3]
    u_bore_eme = sc_pointing_eme_cartesian[sc_detecting_id, :]
    iod_pos_eme = ast_iod_eme_ae_kms[:3].copy()

    iod_pos_eme_clamped, inside_eme, info_eme = util.clamp_point_into_fov_cone(
        iod_pos_eme,
        sc_pos_xyz=sc_pos_eme,
        boresight_u_xyz=u_bore_eme,
        theta_h_rad=theta_h_rad,
    )
    if not inside_eme:
        ast_iod_eme_ae_kms[:3] = iod_pos_eme_clamped

    # -----------------------------
    # Topocentric RA/Dec/rho (for each s/c) from the IOD state
    # -----------------------------
    ast_iod_topoeme_radecrho_radkms = util.topocentric_alpha_delta_rho_6d(
        ast_iod_eme_ae_kms[:3], ast_iod_eme_ae_kms[3:],
        sc_eme_ae_kms[:, :3], sc_eme_ae_kms[:, 3:]
    )
    ast_iod_toposecr_radecrho_radkms = util.topocentric_alpha_delta_rho_6d(
        ast_iod_secr_ae_kms[:3], ast_iod_secr_ae_kms[3:],
        sc_secr_se_kms[:, :3], sc_secr_se_kms[:, 3:]
    )

    # -----------------------------
    # Initial covariance: topo (ra,dec,rho,...) -> cartesian (xyz,xyz_dot)
    #
    # NOTE: Your util.cov_radec_rho_6d_to_xyz_6d currently returns something.
    # If it returns (M,6,6), you must choose how to initialize a SINGLE asteroid
    # covariance for the UKF (e.g., use detecting SC only, or fuse across SC).
    # -----------------------------
    topo_std_degkms = np.asarray(config["iod_cov_topo_std"], dtype=float)  # (6,)
    topo_std_radkms = util.topo_std_degkms_to_radkms(topo_std_degkms)
    P_topo = np.diag(topo_std_radkms ** 2)  # (6,6)

    P_cart_eme = util.cov_radec_rho_6d_to_xyz_6d(
        ast_iod_topoeme_radecrho_radkms,  # (M,6)
        P_topo
    )
    P_cart_secr = util.cov_radec_rho_6d_to_xyz_6d(
        ast_iod_toposecr_radecrho_radkms,  # (M,6)
        P_topo
    )

    # Pick a single (6,6) for the filter.
    # Common choice: use the detecting spacecraft only.
    if P_cart_eme.ndim == 3:
        P0_eme = P_cart_eme[sc_detecting_id, :, :]
    else:
        P0_eme = P_cart_eme

    # Initial UKF state:
    x0_eme_kms = ast_iod_eme_ae_kms.astype(float).reshape(6)

    setup = {
        "M": M,
        "sc_detecting_id": sc_detecting_id,
        "epochs": {
            "ae_jdtdb": ae_jdtdb,
            "ae_et": ae_et,
        },
        "frames": {
            "earth_helio_ae_kms": earth_helio_ae_kms,
            "ast_helio_ae_kms": ast_helio_ae_kms,
            "ast_eme_ae_kms": ast_eme_ae_kms,
            "ast_secr_ae_kms": ast_secr_ae_kms,
            "sc_helio_se_kms": sc_helio_se_kms,
            "earth_helio_se_kms": earth_helio_se_kms,
            "sc_secr_se_kms": sc_secr_se_kms,
            "sc_eme_ae_kms": sc_eme_ae_kms,
            "sc_pointing_sunearth_cartesian": sc_pointing_sunearth_cartesian,
            "sc_pointing_eme_cartesian": sc_pointing_eme_cartesian,
            "sc_pointing_secr_angle_rad": sc_pointing_secr_angle_rad,
            "sc_pointing_eme_angle_rad": sc_pointing_eme_angle_rad,
        },
        "iod": {
            "ast_iod_eme_ae_kms": ast_iod_eme_ae_kms,
            "ast_iod_secr_ae_kms": ast_iod_secr_ae_kms,
            "clamp": {
                "inside_secr": inside_secr,
                "info_secr": info_secr,
                "inside_eme": inside_eme,
                "info_eme": info_eme,
            },
            "topo": {
                "topoeme_radecrho_radkms": ast_iod_topoeme_radecrho_radkms,
                "toposecr_radecrho_radkms": ast_iod_toposecr_radecrho_radkms,
            },
            "cov": {
                "P_topo_radkms": P_topo,
                "P_cart_eme": P_cart_eme,
                "P_cart_secr": P_cart_secr,
            },
        },
        "x0_eme_kms": x0_eme_kms,
        "P0_eme": P0_eme,
    }

    return setup


class OD_UKF:
    """
    Skeleton UKF for orbit determination with state x=[r; v] in EME/J2000.

    - Process model: user-provided n-body propagator for each sigma point.
    - Process noise: RTN-defined continuous-time white acceleration spectral density,
      mapped to EME/J2000 and discretized to Q(dt).
    - Measurement model: angles-only as unit LOS vector z = rho_hat (3x1).
      Measurement noise: given sigma_ra/sigma_dec (+ sigma_pointing), mapped to
      unit-vector covariance R(t) via Jacobian.

    This version updates predict() to an AUGMENTED UKF predict, and supports:
      - t1 as a scalar epoch -> returns (x_pred, P_pred)
      - t1 as a 1D array/list of epochs -> returns (X_pred, P_pred) with shapes
            X_pred: (K,6), P_pred: (K,6,6)
        using SEQUENTIAL prediction (each step uses previous predicted (x,P)).
    """

    def __init__(
        self,
        x0,
        P0,
        *,
        Sa_rtn=(0.0, 0.0, 0.0),            # (Sa_R, Sa_T, Sa_N) in (km/s^2)^2 / s
        meas_units="arcsec",               # "rad" | "arcsec" | "mas"
        sigma_ra=1.0,                      # in meas_units
        sigma_dec=1.0,                     # in meas_units
        sigma_pointing=0.0,                # in meas_units
        ukf_alpha=1e-3,
        ukf_beta=2.0,
        ukf_kappa=0.0,
        eps=1e-12,
    ):
        self.x = np.asarray(x0, dtype=float).reshape(6)
        self.P = np.asarray(P0, dtype=float).reshape(6, 6)

        self.Sa_rtn = np.asarray(Sa_rtn, dtype=float)  # (3,) or (3,3), km-units
        self.meas_units = meas_units
        self.sigma_ra = float(sigma_ra)
        self.sigma_dec = float(sigma_dec)
        self.sigma_pointing = float(sigma_pointing)

        self.alpha = float(ukf_alpha)
        self.beta = float(ukf_beta)
        self.kappa = float(ukf_kappa)

        self.eps = float(eps)


    # -----------------------------
    # Core builders: Q(dt), R(t)
    # -----------------------------

    def build_Q(self, dt, r_eme_km=None, v_eme_km_s=None):
        """
        Build discrete-time Q(dt) in EME/J2000 (km units) from RTN acceleration
        spectral density Sa_rtn.

        Q(dt) = [[dt^3/3 * Sa_eme, dt^2/2 * Sa_eme],
                 [dt^2/2 * Sa_eme, dt     * Sa_eme]]
        """
        dt = float(dt)
        if r_eme_km is None or v_eme_km_s is None:
            r = self.x[:3]
            v = self.x[3:]
        else:
            r = np.asarray(r_eme_km, dtype=float).reshape(3)
            v = np.asarray(v_eme_km_s, dtype=float).reshape(3)

        C = self._C_RTN2EME(r, v)  # (3,3)

        Sa_rtn = np.asarray(self.Sa_rtn, dtype=float)
        if Sa_rtn.shape == (3,):
            Sa_rtn_mat = np.diag(Sa_rtn)
        elif Sa_rtn.shape == (3, 3):
            Sa_rtn_mat = Sa_rtn
        else:
            raise ValueError("Sa_rtn must have shape (3,) or (3,3).")

        Sa_eme = C @ Sa_rtn_mat @ C.T  # (3,3)

        dt2 = dt * dt
        dt3 = dt2 * dt

        Q_rr = (dt3 / 3.0) * Sa_eme
        Q_rv = (dt2 / 2.0) * Sa_eme
        Q_vv = dt * Sa_eme

        Q = np.block([[Q_rr, Q_rv],
                      [Q_rv, Q_vv]])
        return Q

    def build_R(self, ra_rad, dec_rad, *, sigma_ra=None, sigma_dec=None, sigma_pointing=None, units=None):
        if sigma_ra is None:
            sigma_ra = self.sigma_ra
        if sigma_dec is None:
            sigma_dec = self.sigma_dec
        if sigma_pointing is None:
            sigma_pointing = self.sigma_pointing
        if units is None:
            units = self.meas_units

        s_ra, s_dec, s_pt = self._sigmas_to_rad(sigma_ra, sigma_dec, sigma_pointing, units)

        s_ra2 = s_ra**2 + s_pt**2
        s_de2 = s_dec**2 + s_pt**2

        R_ang = np.array([[s_ra2, 0.0],
                          [0.0,  s_de2]], dtype=float)

        c = np.cos(dec_rad)
        s = np.sin(dec_rad)
        cra = np.cos(ra_rad)
        sra = np.sin(ra_rad)

        J = np.array([
            [-c * sra,   -s * cra],
            [ c * cra,   -s * sra],
            [ 0.0,        c      ]
        ], dtype=float)

        R_hat = J @ R_ang @ J.T
        return R_hat

    # -----------------------------
    # Measurement model (unit LOS)
    # -----------------------------

    @staticmethod
    def h_los_unitvec(r_obj_km, r_obs_km, eps=1e-12):
        rho = np.asarray(r_obj_km, dtype=float) - np.asarray(r_obs_km, dtype=float)
        n = np.linalg.norm(rho)
        return rho / max(n, eps)

    @staticmethod
    def los_to_ra_dec(rho_hat, eps=1e-12):
        x, y, z = rho_hat
        ra = np.arctan2(y, x)
        dec = np.arctan2(z, max(np.sqrt(x*x + y*y), eps))
        return ra, dec

    # -----------------------------
    # UKF predict/update
    # -----------------------------

    def _sigma_points_augmented(self, x, P, Q, jitter=1e-12):
        """
        Augmented sigma points for additive process noise:
            x_{k+1} = f(x_k) + w,  w ~ N(0, Q)

        Constructs sigma points for augmented vector [x; w], mean [x; 0],
        covariance blockdiag(P, Q).

        Returns:
          Xa: (2na+1, na) augmented sigma points
          Wm, Wc: (2na+1,) weights
          n: state dim
          qn: noise dim
        """
        x = np.asarray(x, dtype=float).reshape(6)
        P = np.asarray(P, dtype=float).reshape(6, 6)
        Q = np.asarray(Q, dtype=float)

        n = x.size
        if Q.shape != (n, n):
            raise ValueError(f"Augmented predict expects Q to be (6,6). Got {Q.shape}.")

        qn = n
        na = n + qn

        xa = np.zeros(na, dtype=float)
        xa[:n] = x  # noise mean is 0

        Pa = np.zeros((na, na), dtype=float)
        Pa[:n, :n] = P
        Pa[n:, n:] = Q

        lam = self.alpha**2 * (na + self.kappa) - na
        c = na + lam

        Wm = np.full(2 * na + 1, 1.0 / (2.0 * c), dtype=float)
        Wc = np.full(2 * na + 1, 1.0 / (2.0 * c), dtype=float)
        Wm[0] = lam / c
        Wc[0] = lam / c + (1.0 - self.alpha**2 + self.beta)

        # Cholesky with jitter fallback
        Pa = self._symmetrize(Pa)
        try:
            S = np.linalg.cholesky(c * Pa)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky(c * (Pa + jitter * np.eye(na)))

        Xa = np.empty((2 * na + 1, na), dtype=float)
        Xa[0] = xa
        for i in range(na):
            Xa[1 + i]      = xa + S[:, i]
            Xa[1 + i + na] = xa - S[:, i]

        return Xa, Wm, Wc, n, qn


    def predict(self, t0, t1, propagate_sigma_points_fn, observer_ephem_fn=None):
        """
        Augmented UKF prediction.

        Parameters
        ----------
        t0 : float
            Initial epoch.
        t1 : float OR array-like of float
            Target epoch(s). If array-like, predictions are SEQUENTIAL:
              (x,P) at each step becomes prior for next step.
        propagate_sigma_points_fn : callable
            Should take (X_sigma, t0, t1) and return propagated sigma points:
              X_prop = propagate_sigma_points_fn(X_sigma, t0, t1)
            where X_sigma is (Ns, 6) and X_prop is (Ns, 6).

        Returns
        -------
        If t1 is scalar:
            (x_pred, P_pred)
        If t1 is array-like with K entries:
            (X_pred, P_pred) where
              X_pred has shape (K,6),
              P_pred has shape (K,6,6)

        Notes
        -----
        Uses additive-noise augmented UKF:
            X_aug sigma points -> propagate state part -> add noise part -> recombine.
        This replaces the non-augmented "P += Q(dt)" step.
        """
        t1_arr = np.asarray(t1, dtype=float).ravel()
        scalar_input = (t1_arr.size == 1)

        x_curr = np.asarray(self.x, dtype=float).reshape(6)
        P_curr = np.asarray(self.P, dtype=float).reshape(6, 6)

        X_out = []
        P_out = []

        t_prev = float(t0)

        for t_next in t1_arr:
            t_next = float(t_next)
            dt = float(t_next - t_prev)
            if dt < 0.0:
                raise ValueError(f"predict expects non-decreasing epochs: got dt={dt} from {t_prev} to {t_next}")

            # Process noise for this interval
            Q = self.build_Q(dt, r_eme_km=x_curr[:3], v_eme_km_s=x_curr[3:])

            # Augmented sigma points
            Xa, Wm, Wc, n, qn = self._sigma_points_augmented(x_curr, P_curr, Q)

            X_state = Xa[:, :n]   # (Ns,6)
            W_noise = Xa[:, n:]   # (Ns,6)

            # Propagate the state sigma points
            X_prop = propagate_sigma_points_fn(X_state, t_prev, t_next)  # (Ns,6)

            # Additive-noise injection
            X_prop_noisy = X_prop + W_noise

            # Mean and covariance
            x_pred = np.sum(Wm[:, None] * X_prop_noisy, axis=0)
            P_pred = np.zeros((6, 6), dtype=float)
            for i in range(X_prop_noisy.shape[0]):
                dx = (X_prop_noisy[i] - x_pred).reshape(6, 1)
                P_pred += Wc[i] * (dx @ dx.T)

            x_curr = x_pred
            P_curr = self._symmetrize(P_pred)
            t_prev = t_next

            X_out.append(x_curr.copy())
            P_out.append(P_curr.copy())

        X_out = np.stack(X_out, axis=0)
        P_out = np.stack(P_out, axis=0)

        if scalar_input:
            return X_out[0], P_out[0]
        return X_out, P_out


    def propagate_priors(self, t0, t_grid, propagate_many_fn):
        """
        Propagate priors (mean/cov) to a set of future epochs WITHOUT measurements.

        This is a "distribution push-forward" using a single sigma-point set at t0:
          1) Generate sigma points from (self.x, self.P) at t0
          2) Propagate ALL sigma points to ALL epochs in t_grid in one call
          3) Recombine mean/cov at each epoch
          4) Add process noise Q(dt) for dt = (t_k - t0) in seconds

        Parameters
        ----------
        t0 : float
            Initial epoch (JDTDB).
        t_grid : float OR array-like
            Target epoch(s) in JDTDB. Must be >= t0 and non-decreasing.
        propagate_many_fn : callable
            Must support:
                X_sig_t = propagate_many_fn(X_sigma, t0, t_grid)
            where:
                X_sigma: (Ns,6)
            and returns:
                - if t_grid is scalar: (Ns,6)
                - if t_grid is length K: (K, Ns, 6)   (time-major)

            This matches your NBodyPropagator.propagate_multiple_objects.

        Returns
        -------
        If t_grid is scalar:
            (x_pred, P_pred) with shapes (6,), (6,6)
        If t_grid is array-like with K entries:
            (X_pred, P_pred) with shapes (K,6), (K,6,6)

        Notes
        -----
        - Does NOT update self.x, self.P.
        - Uses dt_seconds = (t_k - t0) * 86400 for Q().
        """
        SEC_PER_DAY = 86400.0

        t_arr = np.asarray(t_grid, dtype=float).ravel()
        scalar_input = (t_arr.size == 1)

        if t_arr.size == 0:
            raise ValueError("t_grid must be non-empty")

        t0 = float(t0)
        if np.any(t_arr < t0 - 1e-15):
            raise ValueError("t_grid must be >= t0")
        if t_arr.size > 1 and np.any(np.diff(t_arr) < -1e-15):
            raise ValueError("t_grid must be non-decreasing")

        # Sigma points at t0 from current filter state (do NOT modify self.x/self.P)
        X0_sigma, Wm, Wc = self._sigma_points(self.x, self.P)  # (Ns,6), (Ns,), (Ns,)
        Ns = X0_sigma.shape[0]

        # Propagate sigma points to all requested epochs in one call
        Xsig_t = propagate_many_fn(X0_sigma, t0, t_arr)

        # Normalize return shape to (K, Ns, 6)
        if scalar_input:
            Xsig_t = np.asarray(Xsig_t, dtype=float)
            if Xsig_t.shape != (Ns, 6):
                raise ValueError(f"Expected propagated sigma points (Ns,6) for scalar t_grid, got {Xsig_t.shape}")
            Xsig_t = Xsig_t.reshape(1, Ns, 6)
        else:
            Xsig_t = np.asarray(Xsig_t, dtype=float)
            if Xsig_t.shape != (t_arr.size, Ns, 6):
                raise ValueError(
                    f"Expected propagated sigma points (K,Ns,6) with K={t_arr.size}, Ns={Ns}, got {Xsig_t.shape}"
                )

        K = Xsig_t.shape[0]
        X_pred = np.zeros((K, 6), dtype=float)
        P_pred = np.zeros((K, 6, 6), dtype=float)

        # Recombine at each epoch
        for k in range(K):
            Xk = Xsig_t[k]  # (Ns,6)

            # mean
            xk = np.sum(Wm[:, None] * Xk, axis=0)  # (6,)
            X_pred[k] = xk

            # covariance from transformed sigma points
            Pk = np.zeros((6, 6), dtype=float)
            for i in range(Ns):
                dx = (Xk[i] - xk).reshape(6, 1)
                Pk += Wc[i] * (dx @ dx.T)

            # add process noise for whole interval t0 -> t_k (dt in seconds!)
            dt_sec = float((t_arr[k] - t0) * SEC_PER_DAY)
            if dt_sec > 0.0:
                Pk += self.build_Q(dt_sec, r_eme_km=xk[:3], v_eme_km_s=xk[3:])

            P_pred[k] = self._symmetrize(Pk)

        if scalar_input:
            return X_pred[0], P_pred[0]
        return X_pred, P_pred


    def _sigma_points(self, x, P, jitter=1e-12):
        """
        Standard scaled unscented transform sigma points.

        Inputs:
          x: (n,)
          P: (n,n)

        Returns:
          X:  (2n+1, n)
          Wm: (2n+1,)
          Wc: (2n+1,)
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        P = np.asarray(P, dtype=float)
        n = int(x.size)
        if P.shape != (n, n):
            raise ValueError(f"P must be ({n},{n}), got {P.shape}")

        lam = self.alpha ** 2 * (n + self.kappa) - n
        c = n + lam
        if c <= 0.0:
            raise ValueError(f"Invalid UKF scaling: n+lambda={c} <= 0. Adjust alpha/kappa.")

        # weights
        Wm = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=float)
        Wc = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=float)
        Wm[0] = lam / c
        Wc[0] = lam / c + (1.0 - self.alpha ** 2 + self.beta)

        # sigma points
        P = self._symmetrize(P)
        try:
            S = np.linalg.cholesky(c * P)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky(c * (P + jitter * np.eye(n)))

        X = np.empty((2 * n + 1, n), dtype=float)
        X[0] = x
        for i in range(n):
            X[1 + i] = x + S[:, i]
            X[1 + i + n] = x - S[:, i]
        return X, Wm, Wc


    def update_angles_unitvec(self, z_rhohat, r_obs_km, R_hat):
        z = np.asarray(z_rhohat, dtype=float).reshape(3)
        r_obs = np.asarray(r_obs_km, dtype=float).reshape(3)
        R = np.asarray(R_hat, dtype=float).reshape(3, 3)

        X, Wm, Wc = self._sigma_points(self.x, self.P)

        Zsig = np.zeros((X.shape[0], 3), dtype=float)
        for i in range(X.shape[0]):
            Zsig[i] = self.h_los_unitvec(X[i, :3], r_obs, eps=self.eps)

        z_pred = np.sum(Wm[:, None] * Zsig, axis=0)

        S = np.zeros((3, 3), dtype=float)
        Pxz = np.zeros((6, 3), dtype=float)
        for i in range(Zsig.shape[0]):
            dz = (Zsig[i] - z_pred).reshape(3, 1)
            dx = (X[i] - self.x).reshape(6, 1)
            S += Wc[i] * (dz @ dz.T)
            Pxz += Wc[i] * (dx @ dz.T)

        S = S + R
        K = Pxz @ np.linalg.inv(S)

        innov = (z - z_pred).reshape(3, 1)
        self.x = self.x + (K @ innov).reshape(6)
        self.P = self._symmetrize(self.P - K @ S @ K.T)

    # -----------------------------
    # Helpers
    # ----------------------------
    def _C_RTN2EME(self, r, v):
        r = np.asarray(r, dtype=float).reshape(3)
        v = np.asarray(v, dtype=float).reshape(3)

        r_norm = np.linalg.norm(r)
        if r_norm < self.eps:
            raise ValueError("r norm too small to define RTN")

        Rhat = r / r_norm
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        if h_norm < self.eps:
            raise ValueError("r x v too small to define RTN")

        Nhat = h / h_norm
        That = np.cross(Nhat, Rhat)
        t_norm = np.linalg.norm(That)
        if t_norm < self.eps:
            raise ValueError("T axis degenerate")
        That = That / t_norm

        return np.column_stack((Rhat, That, Nhat))

    @staticmethod
    def _symmetrize(A):
        return 0.5 * (A + A.T)

    @staticmethod
    def _sigmas_to_rad(sigma_ra, sigma_dec, sigma_pointing, units):
        if units == "rad":
            return float(sigma_ra), float(sigma_dec), float(sigma_pointing)
        if units == "arcsec":
            return (float(sigma_ra) * ARCSEC_TO_RAD,
                    float(sigma_dec) * ARCSEC_TO_RAD,
                    float(sigma_pointing) * ARCSEC_TO_RAD)
        if units == "mas":
            return (float(sigma_ra) * MAS_TO_RAD,
                    float(sigma_dec) * MAS_TO_RAD,
                    float(sigma_pointing) * MAS_TO_RAD)
        raise ValueError("units must be 'rad', 'arcsec', or 'mas'")



