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

    This is a skeleton: predict/update logic is minimal; you plug in your propagator
    and (optionally) a full UKF implementation.
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

        self.dimL = len(self.x)
        self.lambd = ukf_alpha ** 2 * (self.dimL + ukf_kappa) - self.dimL
        self.w_0_m = self.lambd / (self.lambd + self.dimL)  # first weight for computing the mean
        self.w_j_m = 0.5 / (self.lambd + self.dimL)  # consequent weights for computing the mean
        self.w_0_c = self.w_0_m + (1 - ukf_alpha ** 2 + ukf_beta)  # first weight for computing covariance
        self.w_j_c = self.w_j_m


    # -----------------------------
    # Core builders: Q(dt), R(t)
    # -----------------------------

    def build_Q(self, dt, r_eme_km=None, v_eme_km_s=None):
        """
        Build discrete-time Q(dt) in EME/J2000 (km units) from RTN acceleration
        spectral density Sa_rtn.

        Q(dt) = [[dt^3/3 * Sa_eme, dt^2/2 * Sa_eme],
                 [dt^2/2 * Sa_eme, dt     * Sa_eme]]

        If r_eme_km/v_eme_km_s not provided, uses current self.x.
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
        """
        Build epoch-dependent measurement covariance R(t) for unit LOS vector z=rho_hat (3x1),
        from (sigma_ra, sigma_dec, sigma_pointing) in angle space.

        R_hat = J(ra,dec) * diag(sigma_ra^2 + sigma_p^2, sigma_dec^2 + sigma_p^2) * J^T
        where rho_hat(ra,dec) = [cos d cos a, cos d sin a, sin d]^T.

        Returns a (3,3) covariance.
        """
        if sigma_ra is None:
            sigma_ra = self.sigma_ra
        if sigma_dec is None:
            sigma_dec = self.sigma_dec
        if sigma_pointing is None:
            sigma_pointing = self.sigma_pointing
        if units is None:
            units = self.meas_units

        s_ra, s_dec, s_pt = self._sigmas_to_rad(sigma_ra, sigma_dec, sigma_pointing, units)

        # Add pointing in quadrature (common assumption; independent, isotropic)
        s_ra2 = s_ra**2 + s_pt**2
        s_de2 = s_dec**2 + s_pt**2

        R_ang = np.array([[s_ra2, 0.0],
                          [0.0,  s_de2]], dtype=float)

        c = np.cos(dec_rad)
        s = np.sin(dec_rad)
        cra = np.cos(ra_rad)
        sra = np.sin(ra_rad)

        # J = d(rho_hat)/d[ra,dec]
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
        """
        z = rho_hat = (r_obj - r_obs) / ||r_obj - r_obs||  in EME/J2000.
        """
        rho = np.asarray(r_obj_km, dtype=float) - np.asarray(r_obs_km, dtype=float)
        n = np.linalg.norm(rho)
        return rho / max(n, eps)

    @staticmethod
    def los_to_ra_dec(rho_hat, eps=1e-12):
        """
        Convert unit LOS vector to (RA, Dec) in radians.
        Dec uses atan2 form for numerical robustness.
        """
        x, y, z = rho_hat
        ra = np.arctan2(y, x)
        dec = np.arctan2(z, max(np.sqrt(x*x + y*y), eps))
        return ra, dec

    # -----------------------------
    # UKF scaffolding (minimal)
    # -----------------------------

    def predict(self, t0, t1, propagate_sigma_points_fn, observer_ephem_fn=None):
        """
        Prediction step skeleton.

        Parameters
        ----------
        t0, t1 : float
            Epochs (seconds or whatever you use consistently).
        propagate_sigma_points_fn : callable
            Should take (X_sigma, t0, t1) and return propagated sigma points.
            Signature suggestion:
                X_prop = propagate_sigma_points_fn(X_sigma, t0, t1)
            where X_sigma is (2n+1, 6).
        observer_ephem_fn : optional
            Not used in predict; kept for symmetry.

        Notes
        -----
        This skeleton uses a non-augmented predict: propagate sigma points then add Q(dt).
        If you want augmented UKF, you’ll generate sigma points in augmented space and
        inject noise inside your propagator.
        """
        dt = float(t1 - t0)
        X, Wm, Wc = self._sigma_points(self.x, self.P)
        X_prop = propagate_sigma_points_fn(X, t0, t1)

        x_pred = np.sum(Wm[:, None] * X_prop, axis=0)
        P_pred = np.zeros((6, 6), dtype=float)
        for i in range(X_prop.shape[0]):
            dx = (X_prop[i] - x_pred).reshape(6, 1)
            P_pred += Wc[i] * (dx @ dx.T)

        # Add process noise for this dt (in EME)
        P_pred += self.build_Q(dt, r_eme_km=x_pred[:3], v_eme_km_s=x_pred[3:])

        self.x, self.P = x_pred, self._symmetrize(P_pred)

    def update_angles_unitvec(self, z_rhohat, r_obs_km, R_hat):
        """
        Measurement update skeleton for unit LOS measurement z = rho_hat.

        Parameters
        ----------
        z_rhohat : (3,)
            Measured unit LOS vector in EME/J2000.
        r_obs_km : (3,)
            Observer position at measurement time (EME/J2000).
        R_hat : (3,3)
            Measurement covariance in unit-vector space (from build_R).

        Notes
        -----
        Uses a standard UKF measurement update (non-iterated).
        """
        z = np.asarray(z_rhohat, dtype=float).reshape(3)
        r_obs = np.asarray(r_obs_km, dtype=float).reshape(3)
        R = np.asarray(R_hat, dtype=float).reshape(3, 3)

        X, Wm, Wc = self._sigma_points(self.x, self.P)

        # Predicted measurements for each sigma point
        Zsig = np.zeros((X.shape[0], 3), dtype=float)
        for i in range(X.shape[0]):
            Zsig[i] = self.h_los_unitvec(X[i, :3], r_obs, eps=self.eps)

        z_pred = np.sum(Wm[:, None] * Zsig, axis=0)

        S = np.zeros((3, 3), dtype=float)   # P_zz
        Pxz = np.zeros((6, 3), dtype=float) # P_xz
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
    # -----------------------------

    def _sigma_points(self, x, P):
        """
        Standard scaled unscented transform sigma points for dimension n=6.
        Returns:
          X : (2n+1, n)
          Wm, Wc : (2n+1,)
        """
        x = np.asarray(x, dtype=float).reshape(6)
        P = np.asarray(P, dtype=float).reshape(6, 6)
        n = x.size

        lam = self.alpha**2 * (n + self.kappa) - n
        c = n + lam

        # weights
        Wm = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=float)
        Wc = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=float)
        Wm[0] = lam / c
        Wc[0] = lam / c + (1.0 - self.alpha**2 + self.beta)

        # sigma points
        S = np.linalg.cholesky(self._symmetrize(c * P))
        X = np.empty((2 * n + 1, n), dtype=float)
        X[0] = x
        for i in range(n):
            X[1 + i]     = x + S[:, i]
            X[1 + i + n] = x - S[:, i]
        return X, Wm, Wc

    def _C_RTN2EME(self, r, v):
        """
        Build RTN basis vectors expressed in EME, return C_RTN->EME with columns [R, T, N].
        """
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


