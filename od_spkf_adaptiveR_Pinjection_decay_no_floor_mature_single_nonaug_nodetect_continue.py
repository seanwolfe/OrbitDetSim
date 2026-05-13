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
        sc_secr_se_kms, earth_helio_ae_kms, layout="batch", obj_hint="(batch, 6)"
    )
    sc_eme_ae_kms = util.geo_eclip_to_geo_eme_generic(sc_geoeclip_ae_kms, layout="batch", hint="(batch, 6)")

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


def process_tracklet_until_update_with_prior_epoch(
    *,
    ukf,
    prior_epoch_jdtdb,
    n_body_propagator,
    epochs_k,
    noisy_meas_k,
    sc_states_k,
    detection_res_k,
):
    epochs_k = np.asarray(epochs_k, dtype=float).ravel()
    noisy_meas_k = np.asarray(noisy_meas_k, dtype=float)
    sc_states_k = np.asarray(sc_states_k, dtype=float)

    if noisy_meas_k.ndim != 3 or noisy_meas_k.shape[2] != 2:
        raise ValueError(f"noisy_meas_k must be (M,N,2), got {noisy_meas_k.shape}")
    if sc_states_k.ndim != 3 or sc_states_k.shape[2] != 6:
        raise ValueError(f"sc_states_k must be (M,N,6), got {sc_states_k.shape}")

    M, N, _ = noisy_meas_k.shape
    if sc_states_k.shape[:2] != (M, N):
        raise ValueError(
            f"sc_states_k first dims must match noisy_meas_k: expected {(M, N)}, got {sc_states_k.shape[:2]}"
        )
    if epochs_k.size != N:
        raise ValueError(f"epochs_k must have length N={N}, got {epochs_k.size}")

    detecting_ids = tuple(sorted(
        i for i, d in enumerate(detection_res_k)
        if bool(d.get("detected", False))
    ))

    t_prev = float(prior_epoch_jdtdb)
    update_history = []
    processed_epochs = []

    if len(detecting_ids) == 0:
        # Prediction-only tracklet: preserve the same timing as a normal
        # sequential measurement pass, but skip all measurement updates and
        # range/range-rate covariance injection. This lets the UKF mean and
        # covariance advance through the missed observation window via the
        # dynamics and Q(dt), so downstream attitude coordination sees a
        # current prior rather than stale x/P.
        for j in range(N):
            t_meas = float(epochs_k[j])
            x_prior = ukf.x.copy()
            P_prior = ukf.P.copy()

            ukf.predict(
                t_prev,
                t_meas,
                n_body_propagator.propagate_multiple_objects
            )

            processed_epochs.append(t_meas)
            update_history.append({
                "frame_index": int(j),
                "epoch_jdtdb": t_meas,
                "x_prior": x_prior,
                "P_prior": P_prior,
                "updates": [],
                "x_post": ukf.x.copy(),
                "P_post": ukf.P.copy(),
                "epoch_injection_info": {
                    "applied": False,
                    "reason": "no_detection_prediction_only",
                    "mode": "prediction_only_no_measurement_update",
                    "n_observers": 0,
                },
            })
            t_prev = t_meas

        return {
            "had_detection": False,
            "detecting_ids": tuple(),
            "n_detections": 0,
            "processed_epochs": np.asarray(processed_epochs, dtype=float),
            "posterior_x": ukf.x.copy(),
            "posterior_P": ukf.P.copy(),
            "update_history": update_history,
        }


    # Decide covariance-injection policy once per tracklet, not once per frame.
    #
    # Logic:
    #   - Multi-observer tracklets increment the maturity counter and use decayed injection.
    #   - Early single-observer tracklets still use full injection, because range/range-rate
    #     are weakly observable from angles-only data.
    #   - Mature single-observer tracklets, after enough previous multi-observer tracklets,
    #     do NOT reset the maturity counter. They use the configured policy:
    #         "none"    -> skip artificial range/range-rate injection
    #         "decayed" -> inject with the existing multi-observer decay factor
    #         "full"    -> inject full single-observer uncertainty, but keep maturity history
    multiobs_threshold = int(getattr(ukf, "multi_observer_maturity_threshold", 3))
    mature_single_policy = str(
        getattr(ukf, "mature_single_observer_injection", "decayed")
    ).lower().strip()
    if mature_single_policy not in ("none", "decayed", "full"):
        raise ValueError(
            "ukf.mature_single_observer_injection must be one of "
            "'none', 'decayed', or 'full'"
        )

    if len(detecting_ids) >= 2:
        ukf.multi_observer_update_count += 1
        if ukf.p_injection_decay_tau <= 0.0:
            ukf.decay = 0.0
        else:
            ukf.decay = float(
                np.exp(-ukf.multi_observer_update_count / ukf.p_injection_decay_tau)
            )
        mode = "multi_observer_decayed_injection"
        apply_range_rate_injection = True

    else:
        mature_multiobs_prior = ukf.multi_observer_update_count >= multiobs_threshold

        if mature_multiobs_prior:
            if mature_single_policy == "none":
                # Let Q, dynamics, and the single-observer measurement geometry evolve P naturally.
                # Keep decay consistent for diagnostics, but do not apply injection below.
                if ukf.p_injection_decay_tau <= 0.0:
                    ukf.decay = 0.0
                else:
                    ukf.decay = float(
                        np.exp(-ukf.multi_observer_update_count / ukf.p_injection_decay_tau)
                    )
                mode = "single_observer_mature_prior_no_injection"
                apply_range_rate_injection = False

            elif mature_single_policy == "decayed":
                if ukf.p_injection_decay_tau <= 0.0:
                    ukf.decay = 0.0
                else:
                    ukf.decay = float(
                        np.exp(-ukf.multi_observer_update_count / ukf.p_injection_decay_tau)
                    )
                mode = "single_observer_mature_prior_decayed_injection"
                apply_range_rate_injection = True

            else:  # mature_single_policy == "full"
                ukf.decay = 1.0
                mode = "single_observer_mature_prior_full_injection"
                apply_range_rate_injection = True

        else:
            # Early single-observer regime: keep the original conservative behaviour,
            # but do not wipe out a nonzero multi-observer count unless it is below threshold.
            ukf.decay = 1.0
            mode = "single_observer_full_injection"
            apply_range_rate_injection = True

    # Prefer a real observing spacecraft LOS for injection, not a mean observer.
    primary_sid = int(getattr(ukf, "primary_detecting_id", detecting_ids[0]))

    for j in range(N):
        t_meas = float(epochs_k[j])

        ukf.predict(
            t_prev,
            t_meas,
            n_body_propagator.propagate_multiple_objects
        )

        x_prior = ukf.x.copy()
        P_prior = ukf.P.copy()

        epoch_updates = []
        epoch_r_obs_by_sid = {}

        for sid in detecting_ids:
            ra_meas = noisy_meas_k[sid, j, 0]
            dec_meas = noisy_meas_k[sid, j, 1]

            if not (np.isfinite(ra_meas) and np.isfinite(dec_meas)):
                continue

            r_obs_km = np.asarray(sc_states_k[sid, j, :3], dtype=float)

            z_hat = ukf.ra_dec_to_los_unitvec(ra_meas, dec_meas)
            R_hat = ukf.build_R(ra_meas, dec_meas)

            update_info = ukf.update_angles_unitvec(z_hat, r_obs_km, R_hat)

            epoch_r_obs_by_sid[int(sid)] = r_obs_km.copy()

            epoch_updates.append({
                "sid": int(sid),
                "epoch_jdtdb": t_meas,
                "ra_rad": float(ra_meas),
                "dec_rad": float(dec_meas),
                "r_obs_km": r_obs_km.copy(),
                "update_info": update_info,
                "single_observer_injection_info": None,
            })

        # Inject once per epoch after all same-epoch measurement updates.
        injection_info = None
        if len(epoch_updates) > 0:
            if primary_sid in epoch_r_obs_by_sid:
                r_obs_for_injection = epoch_r_obs_by_sid[primary_sid]
                injection_sid = primary_sid
            else:
                # Fallback to the first actual observer that had a valid measurement.
                injection_sid = int(epoch_updates[0]["sid"])
                r_obs_for_injection = epoch_r_obs_by_sid[injection_sid]

            if apply_range_rate_injection:
                injection_info = ukf.inject_decaying_range_rate_uncertainty(
                    r_obs_km=r_obs_for_injection,
                    n_observers=len(detecting_ids),
                    mode=mode,
                )
            else:
                injection_info = {
                    "applied": False,
                    "reason": "mature_single_observer_no_injection",
                    "mode": mode,
                    "n_observers": int(len(detecting_ids)),
                    "decay": float(getattr(ukf, "decay", np.nan)),
                    "tau": float(getattr(ukf, "p_injection_decay_tau", np.nan)),
                    "multi_observer_update_count": int(getattr(ukf, "multi_observer_update_count", 0)),
                    "multi_observer_maturity_threshold": int(multiobs_threshold),
                    "mature_single_observer_injection": mature_single_policy,
                }
            injection_info["injection_sid"] = injection_sid

            for u in epoch_updates:
                u["single_observer_injection_info"] = injection_info

            processed_epochs.append(t_meas)
            update_history.append({
                "frame_index": int(j),
                "epoch_jdtdb": t_meas,
                "x_prior": x_prior.copy(),
                "P_prior": P_prior.copy(),
                "updates": epoch_updates,
                "x_post": ukf.x.copy(),
                "P_post": ukf.P.copy(),
                "epoch_injection_info": injection_info,
            })

        t_prev = t_meas

    return {
        "had_detection": True,
        "detecting_ids": detecting_ids,
        "n_detections": len(detecting_ids),
        "processed_epochs": np.asarray(processed_epochs, dtype=float),
        "posterior_x": ukf.x.copy(),
        "posterior_P": ukf.P.copy(),
        "update_history": update_history,
    }


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
        sigma_meas=None,                   # optional isotropic angular noise; None -> use RA/Dec/pointing
        adaptive_R_window=20,              # innovation window for adaptive measurement covariance
        adaptive_R_min_samples=5,          # minimum samples before using adaptive R
        adaptive_R_psd_floor=1e-18,        # numerical eigenvalue floor for adaptive R in unit-vector space
        sigma_rho_single_obs_km=None,      # range uncertainty enforced during single-observer updates [km]
        sigma_rhodot_single_obs_km_s=None, # range-rate uncertainty enforced during single-observer updates [km/s]
        p_injection_decay_tau=1.0,         # multi-observer exponential decay constant in update counts; no final floor
        multi_observer_maturity_threshold=3,  # after this many multi-observer tracklets, later single-observer arcs are treated as mature
        mature_single_observer_injection="decayed",  # one of: "none", "decayed", "full"
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
        self.sigma_meas = None if sigma_meas is None else float(sigma_meas)

        self.adaptive_R_window = int(adaptive_R_window)
        self.adaptive_R_min_samples = int(adaptive_R_min_samples)
        self.adaptive_R_psd_floor = float(adaptive_R_psd_floor)
        self.innov_hist = []
        self.last_R_adaptive = None
        self.last_R_eff = None

        self.sigma_rho_single_obs_km = (
            None if sigma_rho_single_obs_km is None
            else float(sigma_rho_single_obs_km)
        )
        self.sigma_rhodot_single_obs_km_s = (
            None if sigma_rhodot_single_obs_km_s is None
            else float(sigma_rhodot_single_obs_km_s)
        )
        self.p_injection_decay_tau = float(p_injection_decay_tau)
        self.multi_observer_update_count = 0
        self.multi_observer_maturity_threshold = int(multi_observer_maturity_threshold)
        self.mature_single_observer_injection = str(mature_single_observer_injection).lower().strip()
        if self.mature_single_observer_injection not in ("none", "decayed", "full"):
            raise ValueError(
                "mature_single_observer_injection must be one of "
                "'none', 'decayed', or 'full'"
            )
        self.decay=1.0

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

    def build_R(self, ra_rad, dec_rad, *, sigma_ra=None, sigma_dec=None, sigma_pointing=None, sigma_meas=None, units=None):
        if sigma_ra is None:
            sigma_ra = self.sigma_ra
        if sigma_dec is None:
            sigma_dec = self.sigma_dec
        if sigma_pointing is None:
            sigma_pointing = self.sigma_pointing
        if units is None:
            units = self.meas_units
        # If no per-call isotropic measurement uncertainty is supplied, use the
        # object's default. If that default is also None, use RA/Dec/pointing.
        if sigma_meas is None:
            sigma_meas = self.sigma_meas

        if sigma_meas is None:
            s_ra, s_dec, s_pt = self._sigmas_to_rad(sigma_ra, sigma_dec, sigma_pointing, units)

            s_ra2 = s_ra**2 + s_pt**2
            s_de2 = s_dec**2 + s_pt**2

            R_ang = np.array([[s_ra2, 0.0],
                              [0.0,  s_de2]], dtype=float)
        else:
            s_meas, _, _ = self._sigmas_to_rad(sigma_meas, sigma_meas, sigma_meas, units)  # re use for conveniece
            R_ang = np.array([[s_meas ** 2, 0.0],
                              [0.0, s_meas ** 2]], dtype=float)

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


    @staticmethod
    def ra_dec_to_los_unitvec(ra_rad, dec_rad):
        """
        Convert RA/Dec [rad] to Cartesian LOS unit vector.
        """
        ra = float(ra_rad)
        dec = float(dec_rad)

        cdec = np.cos(dec)
        return np.array([
            cdec * np.cos(ra),
            cdec * np.sin(ra),
            np.sin(dec)
        ], dtype=float)


    def predict(self, t0_jdtdb, t1_jdtdb, propagate_sigma_points_fn):
        """
        Additive-process-noise UKF prediction.

        Epochs are in JDTDB days, while build_Q expects dt in seconds.

        This version does NOT augment the sigma-point state with process-noise
        states. It:
          1) generates sigma points from the 6D state covariance P,
          2) propagates those 6D state sigma points,
          3) recombines the propagated sigma points into x_pred and P_pred,
          4) adds Q(dt) directly to P_pred.
        """
        SEC_PER_DAY = 86400.0

        t1_arr = np.asarray(t1_jdtdb, dtype=float).ravel()
        scalar_input = (t1_arr.size == 1)

        x_curr = np.asarray(self.x, dtype=float).reshape(6)
        P_curr = np.asarray(self.P, dtype=float).reshape(6, 6)

        X_out = []
        P_out = []

        t_prev = float(t0_jdtdb)

        for t_next in t1_arr:
            t_next = float(t_next)
            dt_days = float(t_next - t_prev)
            if dt_days < 0.0:
                raise ValueError(
                    f"predict expects non-decreasing epochs: "
                    f"got dt_days={dt_days} from {t_prev} to {t_next}"
                )

            dt_sec = dt_days * SEC_PER_DAY

            # Process noise for this interval, with dt in seconds.
            Q = self.build_Q(dt_sec, r_eme_km=x_curr[:3], v_eme_km_s=x_curr[3:])

            # State-only sigma points. No augmented process-noise states.
            # print(".predict (start)")
            X_state, Wm, Wc = self._sigma_points(x_curr, P_curr)

            # Propagate state sigma points from t_prev to t_next.
            X_prop = propagate_sigma_points_fn(X_state, t_prev, t_next)
            X_prop = np.asarray(X_prop, dtype=float)

            if X_prop.shape != X_state.shape:
                raise ValueError(
                    f"Expected propagated sigma points shape {X_state.shape}, "
                    f"got {X_prop.shape}"
                )

            # Mean and covariance recombination from propagated state sigma points.
            x_pred = np.sum(Wm[:, None] * X_prop, axis=0)
            P_pred = np.zeros((6, 6), dtype=float)
            for i in range(X_prop.shape[0]):
                dx = (X_prop[i] - x_pred).reshape(6, 1)
                P_pred += Wc[i] * (dx @ dx.T)

            # Add additive process noise after recombination.
            P_pred += Q

            x_curr = x_pred
            P_curr = self._symmetrize(P_pred)
            t_prev = t_next

            X_out.append(x_curr.copy())
            P_out.append(P_curr.copy())

        X_out = np.stack(X_out, axis=0)
        P_out = np.stack(P_out, axis=0)

        # commit to filter state
        self.x = X_out[-1].copy()
        self.P = P_out[-1].copy()

        # print(".predict (end)")
        # P = self.P
        # evals = np.linalg.eigvalsh(P)
        # print("Pa diag:", np.diag(P))
        # print("Pa eig min:", evals[0])
        # print("Pa eig max:", evals[-1])
        # print("Pa cond approx:", evals[-1] / max(evals[0], 1e-300))

        if scalar_input:
            return X_out[0], P_out[0]
        return X_out, P_out


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
        # print(np.diagonal(Q))

        lam = self.alpha**2 * (na + self.kappa) - na
        c = na + lam

        Wm = np.full(2 * na + 1, 1.0 / (2.0 * c), dtype=float)
        Wc = np.full(2 * na + 1, 1.0 / (2.0 * c), dtype=float)
        Wm[0] = lam / c
        Wc[0] = lam / c + (1.0 - self.alpha**2 + self.beta)

        # Cholesky with jitter fallback
        Pa = self._symmetrize(Pa)

        # evals = np.linalg.eigvalsh(Pa)
        # print("Pa diag:", np.diag(Pa))
        # print("Pa eig min:", evals[0])
        # print("Pa eig max:", evals[-1])
        # print("Pa cond approx:", evals[-1] / max(evals[0], 1e-300))

        # print(np.diagonal(Pa))
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

    def propagate_priors(self, t0, t_grid, propagate_many_fn):
        """
        Propagate priors (mean/cov) to a set of future epochs WITHOUT measurements.

        Always returns time-major arrays, even if only one epoch is provided:

            X_pred shape: (K, 6)
            P_pred shape: (K, 6, 6)

        where K = number of epochs in t_grid.

        This avoids caller-side shape ambiguity such as:
            x_ts[:, :3]
        failing when K == 1.
        """
        SEC_PER_DAY = 86400.0

        # Treat input uniformly as a 1D array of epochs.
        t_arr = np.asarray(t_grid, dtype=float).reshape(-1)

        if t_arr.size == 0:
            raise ValueError("t_grid must be non-empty")

        t0 = float(t0)

        if np.any(t_arr < t0 - 1e-15):
            raise ValueError("t_grid must be >= t0")

        if t_arr.size > 1 and np.any(np.diff(t_arr) < -1e-15):
            raise ValueError("t_grid must be non-decreasing")

        # Sigma points at t0 from current filter state.
        # This does NOT modify self.x or self.P.
        X0_sigma, Wm, Wc = self._sigma_points(self.x, self.P)
        Ns = X0_sigma.shape[0]
        K = t_arr.size

        # Propagate sigma points to all requested epochs.
        Xsig_t = propagate_many_fn(X0_sigma, t0, t_arr)
        Xsig_t = np.asarray(Xsig_t, dtype=float)

        # Normalize propagated sigma-point shape to (K, Ns, 6).
        #
        # Some propagators may return:
        #   - (Ns, 6) when K == 1
        #   - (1, Ns, 6) when K == 1
        #   - (K, Ns, 6) when K > 1
        #
        # Accept both valid K == 1 forms.
        if K == 1:
            if Xsig_t.shape == (Ns, 6):
                Xsig_t = Xsig_t.reshape(1, Ns, 6)
            elif Xsig_t.shape == (1, Ns, 6):
                pass
            else:
                raise ValueError(
                    f"Expected propagated sigma points shape (Ns,6) or (1,Ns,6) "
                    f"for one target epoch, got {Xsig_t.shape}; Ns={Ns}"
                )
        else:
            if Xsig_t.shape != (K, Ns, 6):
                raise ValueError(
                    f"Expected propagated sigma points shape (K,Ns,6) with "
                    f"K={K}, Ns={Ns}, got {Xsig_t.shape}"
                )

        X_pred = np.zeros((K, 6), dtype=float)
        P_pred = np.zeros((K, 6, 6), dtype=float)

        for k in range(K):
            Xk = Xsig_t[k]  # (Ns, 6)

            # Mean
            xk = np.sum(Wm[:, None] * Xk, axis=0)
            X_pred[k] = xk

            # Covariance from transformed sigma points
            Pk = np.zeros((6, 6), dtype=float)
            for i in range(Ns):
                dx = (Xk[i] - xk).reshape(6, 1)
                Pk += Wc[i] * (dx @ dx.T)

            # Add process noise for interval t0 -> t_k.
            dt_sec = float((t_arr[k] - t0) * SEC_PER_DAY)
            if dt_sec > 0.0:
                Pk += self.build_Q(
                    dt_sec,
                    r_eme_km=xk[:3],
                    v_eme_km_s=xk[3:]
                )

            P_pred[k] = self._symmetrize(Pk)

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
        # evals = np.linalg.eigvalsh(P)
        # print("Pa diag:", np.diag(P))
        # print("Pa eig min:", evals[0])
        # print("Pa eig max:", evals[-1])
        # print("Pa cond approx:", evals[-1] / max(evals[0], 1e-300))

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


    def update_angles_unitvec(self, z_rhohat, r_obs_km, R_hat, *, use_adaptive_R=True):
        """
        UKF unit-vector angles-only update with optional adaptive R.

        The adaptive measurement covariance is the UKF analogue of

            R_k = Cov(innovation) - S_model

        where S_model is the predicted measurement covariance from sigma-point
        spread before adding measurement noise. This adaptive R is used
        regardless of observer count; single-observer observability protection
        is handled separately by inject_single_observer_range_rate_uncertainty().
        """
        z = np.asarray(z_rhohat, dtype=float).reshape(3)
        r_obs = np.asarray(r_obs_km, dtype=float).reshape(3)
        R_nominal = np.asarray(R_hat, dtype=float).reshape(3, 3)
        # print(np.diag(self.P))
        # print(".update_angles_unitvec (start)")
        X, Wm, Wc = self._sigma_points(self.x, self.P)

        Zsig = np.zeros((X.shape[0], 3), dtype=float)
        for i in range(X.shape[0]):
            Zsig[i] = self.h_los_unitvec(X[i, :3], r_obs, eps=self.eps)

        z_pred = np.sum(Wm[:, None] * Zsig, axis=0)

        S_model = np.zeros((3, 3), dtype=float)
        Pxz = np.zeros((6, 3), dtype=float)
        for i in range(Zsig.shape[0]):
            dz = (Zsig[i] - z_pred).reshape(3, 1)
            dx = (X[i] - self.x).reshape(6, 1)
            S_model += Wc[i] * (dz @ dz.T)
            Pxz += Wc[i] * (dx @ dz.T)
        S_model = self._symmetrize(S_model)

        innov = (z - z_pred).reshape(3)

        if use_adaptive_R:
            R_eff, adaptive_info = self.estimate_adaptive_R(
                innov=innov,
                S_model=S_model,
                R_fallback=R_nominal,
            )
        else:
            R_eff = R_nominal.copy()

            adaptive_info = {
                "used_adaptive_R": False,
                "reason": "disabled",
                "n_innov": len(self.innov_hist),
            }

        S = self._symmetrize(S_model + R_eff)
        K = Pxz @ np.linalg.pinv(S)

        self.x = self.x + (K @ innov.reshape(3, 1)).reshape(6)
        self.P = self._symmetrize(self.P - K @ S @ K.T)
        self.P = self._project_psd(self.P, eig_floor=1e-10)

        # print(".update_angles_unitvec (end)")
        # P = self.P
        # evals = np.linalg.eigvalsh(P)
        # print("Pa diag:", np.diag(P))
        # print("Pa eig min:", evals[0])
        # print("Pa eig max:", evals[-1])
        # print("Pa cond approx:", evals[-1] / max(evals[0], 1e-300))

        self.last_R_eff = R_eff.copy()

        return {
            "z_pred": z_pred.copy(),
            "innovation": innov.copy(),
            "S_model": S_model.copy(),
            "R_eff": R_eff.copy(),
            "S": S.copy(),
            "adaptive_R_info": adaptive_info,
        }

    def estimate_adaptive_R(self, innov, S_model, R_fallback):
        """
        Estimate adaptive measurement covariance for the UKF measurement update.

        R_adapt = sample_cov(innovations) - S_model

        Only a numerical PSD floor is enforced. No physical sensor floor is
        imposed here, because the single-observer ambiguity is handled by state
        covariance injection, not by inflating or flooring R.
        """
        innov = np.asarray(innov, dtype=float).reshape(3)
        S_model = np.asarray(S_model, dtype=float).reshape(3, 3)
        R_fallback = np.asarray(R_fallback, dtype=float).reshape(3, 3)

        self.innov_hist.append(innov.copy())
        if len(self.innov_hist) > self.adaptive_R_window:
            self.innov_hist.pop(0)

        n_innov = len(self.innov_hist)
        if n_innov < max(2, self.adaptive_R_min_samples):
            self.last_R_adaptive = R_fallback.copy()
            return R_fallback.copy(), {
                "used_adaptive_R": False,
                "reason": "insufficient_innovation_samples",
                "n_innov": n_innov,
            }

        V = np.asarray(self.innov_hist, dtype=float)  # (N,3)
        Sigma_innov = np.cov(V.T, bias=True)
        Sigma_innov = self._symmetrize(Sigma_innov)

        R_adapt_raw = self._symmetrize(Sigma_innov - S_model)
        R_adapt = self._project_psd(R_adapt_raw, eig_floor=self.adaptive_R_psd_floor)
        Delta = self._project_psd(R_adapt - R_fallback, eig_floor=0.0)
        R_adapt = self._symmetrize(R_fallback + Delta)

        self.last_R_adaptive = R_adapt.copy()
        return R_adapt, {
            "used_adaptive_R": True,
            "n_innov": n_innov,
            "Sigma_innov": Sigma_innov.copy(),
            "R_adapt_raw": R_adapt_raw.copy(),
            "R_adapt": R_adapt.copy(),
        }

    def inject_decaying_range_rate_uncertainty(
        self,
        r_obs_km,
        *,
        n_observers,
        sigma_rho_km=None,
        sigma_rhodot_km_s=None,
        tau=None,
        mode=None,
    ):
        """
        Enforce LOS range/range-rate covariance with observer-count-dependent decay.

        Behavior:
          The caller sets self.decay and passes a mode string. This helper only
          applies the configured range/range-rate uncertainty scaled by self.decay:

              sigma_eff = sigma_single * self.decay

          The process_tracklet_until_update_with_prior_epoch() wrapper decides
          whether the current tracklet is early single-observer, multi-observer,
          or mature-prior single-observer. No terminal minimum floor is used,
          so the enforced uncertainty can approach zero after enough geometry.

        This operates on P, not R. Adaptive R remains active regardless of
        observer count.
        """
        n_observers = int(n_observers)

        if sigma_rho_km is None:
            sigma_rho_km = self.sigma_rho_single_obs_km
        if sigma_rhodot_km_s is None:
            sigma_rhodot_km_s = self.sigma_rhodot_single_obs_km_s
        if tau is None:
            tau = self.p_injection_decay_tau
        tau = float(tau)

        if sigma_rho_km is None and sigma_rhodot_km_s is None:
            return {
                "applied": False,
                "reason": "range_rate_uncertainty_disabled",
                "n_observers": n_observers,
            }

        sigma_rho_eff = None if sigma_rho_km is None else float(sigma_rho_km) * self.decay
        sigma_rhodot_eff = None if sigma_rhodot_km_s is None else float(sigma_rhodot_km_s) * self.decay

        info = self.inject_range_rate_uncertainty(
            r_obs_km=r_obs_km,
            sigma_rho_km=sigma_rho_eff,
            sigma_rhodot_km_s=sigma_rhodot_eff,
        )
        info.update({
            "mode": mode,
            "n_observers": n_observers,
            "decay": float(self.decay),
            "tau": float(tau),
            "multi_observer_update_count": int(self.multi_observer_update_count),
            "sigma_rho_base_km": None if sigma_rho_km is None else float(sigma_rho_km),
            "sigma_rhodot_base_km_s": None if sigma_rhodot_km_s is None else float(sigma_rhodot_km_s),
            "sigma_rho_eff_km": sigma_rho_eff,
            "sigma_rhodot_eff_km_s": sigma_rhodot_eff,
        })
        return info


    def inject_range_rate_uncertainty(
        self,
        r_obs_km,
        *,
        sigma_rho_km=None,
        sigma_rhodot_km_s=None,
    ):
        """
        Enforce covariance along the instantaneous LOS range and range-rate
        directions. This helper has no observer-count logic and no terminal
        floor beyond the sigma values supplied by the caller.
        """
        if sigma_rho_km is None and sigma_rhodot_km_s is None:
            return {
                "applied": False,
                "reason": "no_sigma_values_supplied",
            }

        r_obs = np.asarray(r_obs_km, dtype=float).reshape(3)
        rho_vec = self.x[:3] - r_obs
        rho_norm = np.linalg.norm(rho_vec)
        if rho_norm < self.eps:
            return {
                "applied": False,
                "reason": "rho_norm_too_small",
            }

        rho_hat = rho_vec / rho_norm
        uuT = np.outer(rho_hat, rho_hat)

        info = {
            "applied": True,
            "rho_hat": rho_hat.copy(),
            "rho_norm_km": float(rho_norm),
            "position_injected": False,
            "velocity_injected": False,
            "sigma_rho_km": None if sigma_rho_km is None else float(sigma_rho_km),
            "sigma_rhodot_km_s": None if sigma_rhodot_km_s is None else float(sigma_rhodot_km_s),
            "var_rho_before": None,
            "var_rho_after": None,
            "var_rhodot_before": None,
            "var_rhodot_after": None,
        }

        if sigma_rho_km is not None:
            var_target = float(sigma_rho_km) ** 2
            var_before = float(rho_hat @ self.P[:3, :3] @ rho_hat)
            info["var_rho_before"] = var_before
            if var_before < var_target:
                self.P[:3, :3] += (var_target - var_before) * uuT
                info["position_injected"] = True
            info["var_rho_after"] = float(rho_hat @ self.P[:3, :3] @ rho_hat)

        if sigma_rhodot_km_s is not None:
            var_target = float(sigma_rhodot_km_s) ** 2
            var_before = float(rho_hat @ self.P[3:, 3:] @ rho_hat)
            info["var_rhodot_before"] = var_before
            if var_before < var_target:
                self.P[3:, 3:] += (var_target - var_before) * uuT
                info["velocity_injected"] = True
            info["var_rhodot_after"] = float(rho_hat @ self.P[3:, 3:] @ rho_hat)
        self.P = self._symmetrize(self.P)
        return info


    @staticmethod
    def _project_psd(A, eig_floor=0.0):
        A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
        eigval, eigvec = np.linalg.eigh(A)
        eigval = np.maximum(eigval, float(eig_floor))
        A_psd = eigvec @ np.diag(eigval) @ eigvec.T
        return 0.5 * (A_psd + A_psd.T)

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



