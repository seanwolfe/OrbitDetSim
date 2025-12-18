#!/usr/bin/env python3
import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import j_cost_exclusion_slew_angularcov as j_base
import numpy as np
import yaml
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

# ---------------------------------------------------------------------
# YOU MUST import your own optimizer + slew model from your project
# ---------------------------------------------------------------------
# from your_module import optimize_pointing_lbfgs_joint, theta_s_of_dt

# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def wrap_to_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


# ---------------------------------------------------------------------
# Coverage: count how many agents cover a point (3D geometry, z=0)
# Covered if angle(u_i, direction_to_point) <= theta_h
# ---------------------------------------------------------------------
def coverage_count_point(p_point: np.ndarray,
                         p_agents: np.ndarray,
                         u_boresights: np.ndarray,
                         theta_h: float) -> int:
    M = p_agents.shape[0]
    cnt = 0
    for i in range(M):
        d = j_base.unit(p_point - p_agents[i])
        if d is None:
            continue
        if j_base.angle_between(u_boresights[i], d) <= theta_h + 1e-12:
            cnt += 1
    return int(cnt)



# ---------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------
def run_one_trial(cfg: Dict[str, Any], trial_seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(trial_seed)

    # ---------------- geometry ----------------
    geom = cfg["geometry"]
    M = int(geom["M"])

    x_agents = rng.uniform(float(geom["x_agents_min"]), float(geom["x_agents_max"]), size=M)
    y_agents = rng.uniform(float(geom["y_agents_min"]), float(geom["y_agents_max"]), size=M)
    p_agents = np.stack([x_agents, y_agents, np.zeros(M)], axis=1)

    p_hat_2d = np.array([
        rng.uniform(float(geom["x_target_min"]), float(geom["x_target_max"])),
        rng.uniform(float(geom["y_target_min"]), float(geom["y_target_max"]))
    ], dtype=float)

    # ---------------- fov ----------------
    fov = cfg["fov"]
    theta_h = np.deg2rad(rng.uniform(float(fov["theta_h_min_deg"]), float(fov["theta_h_max_deg"])))

    # ---------------- slew model params ----------------
    slew = cfg.get("slew_model", {})
    # If you want these to be randomizable too, put min/max in YAML.
    tau_max = float(slew.get("tau_max", 0.004))
    h_max   = float(slew.get("h_max",   0.015))
    m_m     = float(slew.get("m_m",     50.0))
    l_m     = float(slew.get("l_m",     0.5))
    m_t     = float(slew.get("m_t",     5.0))
    d_t     = float(slew.get("d_t",     0.28))
    z_0     = float(slew.get("z_0",     0.3))

    I_max = (1.0 / 6.0) * m_m * (l_m / 2.0) ** 2 + (1.0 / 2.0) * m_t * (d_t / 2.0) ** 2 + m_t * z_0 ** 2
    alpha_max = 1.63 * tau_max / I_max
    omega_max = 1.63 * h_max / I_max

    # ---------------- motion: draw v and a from uniform bounds ----------------
    motion = cfg.get("motion", {})
    vmin = np.array(motion.get("v_min", [-0.02, -0.02]), dtype=float)
    vmax = np.array(motion.get("v_max", [ 0.02,  0.02]), dtype=float)
    amin = np.array(motion.get("a_min", [-5e-4, -5e-4]), dtype=float)
    amax = np.array(motion.get("a_max", [ 5e-4,  5e-4]), dtype=float)

    v_target = rng.uniform(vmin, vmax)
    a_target = rng.uniform(amin, amax)

    # ---------------- covariance in spherical space ----------------
    cov = cfg["covariance"]
    d_mahal = float(cov["d_mahal"])

    # These need to be in YAML for your spherical model
    # (degrees in YAML are convenient)
    sigma_ra0  = np.deg2rad(float(cov.get("sigma_ra0_deg",  2.0)))
    sigma_dec0 = np.deg2rad(float(cov.get("sigma_dec0_deg", 2.0)))
    sigma_rho0 = float(cov.get("sigma_rho0", 1.0))

    growth_rate = float(cov.get("growth_rate", 0.0))
    i_ref = int(cov.get("i_ref", 0))

    P_adr_0 = np.diag([sigma_ra0**2, sigma_dec0**2, sigma_rho0**2])

    # ---------------- EMS ----------------
    ems = cfg.get("ems", {})
    p_em = np.array(ems.get("p_em", [0.0, 0.0, 0.0]), dtype=float)
    R_em = float(ems.get("R_em", 0.5))
    alpha_s = np.deg2rad(float(ems.get("alpha_s_deg", 4.5)))
    lambda_em = float(ems.get("lambda_em", 1.0))
    beta_zeta = float(ems.get("beta_zeta", 1500.0))


    # ---------------- initial boresights: axis aligned toward mean (your logic) ----------------
    axis_dirs = np.array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, -1.0],
                          [-1.0, 0.0]])
    axis_angles = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])

    p_agents_2d = p_agents[:, :2]
    pointing_angles = np.zeros(M)
    for i in range(M):
        rel = p_hat_2d - p_agents_2d[i]
        n = np.linalg.norm(rel)
        if n < 1e-12:
            pointing_angles[i] = 0.0
        else:
            rel_unit = rel / n
            dots = axis_dirs @ rel_unit
            pointing_angles[i] = axis_angles[np.argmax(dots)]

    u_curr_agents = np.stack([np.sin(pointing_angles),
                              np.cos(pointing_angles),
                              np.zeros(M)], axis=1)


    # ---------------- epoch grid ----------------
    epochs = cfg.get("epochs", {})
    dt_min = float(epochs.get("dt_min", 10.0))
    dt_max = float(epochs.get("dt_max", 600.0))
    n_dt = int(epochs.get("n_dt", 60))
    delta_ts_mean = np.linspace(dt_min, dt_max, n_dt)

    # ---------------- sample a "true" point from the initial covariance at first epoch ----------------
    # Build P_p at dt0 (first epoch)
    dt0 = float(delta_ts_mean[0])
    p_hat_2d_t0 = p_hat_2d + v_target * dt0 + 0.5 * a_target * dt0 ** 2
    p_hat_t0 = np.array([p_hat_2d_t0[0], p_hat_2d_t0[1], 0.0], dtype=float)

    scale0 = np.exp(growth_rate * dt0)
    P_adr_t0 = (scale0 ** 2) * P_adr_0
    alpha0, delta0, rho0 = j_base.topocentric_alpha_delta_rho(p_hat_t0, p_agents[i_ref])
    P_p_t0 = j_base.cov_radec_rho_to_xyz(alpha0, delta0, rho0, P_adr_t0)
    P_p2_t0 = P_p_t0[:2, :2]

    sample_pt = j_base.sample_from_uncertainty_2d(p_hat_2d_t0, P_p2_t0, d_mahal=d_mahal, rng=rng)

    # ---------------- Mean method: earliest epoch all can point to mean ----------------
    dtm, info = j_base.earliest_epoch_all_can_point_to_mean(
        delta_ts_mean,
        p_hat_2d, v_target, a_target,
        p_agents, u_curr_agents,
        theta_h, alpha_max, omega_max,
        p_em, R_em, alpha_s
    )

    if dtm is not None:
        delta_ts = np.linspace(dt_min, dtm + 50, int((dtm + 50) / 10))
    else:
        delta_ts = delta_ts_mean


    # ---------------- Proposed method: choose epoch with minimum cost ----------------
    opt = cfg["optimizer"]
    kappa_sigma = float(opt["kappa_sigma"])
    n_mc = int(opt["n_mc"])
    n_restarts = int(opt.get("n_restarts", 1))

    best = {
        "cost": np.inf,
        "dt": None,
        "u_star": None,
        "theta_req_sum_deg": None,
        "coverage": None
    }

    for dt in delta_ts:
        dt = float(dt)

        # mean at epoch
        p_hat_2d_t = p_hat_2d + v_target * dt + 0.5 * a_target * dt**2
        p_hat_t = np.array([p_hat_2d_t[0], p_hat_2d_t[1], 0.0], dtype=float)

        # grow cov in (a,d,r) and map to xyz
        scale = np.exp(growth_rate * dt)
        P_adr_t = (scale**2) * P_adr_0
        alpha_t, delta_t, rho_t = j_base.topocentric_alpha_delta_rho(p_hat_t, p_agents[i_ref])
        P_p_t = j_base.cov_radec_rho_to_xyz(alpha_t, delta_t, rho_t, P_adr_t)

        # slew limit per epoch
        theta_s_t = float(j_base.theta_s_of_dt(dt, alpha_max, omega_max))
        theta_s_t = float(np.clip(theta_s_t, 0.0, np.deg2rad(179.0)))
        theta_s_list_t = np.full(M, theta_s_t)

        # propagate sample point to this epoch (consistent with your approach)
        if dt == dt0:
            sample_pt_t = sample_pt.copy()
        else:
            sample_pt_t = sample_pt + v_target * dt + 0.5 * a_target * dt**2
        sample_pt_3d = np.array([sample_pt_t[0], sample_pt_t[1], 0.0], dtype=float)

        # run optimizer
        u_star, ang_star, J_star, history, cost_star = j_base.optimize_pointing_lbfgs_joint(
            p_hat_t, P_p_t, p_agents, u_curr_agents,
            theta_h, theta_s_list_t,
            d_M=d_mahal, kappa_sigma=kappa_sigma,
            n_mc=n_mc, seed=trial_seed, n_restarts=n_restarts,
            p_em=p_em, R_em=R_em, alpha_s=alpha_s,
            lambda_em=lambda_em, beta_zeta=beta_zeta
        )

        if u_star is None:
            continue

        # pick minimum cost
        c = float(cost_star)
        if c < best["cost"]:
            # sum slews relative to initial u_curr
            slew_sum = 0.0
            for i in range(M):
                slew_sum += j_base.angle_between(u_curr_agents[i], u_star[i])
            slew_sum_deg = float(np.rad2deg(slew_sum))

            cov_cnt = coverage_count_point(sample_pt_3d, p_agents, u_star, theta_h)

            best.update({
                "cost": c,
                "dt": dt,
                "u_star": u_star,
                "theta_req_sum_deg": slew_sum_deg,
                "coverage": cov_cnt
            })

    # Proposed coverage one-hot
    prop0 = prop1 = prop2 = prop3p = 0
    prop_time = np.nan
    prop_slew_sum = np.nan
    if best["dt"] is not None:
        prop_time = float(best["dt"])
        prop_slew_sum = float(best["theta_req_sum_deg"])
        cc = int(best["coverage"])
        if cc <= 0:
            prop0 = 1
        elif cc == 1:
            prop1 = 1
        elif cc == 2:
            prop2 = 1
        else:
            prop3p = 1
    else:
        # no feasible proposed solution -> treat as not covered
        prop0 = 1


    mean0 = mean1 = mean2 = mean3p = 0
    mean_time = np.nan
    mean_slew_sum = np.nan

    if dtm is not None:
        mean_time = float(dtm)
        theta_required = info["theta_required"]
        u_mean = info["u_mean"]

        mean_slew_sum = float(np.rad2deg(np.nansum(theta_required)))

        # propagate sample point to dtm
        if dtm == dt0:
            sample_pt_t = sample_pt.copy()
        else:
            sample_pt_t = sample_pt + v_target * dtm + 0.5 * a_target * dtm**2
        sample_pt_3d = np.array([sample_pt_t[0], sample_pt_t[1], 0.0], dtype=float)

        cc = coverage_count_point(sample_pt_3d, p_agents, u_mean, theta_h)
        if cc <= 0:
            mean0 = 1
        elif cc == 1:
            mean1 = 1
        elif cc == 2:
            mean2 = 1
        else:
            mean3p = 1
    else:
        # no epoch where everyone can point to mean -> treat as not covered
        mean0 = 1

    return {
        "seed": int(trial_seed),

        "Proposed_0_covered": prop0,
        "Proposed_1_covered": prop1,
        "Proposed_2_covered": prop2,
        "Proposed_3plus_covered": prop3p,
        "Proposed_sum_of_slews_deg": float(prop_slew_sum),
        "Proposed_time_sec": float(prop_time),

        "Mean_0_covered": mean0,
        "Mean_1_covered": mean1,
        "Mean_2_covered": mean2,
        "Mean_3plus_covered": mean3p,
        "Mean_sum_of_slews_deg": float(mean_slew_sum),
        "Mean_time_sec": float(mean_time),
    }

# ---------------------------------------------------------------------
# MPI main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_trials = int(cfg["num_trials"])
    base_seed = int(cfg["base_seed"])
    out_csv = cfg["output_csv"]

    # Each rank runs trials: idx where idx % size == rank
    local_rows: List[Dict[str, Any]] = []
    for trial_idx in range(num_trials):
        if (trial_idx % size) != rank:
            continue
        trial_seed = base_seed + trial_idx
        row = run_one_trial(cfg, trial_seed)
        local_rows.append(row)

    gathered = comm.gather(local_rows, root=0)

    if rank == 0:
        # flatten
        rows = [r for sub in gathered for r in sub]

        # sort by seed for readability
        rows.sort(key=lambda d: d["seed"])

        header = [
            "seed",
            "Proposed_0_covered", "Proposed_1_covered", "Proposed_2_covered", "Proposed_3plus_covered",
            "Proposed_sum_of_slews_deg", "Proposed_time_sec",
            "Mean_0_covered", "Mean_1_covered", "Mean_2_covered", "Mean_3plus_covered",
            "Mean_sum_of_slews_deg", "Mean_time_sec"
        ]

        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    main()
