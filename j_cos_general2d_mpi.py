#!/usr/bin/env python
import sys
import csv
import yaml
import numpy as np
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import time

# ==========================
# Import your existing helpers
# ==========================
# Adjust this import to match your project structure
from j_cost_general2d import (
    optimize_pointing_lbfgs_joint,
    coverage_count_2d,
    sample_from_uncertainty_2d,
)


# ==========================
# Single-trial simulation
# ==========================
def run_single_trial(global_seed, cfg):
    """
    Run one randomized geometry + attitude coordination trial and return:
    {
        "seed": int,
        "Proposed_0_covered": 0/1,
        ...
        "Mean_3plus_covered": 0/1,
    }
    """

    # Start timing
    t0 = time.perf_counter()

    rng = np.random.default_rng(global_seed)

    # ---- Unpack config ----
    M = cfg["geometry"]["M"]

    x_line_min = cfg["geometry"]["x_agents_min"]
    x_line_max = cfg["geometry"]["x_agents_max"]
    y_agents_min = cfg["geometry"]["y_agents_min"]
    y_agents_max = cfg["geometry"]["y_agents_max"]

    x_t_min = cfg["geometry"]["x_target_min"]
    x_t_max = cfg["geometry"]["x_target_max"]
    y_t_min = cfg["geometry"]["y_target_min"]
    y_t_max = cfg["geometry"]["y_target_max"]

    theta_h_min_deg = cfg["fov"]["theta_h_min_deg"]
    theta_h_max_deg = cfg["fov"]["theta_h_max_deg"]
    theta_s_deg = cfg["fov"]["theta_s_deg"]

    lambda_min = cfg["covariance"]["lambda_min"]
    lambda_max = cfg["covariance"]["lambda_max"]
    d_mahal = cfg["covariance"]["d_mahal"]

    kappa_sigma = cfg["optimizer"]["kappa_sigma"]
    n_mc = cfg["optimizer"]["n_mc"]
    n_restarts = cfg["optimizer"]["n_restarts"]

    # -----------------------
    # FOV half-angle (rad)
    # -----------------------
    theta_h = np.deg2rad(
        rng.uniform(theta_h_min_deg, theta_h_max_deg)
    )


    theta_s_list = np.array([np.deg2rad(theta_s_deg)] * M)

    # -----------------------
    # Agent positions in plane
    # -----------------------
    x_agents = rng.uniform(x_line_min, x_line_max, size=M)
    y_agents = rng.uniform(y_agents_min, y_agents_max, size=M)
    p_agents_2d = np.stack([x_agents, y_agents], axis=1)  # (M,2)

    # -----------------------
    # Target mean position
    # -----------------------
    p_hat_2d = np.array([
        rng.uniform(x_t_min, x_t_max),
        rng.uniform(y_t_min, y_t_max),
    ])

    # -----------------------
    # Initial pointings: axis-aligned best-guess (as in your main)
    # -----------------------
    axis_dirs = np.array([
        [0.0, 1.0],   # +y
        [1.0, 0.0],   # +x
        [0.0, -1.0],  # -y
        [-1.0, 0.0],  # -x
    ])
    axis_angles = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])

    pointing_angles_init = np.zeros(M)
    for i in range(M):
        rel = p_hat_2d - p_agents_2d[i]  # vector from agent to target
        n = np.linalg.norm(rel)
        if n < 1e-9:
            pointing_angles_init[i] = 0.0
            continue
        rel_unit = rel / n
        dots = axis_dirs @ rel_unit  # cosines with each axis direction
        idx_best = np.argmax(dots)
        pointing_angles_init[i] = axis_angles[idx_best]

    # -----------------------
    # 2D covariance: random eigenvalues + random rotation
    # -----------------------
    lambda1, lambda2 = rng.uniform(lambda_min, lambda_max, size=2)
    D = np.diag([lambda1, lambda2])

    phi_r = rng.uniform(0.0, 2.0 * np.pi)
    R = np.array([
        [np.cos(phi_r), -np.sin(phi_r)],
        [np.sin(phi_r),  np.cos(phi_r)]
    ])

    P_p_2d = R @ D @ R.T

    # Embed into 3D
    p_agents = np.hstack([p_agents_2d, np.zeros((M, 1))])      # (M,3)
    p_hat = np.array([p_hat_2d[0], p_hat_2d[1], 0.0])          # (3,)
    P_p = np.array([
        [P_p_2d[0, 0], P_p_2d[0, 1], 0.0],
        [P_p_2d[1, 0], P_p_2d[1, 1], 0.0],
        [0.0,          0.0,          1e-4],
    ])

    # Current 3D boresight directions from planar angles
    # convention: u = [sin(angle), cos(angle), 0]
    u_curr_agents = np.stack([
        np.sin(pointing_angles_init),
        np.cos(pointing_angles_init),
        np.zeros(M)
    ], axis=1)

    # -----------------------
    # Proposed method: optimize jointly (θ, φ)
    # -----------------------
    u_star, ang_star, J_star, history = optimize_pointing_lbfgs_joint(
        p_hat,
        P_p,
        p_agents,
        u_curr_agents,
        theta_h,
        theta_s_list,
        d_M=d_mahal,
        kappa_sigma=kappa_sigma,
        n_mc=n_mc,
        seed=global_seed,
        n_restarts=n_restarts
    )

    # If no feasible solution (e.g. too-tight slew constraints), we can
    # treat as "no coverage" for proposed method.
    if u_star is None:
        # We still want to generate a sample and evaluate mean method.
        proposed_pointing_angles = None
    else:
        # Convert optimized u* to planar angles
        proposed_pointing_angles = np.arctan2(u_star[:, 0], u_star[:, 1])

    # -----------------------
    # Mean-pointing method: each agent points exactly at p_hat_2d
    # using SAME angle convention as above: angle from +y axis
    # -----------------------
    pointing_angles_mean = np.zeros(M)
    for i in range(M):
        rel = p_hat_2d - p_agents_2d[i]
        n = np.linalg.norm(rel)
        if n < 1e-9:
            pointing_angles_mean[i] = 0.0
        else:
            # angle is atan2(x, y) to match u = [sin(angle), cos(angle), 0]
            pointing_angles_mean[i] = np.arctan2(rel[0], rel[1])

    # -----------------------
    # Draw sample from uncertainty ellipse (Mahalanobis radius d_mahal)
    # -----------------------
    sample_pt = sample_from_uncertainty_2d(
        p_hat_2d, P_p_2d, d_mahal=d_mahal, rng=rng
    )  # shape (2,)

    sample_grid = sample_pt.reshape(1, 2)

    # -----------------------
    # Coverage counts (proposed vs mean) at the sample
    # -----------------------
    # --- Proposed ---
    if proposed_pointing_angles is None:
        coverage_proposed = 0
    else:
        cov_arr_prop = coverage_count_2d(
            sample_grid,
            p_agents_2d,
            proposed_pointing_angles,
            theta_h
        )
        coverage_proposed = int(cov_arr_prop[0])

    # --- Mean-pointing ---
    cov_arr_mean = coverage_count_2d(
        sample_grid,
        p_agents_2d,
        pointing_angles_mean,
        theta_h
    )
    coverage_mean = int(cov_arr_mean[0])

    # -----------------------
    # Map coverage counts to 0/1 indicators per method
    # For each method only one of 0/1/2/3+ is 1.
    # -----------------------
    def coverage_flags(n):
        return {
            "0": int(n == 0),
            "1": int(n == 1),
            "2": int(n == 2),
            "3plus": int(n >= 3),
        }

    prop_flags = coverage_flags(coverage_proposed)
    mean_flags = coverage_flags(coverage_mean)

    # End timing
    t1 = time.perf_counter()
    comp_time = t1 - t0

    result_row = {
        "seed": int(global_seed),
        "comp_time_sec": comp_time,
        "Proposed_0_covered": prop_flags["0"],
        "Proposed_1_covered": prop_flags["1"],
        "Proposed_2_covered": prop_flags["2"],
        "Proposed_3plus_covered": prop_flags["3plus"],
        "Mean_0_covered": mean_flags["0"],
        "Mean_1_covered": mean_flags["1"],
        "Mean_2_covered": mean_flags["2"],
        "Mean_3plus_covered": mean_flags["3plus"],
    }

    return result_row


# ==========================
# MPI driver
# ==========================
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ----------------------
    # Rank 0 reads config
    # ----------------------
    if rank == 0:
        if len(sys.argv) < 2:
            raise SystemExit("Usage: mpirun -n N python mpi_attitude_coverage.py config.yaml")
        config_path = sys.argv[1]
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = None

    # Broadcast config to all ranks
    cfg = comm.bcast(cfg, root=0)

    num_trials = cfg["num_trials"]
    base_seed = cfg["base_seed"]
    output_csv = cfg["output_csv"]

    # Global trial indices [0, ..., num_trials-1]
    all_indices = np.arange(num_trials, dtype=int)

    # Simple block decomposition
    local_indices = all_indices[rank::size]

    # Each trial uses seed = base_seed + trial_idx
    local_results = []
    for trial_idx in local_indices:
        seed = base_seed + trial_idx
        row = run_single_trial(seed, cfg)
        local_results.append(row)

    # Gather results at root
    gathered = comm.gather(local_results, root=0)

    if rank == 0:
        # Flatten list-of-lists from all ranks
        all_rows = []
        for sublist in gathered:
            all_rows.extend(sublist)

        # Sort by seed for reproducibility
        all_rows.sort(key=lambda r: r["seed"])

        # Write CSV
        fieldnames = [
            "seed",
            "comp_time_sec",
            "Proposed_0_covered",
            "Proposed_1_covered",
            "Proposed_2_covered",
            "Proposed_3plus_covered",
            "Mean_0_covered",
            "Mean_1_covered",
            "Mean_2_covered",
            "Mean_3plus_covered",
        ]

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

        print(f"Wrote {len(all_rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
