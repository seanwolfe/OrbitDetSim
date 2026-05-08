"""
Plot OD outer-loop snapshots from a saved outer-loop CSV.

This script reconstructs only what is available in the outer-loop CSV:
  - target estimated state: x_est_0..x_est_5
  - target truth state:     x_true_0..x_true_5
  - covariance diagonal:    P_diag_0..P_diag_5  (axis-aligned ellipsoid approximation)
  - spacecraft states:      sc{i}_state_pre/post_0..5
  - spacecraft pointings:   sc{i}_pointing_pre/post_0..2

It calls your existing utilities.plot_od_scenario_3d_new().

Usage from PyCharm:
  1. Edit the block at the bottom under "EDIT THESE".
  2. Press Run.

Assumptions:
  - Run this from the same environment where `utilities.py` is importable.
  - If only P_diag_* is present, the plotted uncertainty ellipsoid is diagonal in EME.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import utilities as util


def _read_yaml(path: Optional[str | os.PathLike]) -> dict:
    if path is None or str(path).strip() == "":
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def _finite_float(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _vec_from_row(row: pd.Series, prefix: str, n: int) -> np.ndarray:
    vals = []
    for i in range(n):
        col = f"{prefix}_{i}"
        vals.append(_finite_float(row[col]) if col in row.index else np.nan)
    return np.asarray(vals, dtype=float)


def _infer_num_spacecraft(columns: Sequence[str]) -> int:
    pat = re.compile(r"^sc(\d+)_state_(?:pre|post)_0$")
    ids = []
    for c in columns:
        m = pat.match(c)
        if m:
            ids.append(int(m.group(1)))
    if not ids:
        raise ValueError(
            "Could not infer spacecraft count. Expected columns like sc0_state_post_0."
        )
    return max(ids) + 1


def _states_from_row(row: pd.Series, num_sc: int, phase: str = "post") -> np.ndarray:
    out = np.zeros((num_sc, 6), dtype=float)
    for sc_id in range(num_sc):
        out[sc_id] = _vec_from_row(row, f"sc{sc_id}_state_{phase}", 6)
    return out


def _pointings_from_row(row: pd.Series, num_sc: int, phase: str = "post") -> np.ndarray:
    out = np.zeros((num_sc, 3), dtype=float)
    for sc_id in range(num_sc):
        out[sc_id] = _vec_from_row(row, f"sc{sc_id}_pointing_{phase}", 3)
    return out


def _normalize_rows(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    n = np.linalg.norm(U, axis=1, keepdims=True)
    return U / np.maximum(n, eps)


def _cov_pos_from_row(row: pd.Series) -> tuple[np.ndarray, str]:
    """Return 3x3 position covariance. Prefer full P columns if present; else diagonal."""
    # Optional future-compatible full position covariance naming conventions.
    full_name_sets = [
        [[f"P_{i}{j}" for j in range(3)] for i in range(3)],
        [[f"P_{i}_{j}" for j in range(3)] for i in range(3)],
        [[f"P_pos_{i}{j}" for j in range(3)] for i in range(3)],
        [[f"P_pos_{i}_{j}" for j in range(3)] for i in range(3)],
    ]

    for names in full_name_sets:
        flat = [c for row_names in names for c in row_names]
        if all(c in row.index for c in flat):
            P = np.array([[_finite_float(row[names[i][j]]) for j in range(3)] for i in range(3)], dtype=float)
            P = 0.5 * (P + P.T)
            return P, "full position covariance"

    diag = _vec_from_row(row, "P_diag", 6)[:3]
    if not np.all(np.isfinite(diag)):
        raise ValueError("Could not reconstruct covariance: missing P_diag_0..P_diag_2.")
    return np.diag(np.maximum(diag, 0.0)), "diagonal covariance approximation from P_diag_0..2"


def _choose_rows(
    df: pd.DataFrame,
    iterations: Optional[Sequence[int]] = None,
    every_k: Optional[int] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    include_terminal: bool = True,
) -> pd.DataFrame:
    if "od_step_idx" not in df.columns:
        raise ValueError("Outer-loop CSV must contain od_step_idx.")

    work = df.copy()
    work["od_step_idx"] = pd.to_numeric(work["od_step_idx"], errors="coerce")
    work = work[np.isfinite(work["od_step_idx"])].copy()
    work["od_step_idx"] = work["od_step_idx"].astype(int)
    work = work.sort_values("od_step_idx")

    if start is not None:
        work = work[work["od_step_idx"] >= int(start)]
    if stop is not None:
        work = work[work["od_step_idx"] <= int(stop)]

    if iterations is not None and len(iterations) > 0:
        wanted = {int(x) for x in iterations}
        selected = work[work["od_step_idx"].isin(wanted)].copy()
    elif every_k is not None and int(every_k) > 1:
        k = int(every_k)
        selected = work[work["od_step_idx"] % k == 0].copy()
    else:
        selected = work.copy()

    if include_terminal and len(work) > 0:
        last_step = int(work["od_step_idx"].iloc[-1])
        if last_step not in set(selected["od_step_idx"].astype(int).tolist()):
            selected = pd.concat([selected, work.iloc[[-1]]], ignore_index=True)
            selected = selected.sort_values("od_step_idx")

    return selected.reset_index(drop=True)


def _default_config_value(config: dict, path: Sequence[str], default):
    cur = config
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _safe_bounds(config: dict, key: str, fallback):
    val = _default_config_value(config, ["three_d_prop", key], fallback)
    try:
        if val is None:
            return None
        if len(val) != 2:
            return fallback
        return (float(val[0]), float(val[1]))
    except Exception:
        return fallback


def _make_title(row: pd.Series, cov_source: str) -> str:
    step = int(_finite_float(row.get("od_step_idx", -1), -1))
    epoch = _finite_float(row.get("epoch_end_jdtdb", np.nan))
    event = str(row.get("event_type", ""))
    det = str(row.get("detecting_ids", ""))
    pos_err = _finite_float(row.get("pos_err_norm", np.nan))

    parts = [f"OD step {step}"]
    if np.isfinite(epoch):
        parts.append(f"epoch={epoch:.8f} JDTDB")
    if event:
        parts.append(event)
    if det:
        parts.append(f"det={det}")
    if np.isfinite(pos_err):
        parts.append(f"pos err={pos_err:.3g} km")
    parts.append(cov_source)
    return " | ".join(parts)


def plot_outer_loop_iterations(
    *,
    outer_csv_path: str | os.PathLike,
    config_path: Optional[str | os.PathLike] = None,
    out_dir: str | os.PathLike = "od_iteration_plots",
    iterations: Optional[Sequence[int]] = None,
    every_k: Optional[int] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    include_terminal: bool = True,
    save_png: bool = True,
    save_pdf: bool = False,
    show: bool = False,
    use_post_state: bool = True,
    show_current_boresights: bool = True,
    show_fov_cones: bool = True,
    show_uncertainty: bool = True,
    show_truth: bool = True,
    show_ems: bool = True,
    show_agent_orbit_tracks: bool = True,
    show_target_mean_traj: bool = True,
    show_true_target_traj: bool = True,
    show_coverage: bool = False,
    fov_style: str = "surface",
    fov_surface_alpha: float = 0.12,
    fov_n_rays: int = 2,
    fov_n_circle: int = 64,
    fov_n_len: int = 20,
    close_figures: bool = True,
):
    """Create 3D OD snapshot plots from one outer-loop CSV."""
    outer_csv_path = Path(outer_csv_path).expanduser().resolve()
    if not outer_csv_path.exists():
        raise FileNotFoundError(f"outer_csv_path not found: {outer_csv_path}")

    config = _read_yaml(config_path)
    df = pd.read_csv(outer_csv_path)
    if df.empty:
        raise ValueError(f"Outer-loop CSV is empty: {outer_csv_path}")

    num_sc = _infer_num_spacecraft(df.columns)
    selected = _choose_rows(
        df,
        iterations=iterations,
        every_k=every_k,
        start=start,
        stop=stop,
        include_terminal=include_terminal,
    )
    if selected.empty:
        raise ValueError("No rows selected. Check iterations/every_k/start/stop.")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plotting constants from config, with fallbacks.
    theta_h_rad = util.fov_deg2_to_half_angle_rad(float(config.get("fov", 17.8)))
    ray_length = float(config.get("ray_length", 8.0e6))
    d_mahal = float(config.get("d_mahal", 3.0))

    ems_center_xyz = np.asarray(_default_config_value(config, ["ems", "p_em"], [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    ems_radius = float(_default_config_value(config, ["ems", "R_em"], 0.0))

    xlim = _safe_bounds(config, "xlim", None)
    ylim = _safe_bounds(config, "ylim", None)
    zlim = _safe_bounds(config, "zlim", None)

    # Full trajectories reconstructed from outer-loop rows.
    x_est_traj = df[[f"x_est_{i}" for i in range(3)]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_true_traj = df[[f"x_true_{i}" for i in range(3)]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    phase = "post" if use_post_state else "pre"

    created = []
    print(f"Loaded {len(df)} outer rows from: {outer_csv_path}")
    print(f"Selected {len(selected)} rows for plotting.")
    print(f"Inferred num_spacecraft={num_sc}")

    for _, row in selected.iterrows():
        step = int(row["od_step_idx"])

        agents_state = _states_from_row(row, num_sc, phase=phase)
        agents_xyz = agents_state[:, :3]
        u_opt = _normalize_rows(_pointings_from_row(row, num_sc, phase="post"))
        u_curr = _normalize_rows(_pointings_from_row(row, num_sc, phase="pre"))

        x_est = _vec_from_row(row, "x_est", 6)
        x_true = _vec_from_row(row, "x_true", 6)
        P_pos, cov_source = _cov_pos_from_row(row)

        # Short per-agent tracks from all outer rows, using the same phase.
        agent_tracks = []
        if show_agent_orbit_tracks:
            for sc_id in range(num_sc):
                cols = [f"sc{sc_id}_state_{phase}_{i}" for i in range(3)]
                if all(c in df.columns for c in cols):
                    track = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                    agent_tracks.append(track)
                else:
                    agent_tracks.append(None)
        else:
            agent_tracks = None

        title = _make_title(row, cov_source)

        fig, ax = util.plot_od_scenario_3d_new(
            t_label=step,
            agents_xyz=agents_xyz,
            u_opt_agents_xyz=u_opt,
            theta_h_rad=theta_h_rad,
            ray_length=ray_length,
            u_curr_agents_xyz=u_curr,
            boresight_line_len=ray_length * 0.1,
            u_init_agents_xyz=None,
            init_boresight_line_len=ray_length * 1.5,
            agent_orbit_tracks_xyz=agent_tracks,
            spacecraft_orbit_xyz=None,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            target_mean_xyz=x_est[:3],
            target_cov_xyz=P_pos,
            d_mahal=d_mahal,
            true_target_xyz=x_true[:3],
            target_mean_traj_xyz=x_est_traj,
            true_target_traj_xyz=x_true_traj,
            true_target_traj_xyz_2=None,
            ems_center_xyz=ems_center_xyz,
            ems_radius=ems_radius,
            show_uncertainty=show_uncertainty,
            show_truth=show_truth,
            show_ems=show_ems,
            show_fov_cones=show_fov_cones,
            show_legend=True,
            show_target_mean_traj=show_target_mean_traj,
            show_true_target_traj=show_true_target_traj,
            show_true_target_traj_2=False,
            show_init_boresights=False,
            show_current_boresights=show_current_boresights,
            show_slew_angle_annotations=False,
            show_agent_name_annotations=True,
            show_agent_orbit_tracks=show_agent_orbit_tracks,
            show_spacecraft_orbit=False,
            show_coverage=show_coverage,
            Nx=int(_default_config_value(config, ["three_d_prop", "Nx"], 60)),
            Ny=int(_default_config_value(config, ["three_d_prop", "Ny"], 60)),
            Nz=int(_default_config_value(config, ["three_d_prop", "Nz"], 40)),
            show_pair_coverage=True,
            show_triple_coverage=True,
            pair_only_exact=True,
            pair_coverage_alpha=0.25,
            triple_coverage_alpha=0.35,
            fov_style=fov_style,
            fov_surface_alpha=fov_surface_alpha,
            fov_surface_color="lightskyblue",
            fov_n_rays=fov_n_rays,
            fov_n_circle=fov_n_circle,
            fov_n_len=fov_n_len,
            title=title,
            label_fontsize=9,
            label_offset_px=10,
            slew_label_offset_px=16,
            fill_alpha=0.10,
            sparse_wire=True,
            init_boresight_lw=1.5,
            init_boresight_alpha=0.95,
            slew_history=None,
        )

        base = f"{outer_csv_path.stem}__odstep_{step:05d}"
        if save_png:
            png_path = out_dir / f"{base}.png"
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
            created.append(str(png_path))
        if save_pdf:
            pdf_path = out_dir / f"{base}.pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
            created.append(str(pdf_path))

        if show:
            plt.show()
        elif close_figures:
            plt.close(fig)

    print("Created plot files:")
    for p in created:
        print("  ", p)
    return created


# =============================================================================
# EDIT THESE, THEN PRESS RUN IN PYCHARM
# =============================================================================
if __name__ == "__main__":
    OUTER_CSV_PATH = "/home/aeromec/Documents/sean/OrbitDetSim/results/od_res/outer_loop/minimoon-NESC000000GM_sc-1_index-10285_initial_tests_spacecraft_6_rank_1_part_1__OD_NBD_TBO_SPACE_CONSTRAINED_BASIN_HOPPING__outer.csv"
    CONFIG_PATH = "overall_orbitdetsim_config.yaml"
    OUT_DIR = "od_iteration_plots"

    # Choose ONE style:
    ITERATIONS = None         # e.g., [0, 1, 2, 5, 10]
    EVERY_K = 10                # e.g., 5 plots every 5th OD step; 1 plots all selected rows
    START = None               # e.g., 0
    STOP = None                # e.g., 50

    plot_outer_loop_iterations(
        outer_csv_path=OUTER_CSV_PATH,
        config_path=CONFIG_PATH,
        out_dir=OUT_DIR,
        iterations=ITERATIONS,
        every_k=EVERY_K,
        start=START,
        stop=STOP,
        include_terminal=True,
        save_png=True,
        save_pdf=False,
        show=False,
        use_post_state=True,
        show_current_boresights=True,
        show_fov_cones=True,
        show_uncertainty=True,
        show_truth=True,
        show_ems=True,
        show_agent_orbit_tracks=True,
        show_target_mean_traj=True,
        show_true_target_traj=True,
        show_coverage=False,  # set True only if you want expensive 3D coverage isosurfaces
        fov_style="surface",  # "surface", "wire", or "both"
    )
