import yaml
import os
import pandas as pd
from Asteroid import Asteroid
from Formation import Formation
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import ast
import spiceypy as spice
import argparse
from matplotlib.collections import LineCollection
import n_body_integrator as nbody
from astropy.time import Time
from scipy.integrate import odeint
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm

# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')
pd.options.mode.chained_assignment = None

import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline
import glob

import json
import re


def parse_vec_cell(cell, expected_len=None, dtype=float, default=None):
    """
    Parse a MASTER cell that may contain:
      - NaN/None/"" -> default (or NaNs)
      - a scalar -> length-1 array (or error if expected_len>1)
      - a list/tuple/np.ndarray -> array
      - a string like "1,2,3" or "1 2 3" or "[1,2,3]" or JSON list
    Returns: np.ndarray shape (N,)
    """
    if default is None:
        default = np.full((expected_len,), np.nan, dtype=dtype) if expected_len else np.array([], dtype=dtype)

    # None / NaN
    if cell is None:
        return default
    try:
        if isinstance(cell, float) and np.isnan(cell):
            return default
    except Exception:
        pass

    # already array-like
    if isinstance(cell, (list, tuple, np.ndarray)):
        arr = np.asarray(cell, dtype=dtype).ravel()
        if expected_len is not None and arr.size != expected_len:
            raise ValueError(f"Expected len={expected_len}, got {arr.size} from {cell}")
        return arr

    # numeric scalar
    if isinstance(cell, (int, float, np.integer, np.floating)):
        arr = np.asarray([cell], dtype=dtype)
        if expected_len is not None and arr.size != expected_len:
            raise ValueError(f"Expected len={expected_len}, got scalar from {cell}")
        return arr

    # string
    s = str(cell).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return default

    # Try JSON list first
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = json.loads(s.replace("(", "[").replace(")", "]"))
            arr = np.asarray(obj, dtype=dtype).ravel()
            if expected_len is not None and arr.size != expected_len:
                raise ValueError(f"Expected len={expected_len}, got {arr.size} from {s}")
            return arr
        except Exception:
            # fall through to delimiter parsing
            pass

    # Split by comma or whitespace (robust)
    # e.g. "1, 2, 3" or "1 2 3" or "1,2 3"
    parts = re.split(r"[,\s]+", s)
    parts = [p for p in parts if p != ""]
    try:
        arr = np.asarray([dtype(p) for p in parts], dtype=dtype).ravel()
    except Exception as e:
        raise ValueError(f"Could not parse vector cell: {cell!r}") from e

    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"Expected len={expected_len}, got {arr.size} from {cell!r}")
    return arr


def mahalanobis_ellipse_points(mu, Sigma, d_mahal=3.0, n=200):
    """
    Points on the ellipse (x-mu)^T Sigma^{-1} (x-mu) = d_mahal^2
    """
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-12)

    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)

    axes_lengths = d_mahal * np.sqrt(eigvals)
    ellipse_local = np.diag(axes_lengths) @ circle
    ellipse_world = (eigvecs @ ellipse_local).T + mu

    return ellipse_world


def plot_fov_wedge(ax, agent_pos, pointing_angle, half_angle,
                   ray_length=50.0, color='tab:blue', alpha=0.15, lw=1.5):
    """
    2D infinite-range cone -> wedge with two rays.
    pointing_angle is in radians, measured from +y axis (front) CCW.
    """
    x0, y0 = agent_pos

    def dir_from_plus_y(theta):
        return np.array([np.sin(theta), np.cos(theta)])

    def ccw_dir_from_plus_x(theta):
        return np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # (M,2)

    left_dir = ccw_dir_from_plus_x(pointing_angle - half_angle)
    right_dir = ccw_dir_from_plus_x(pointing_angle + half_angle)

    left_pt = agent_pos + ray_length * left_dir
    right_pt = agent_pos + ray_length * right_dir

    ax.plot([x0, left_pt[0]], [y0, left_pt[1]], color=color, lw=lw)
    ax.plot([x0, right_pt[0]], [y0, right_pt[1]], color=color, lw=lw)

    ax.fill([x0, left_pt[0], right_pt[0]],
            [y0, left_pt[1], right_pt[1]],
            color=color, alpha=alpha)


def topo_std_degkms_to_radkms(std_degkms):
    """
    Convert topocentric uncertainty std vector from
    (deg, km, deg/s, km/s) -> (rad, km, rad/s, km/s)

    Parameters
    ----------
    std_degkms : array_like, shape (6,)
        [ra, dec, rho, ra_dot, dec_dot, rho_dot]

    Returns
    -------
    std_radkms : ndarray, shape (6,)
    """
    std = np.asarray(std_degkms, dtype=float)
    if std.shape != (6,):
        raise ValueError("Expected std vector of shape (6,)")

    deg2rad = np.pi / 180.0

    std_radkms = std.copy()
    std_radkms[0] *= deg2rad  # RA
    std_radkms[1] *= deg2rad  # Dec
    std_radkms[3] *= deg2rad  # RA_dot
    std_radkms[4] *= deg2rad  # Dec_dot
    # rho, rho_dot unchanged (km, km/s)

    return std_radkms


def topocentric_alpha_delta_rho_6d(p_obj, v_obj, p_sc, v_sc, eps=1e-12):
    """
    Vectorized topocentric (alpha, delta, rho, alpha_dot, delta_dot, rho_dot).

    Accepts either:
      - p_obj, v_obj shape (3,) and p_sc, v_sc shape (3,)  -> returns (6,)
      - p_obj, v_obj shape (3,) and p_sc, v_sc shape (M,3) -> returns (M,6)
      - p_obj, v_obj shape (M,3) and p_sc, v_sc shape (M,3)-> returns (M,6)

    Angles in radians, rho in distance units, rates in rad/s and distance/s.
    """
    p_obj = np.asarray(p_obj, dtype=float)
    v_obj = np.asarray(v_obj, dtype=float)
    p_sc = np.asarray(p_sc, dtype=float)
    v_sc = np.asarray(v_sc, dtype=float)

    # Promote (3,) -> (1,3) for vectorized ops
    def to_2d(a):
        if a.ndim == 1:
            if a.shape != (3,):
                raise ValueError(f"Expected shape (3,), got {a.shape}")
            return a[None, :]
        if a.ndim == 2 and a.shape[1] == 3:
            return a
        raise ValueError(f"Expected shape (3,) or (M,3), got {a.shape}")

    Pobj = to_2d(p_obj)
    Vobj = to_2d(v_obj)
    Psc = to_2d(p_sc)
    Vsc = to_2d(v_sc)

    # Broadcast: allow single object state against many spacecraft states
    # (1,3) vs (M,3) -> (M,3)
    r = Pobj - Psc
    v = Vobj - Vsc

    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

    rho = np.linalg.norm(r, axis=1)
    rho = np.maximum(rho, eps)

    rxy2 = x * x + y * y
    rxy2_safe = np.maximum(rxy2, eps)
    rxy = np.sqrt(rxy2_safe)

    alpha = np.arctan2(y, x)
    delta = np.arctan2(z, rxy)

    # Range rate
    rho_dot = (x * vx + y * vy + z * vz) / rho

    # RA rate
    alpha_dot = (x * vy - y * vx) / rxy2_safe

    # Dec rate
    # delta_dot = (vz*rxy - z*(x*vx + y*vy)/rxy) / rho^2
    # use safe rxy to avoid division by zero
    rxy_safe = np.maximum(rxy, np.sqrt(eps))
    delta_dot = (vz * rxy_safe - z * (x * vx + y * vy) / rxy_safe) / (rho * rho)

    out = np.column_stack([alpha, delta, rho, alpha_dot, delta_dot, rho_dot])

    # If all inputs were (3,), return (6,)
    all_1d = (p_obj.ndim == 1 and v_obj.ndim == 1 and p_sc.ndim == 1 and v_sc.ndim == 1)
    if all_1d:
        return out[0, :]
    return out


def cov_radec_rho_6d_to_xyz_6d(y6, P_adr6, eps=1e-12):
    """
    Propagate covariance from (alpha, delta, rho, alpha_dot, delta_dot, rho_dot)
    to Cartesian (x, y, z, vx, vy, vz), using linearization.

    Parameters
    ----------
    y6 : array_like, shape (6,) or (M,6)
        [alpha, delta, rho, alpha_dot, delta_dot, rho_dot]
        angles in rad, rho in distance units, rates in rad/s and dist/s
    P_adr6 : array_like, shape (6,6)
        Covariance in (alpha, delta, rho, alpha_dot, delta_dot, rho_dot)
        with consistent units (rad, km, rad/s, km/s).

    Returns
    -------
    P_xyz6 : ndarray, shape (6,6) or (M,6,6)
        Cartesian covariance/covariances in (x,y,z,vx,vy,vz).
    """
    Y = np.asarray(y6, dtype=float)
    P = np.asarray(P_adr6, dtype=float)

    if P.shape != (6, 6):
        raise ValueError(f"P_adr6 must have shape (6,6); got {P.shape}")

    single = False
    if Y.ndim == 1:
        if Y.shape != (6,):
            raise ValueError(f"y6 must have shape (6,) or (M,6); got {Y.shape}")
        Y = Y[None, :]
        single = True
    elif Y.ndim == 2:
        if Y.shape[1] != 6:
            raise ValueError(f"y6 must have shape (M,6); got {Y.shape}")
    else:
        raise ValueError(f"y6 must have ndim 1 or 2; got ndim={Y.ndim}")

    M = Y.shape[0]

    alpha = Y[:, 0]
    delta = Y[:, 1]
    rho = Y[:, 2]
    alpha_dot = Y[:, 3]
    delta_dot = Y[:, 4]
    rho_dot = Y[:, 5]

    ca, sa = np.cos(alpha), np.sin(alpha)
    cd, sd = np.cos(delta), np.sin(delta)

    # u(alpha,delta)
    u = np.column_stack([cd * ca, cd * sa, sd])  # (M,3)

    # du/dalpha, du/ddelta
    du_dalpha = np.column_stack([-cd * sa, cd * ca, np.zeros(M)])  # (M,3)
    du_ddelta = np.column_stack([-sd * ca, -sd * sa, cd])  # (M,3)

    # ----- Position Jacobian J_r: (M,3,3) for [alpha, delta, rho] -----
    # columns: [rho*du_dalpha, rho*du_ddelta, u]
    J_r = np.zeros((M, 3, 3), dtype=float)
    J_r[:, :, 0] = rho[:, None] * du_dalpha
    J_r[:, :, 1] = rho[:, None] * du_ddelta
    J_r[:, :, 2] = u

    # ----- Velocity Jacobian J_v: (M,3,6) for [alpha, delta, rho, alpha_dot, delta_dot, rho_dot] -----
    J_v = np.zeros((M, 3, 6), dtype=float)

    # ∂v/∂alpha, ∂v/∂delta
    J_v[:, :, 0] = rho_dot[:, None] * du_dalpha
    J_v[:, :, 1] = rho_dot[:, None] * du_ddelta

    # ∂v/∂rho
    J_v[:, :, 2] = alpha_dot[:, None] * du_dalpha + delta_dot[:, None] * du_ddelta

    # ∂v/∂alpha_dot, ∂v/∂delta_dot, ∂v/∂rho_dot
    J_v[:, :, 3] = rho[:, None] * du_dalpha
    J_v[:, :, 4] = rho[:, None] * du_ddelta
    J_v[:, :, 5] = u

    # ----- Full Jacobian J: (M,6,6) -----
    J = np.zeros((M, 6, 6), dtype=float)
    J[:, 0:3, 0:3] = J_r
    J[:, 3:6, :] = J_v

    # Propagate: P_xyz = J P J^T for each i
    P_xyz = np.einsum('mij,jk,mlk->mil', J, P, J)

    return P_xyz[0] if single else P_xyz


def proj_angle_xy_from_plus_x_ccw(u_agents):
    """
    Compute the 2D projected angle of 3D pointing vectors onto the xy-plane,
    measured from +x, CCW, in radians.

    Parameters
    ----------
    u_agents : array_like, shape (M, 3)
        Pointing vectors (ideally unit). Only x,y are used for the angle.

    Returns
    -------
    alpha : ndarray, shape (M,)
        Angles in radians in (-pi, pi], from +x CCW.
        (Use np.mod(alpha, 2*np.pi) if you want [0, 2*pi).)
    """
    u = np.asarray(u_agents, dtype=float)
    if u.ndim != 2 or u.shape[1] != 3:
        raise ValueError(f"u_agents must have shape (M,3); got {u.shape}")

    x = u[:, 0]
    y = u[:, 1]
    return np.arctan2(y, x)


def coverage_count_2d(grid_pts, p_agents_2d, pointing_angles, theta_h):
    """
    grid_pts: (N,2) array of xy points
    p_agents_2d: (M,2)
    pointing_angles: (M,) angles in radians (from +y axis CCW)
    theta_h: FOV half-angle (scalar, radians)
    Returns:
        counts: (N,) array, number of FOVs that cover each point
    """

    def dir_from_plus_y(theta):
        return np.stack([np.sin(theta), np.cos(theta)], axis=-1)  # (M,2)

    def ccw_dir_from_plus_x(theta):
        return np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # (M,2)

    M = len(p_agents_2d)
    dirs = ccw_dir_from_plus_x(pointing_angles)  # (M,2)
    cos_th = np.cos(theta_h)

    counts = np.zeros(len(grid_pts), dtype=int)

    for i in range(M):
        v = grid_pts - p_agents_2d[i]  # (N,2)
        v_norm = np.linalg.norm(v, axis=1)
        good = v_norm > 1e-9  # avoid divide-by-zero
        v_unit = np.zeros_like(v)
        v_unit[good] = v[good] / v_norm[good, None]

        cosang = np.sum(v_unit * dirs[i], axis=1)
        inside = cosang >= cos_th  # boolean mask
        counts += inside.astype(int)

    return counts


def plot_od_scenario_2d(
        *,
        # epoch/meta
        t_label=None,

        # agents
        agents_xy,  # (M,2)
        pointing_angles_rad,  # (M,)
        theta_h_rad,  # scalar
        ray_length=10.0,

        # optional: current boresight vectors (for dotted line + slew text)
        # if provided: u_curr_agents_xy should be (M,2), not necessarily unit (we normalize)
        u_curr_agents_xy=None,
        boresight_line_len=3.0,

        # optional: orbit tracks / quasi-halo projections (provide whatever you have)
        # list length M; each entry is (K,2) array for that agent
        agent_orbit_tracks_xy=None,

        # coverage grid extents (you choose)
        xlim=None, ylim=None,
        Nx=500, Ny=500,

        # target uncertainty + truth
        target_mean_xy=None,  # (2,)
        target_cov_xy=None,  # (2,2)
        d_mahal=2.0,
        true_target_xy=None,  # (2,)

        # EMS zone (2D cross-section you provide)
        ems_center_xy=None,  # (2,)
        ems_radius=None,  # scalar

        # styling toggles
        show_coverage=True,
        show_uncertainty=True,
        show_truth=True,
        show_ems=True,
        title=None,
):
    """
    Pure visualization function: does not infer any states.
    Everything (positions, angles, mean/cov, tracks) is passed in.

    Returns (fig, ax).
    """
    A = np.asarray(agents_xy, dtype=float)
    M = A.shape[0]
    ang = np.asarray(pointing_angles_rad, dtype=float).reshape(M, )

    # Determine plot bounds if not provided
    if xlim is None or ylim is None:
        # basic fallback using agent positions and optional target mean/truth/ems
        xs = [A[:, 0]]
        ys = [A[:, 1]]
        if target_mean_xy is not None:
            mu = np.asarray(target_mean_xy, dtype=float).reshape(2, )
            xs.append([mu[0]]);
            ys.append([mu[1]])
        if true_target_xy is not None:
            tr = np.asarray(true_target_xy, dtype=float).reshape(2, )
            xs.append([tr[0]]);
            ys.append([tr[1]])
        if ems_center_xy is not None:
            ec = np.asarray(ems_center_xy, dtype=float).reshape(2, )
            xs.append([ec[0]]);
            ys.append([ec[1]])
        xall = np.concatenate([np.asarray(v).ravel() for v in xs])
        yall = np.concatenate([np.asarray(v).ravel() for v in ys])
        pad = 2.0
        if xlim is None:
            xlim = (np.min(xall) - pad, np.max(xall) + pad)
        if ylim is None:
            ylim = (np.min(yall) - pad, np.max(yall) + pad)

    # Coverage colormap (0,1,2,>=3)
    cmap_colors = np.array([
        [1, 1, 1, 1],  # 0 = white
        [0.6, 0.8, 1, 1],  # 1 = light blue
        [0.2, 0.7, 0.2, 1],  # 2 = green
        [1.0, 0.0, 0.0, 1]  # >=3 = red
    ])
    cov_cmap = ListedColormap(cmap_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cov_cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Coverage grid
    space = 5e6
    if show_coverage:
        xg = np.linspace(xlim[0], xlim[1], int(Nx))
        yg = np.linspace(ylim[0], ylim[1], int(Ny))
        XX, YY = np.meshgrid(xg, yg)
        grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

        cov = coverage_count_2d(grid, A, ang, theta_h_rad)
        cov_img = cov.reshape(int(Ny), int(Nx))

        ax.imshow(
            cov_img,
            extent=[xg.min(), xg.max(), yg.min(), yg.max()],
            origin='lower',
            cmap=cov_cmap,
            norm=norm,
            alpha=0.25,
            zorder=-5
        )

    # Orbit tracks / quasi-halo projections
    if agent_orbit_tracks_xy is not None:
        for i, trk in enumerate(agent_orbit_tracks_xy):
            if trk is None:
                continue
            trk = np.asarray(trk, dtype=float)
            if trk.ndim != 2 or trk.shape[1] != 2:
                raise ValueError(f"agent_orbit_tracks_xy[{i}] must be (K,2)")
            ax.plot(trk[:, 0], trk[:, 1], lw=1.2, alpha=0.7,
                    label="Agent orbit (proj.)" if i == 0 else None)

    # FOV wedges + agent markers
    agent_scatter = None
    for i in range(M):
        plot_fov_wedge(ax, A[i], ang[i], theta_h_rad, ray_length,
                       color='tab:blue', alpha=0.12, lw=1.5)
        sc = ax.scatter(A[i, 0], A[i, 1], color='tab:blue', s=60,
                        label='Agent position' if i == 0 else None)
        if agent_scatter is None:
            agent_scatter = sc
        ax.text(A[i, 0], A[i, 1] - 0.35, f"A{i}", color='tab:blue',
                ha='center', va='top')

    # Optional: initial/current boresight dotted line + slew angle annotation
    u_axis_proxy = None
    if u_curr_agents_xy is not None:
        Uc = np.asarray(u_curr_agents_xy, dtype=float)
        if Uc.shape != (M, 2):
            raise ValueError("u_curr_agents_xy must be (M,2) matching agents")

        for i in range(M):
            u = Uc[i]
            nu = np.linalg.norm(u)
            if nu > 0:
                u = u / nu

            p_end = A[i] + u * float(boresight_line_len)
            ln, = ax.plot([A[i, 0], p_end[0]], [A[i, 1], p_end[1]],
                          linestyle=':', color='black', lw=1.2,
                          label='Initial boresight' if i == 0 else None)
            if u_axis_proxy is None:
                u_axis_proxy = ln

            # Compute slew between current u and optimized u (same convention)
            u_opt = np.array([np.sin(ang[i]), np.cos(ang[i])], dtype=float)
            dot = float(np.clip(np.dot(u, u_opt), -1.0, 1.0))
            slew_deg = float(np.degrees(np.arccos(dot)))

            ax.text(A[i, 0], A[i, 1] + 0.5, f"{slew_deg:.1f}°",
                    ha='center', va='bottom', fontsize=9, color='black')

    # Target mean + uncertainty ellipse
    unc_mean_sc = None
    ellipse_line = None
    if show_uncertainty and (target_mean_xy is not None) and (target_cov_xy is not None):
        mu = np.asarray(target_mean_xy, dtype=float).reshape(2, )
        P = np.asarray(target_cov_xy, dtype=float).reshape(2, 2)

        unc_mean_sc = ax.scatter(mu[0], mu[1], color='tab:red', s=80,
                                 marker='x', linewidths=2,
                                 label='Target mean')
        ellipse_pts = mahalanobis_ellipse_points(mu, P, d_mahal=float(d_mahal), n=250)
        ellipse_line, = ax.plot(ellipse_pts[:, 0], ellipse_pts[:, 1],
                                color='tab:red', lw=2,
                                label='Uncertainty ellipse')
        ax.fill(ellipse_pts[:, 0], ellipse_pts[:, 1], color='tab:red', alpha=0.10)

    # True target position
    true_sc = None
    if show_truth and (true_target_xy is not None):
        tr = np.asarray(true_target_xy, dtype=float).reshape(2, )
        true_sc = ax.scatter(tr[0], tr[1], s=60, facecolors='none',
                             edgecolors='green', linewidths=2,
                             label='True position')

    # EMS zone (2D circle)
    ems_line = None
    if show_ems and (ems_center_xy is not None) and (ems_radius is not None) and (float(ems_radius) > 0):
        c = np.asarray(ems_center_xy, dtype=float).reshape(2, )
        R = float(ems_radius)
        th = np.linspace(0, 2 * np.pi, 240)
        x_c = c[0] + R * np.cos(th)
        y_c = c[1] + R * np.sin(th)
        ems_line, = ax.plot(x_c, y_c, color='orange', lw=2,
                            label='EMS zone (2D)')
        ax.fill(x_c, y_c, color='orange', alpha=0.1)
        ax.scatter(c[0], c[1], color='orange', s=60, marker='o')
        ax.text(c[0], c[1] + 0.3, "EMS", color='orange',
                ha='center', va='bottom')

    # Axes/labels/title
    ax.set_aspect('equal', adjustable='box')
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.grid(alpha=0.25)

    if title is None:
        # minimal title if you pass a time label
        if t_label is not None:
            title = f"Scenario @ {t_label}"
        else:
            title = "Scenario (2D)"

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Legend (coverage proxies + FOV proxy + others)
    handles = []

    if agent_scatter is not None:
        handles.append(agent_scatter)

    # FOV proxy
    fov_proxy = Line2D([0], [0], color='tab:blue', lw=1.5, label='Agent FOV')
    handles.append(fov_proxy)

    if u_axis_proxy is not None:
        handles.append(u_axis_proxy)

    if unc_mean_sc is not None:
        handles.append(unc_mean_sc)
    if ellipse_line is not None:
        handles.append(ellipse_line)
    if true_sc is not None:
        handles.append(true_sc)
    if ems_line is not None:
        handles.append(ems_line)

    if show_coverage:
        single_cov_patch = Patch(facecolor=cmap_colors[1], alpha=0.25, label='Single coverage')
        double_cov_patch = Patch(facecolor=cmap_colors[2], alpha=0.25, label='Double coverage')
        triple_cov_patch = Patch(facecolor=cmap_colors[3], alpha=0.25, label='Triple+ coverage')
        handles.extend([single_cov_patch, double_cov_patch, triple_cov_patch])

    ax.legend(handles=handles, loc='upper right')
    return fig, ax


def wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def topocentric_measurements_and_rates(obj_pos, obj_vel, sc_pos, sc_vel, eps=1e-12):
    """
    Compute topocentric RA/Dec/Range and their time derivatives from states.

    Inputs (all arrays shape (N,3)):
      obj_pos, obj_vel : object state (km, km/s)
      sc_pos,  sc_vel  : spacecraft state (km, km/s)

    Outputs (arrays shape (N,)):
      ra (rad), dec (rad), rho (km),
      ra_dot (rad/s), dec_dot (rad/s), rho_dot (km/s)
    """
    r_rel = obj_pos - sc_pos
    v_rel = obj_vel - sc_vel

    x = r_rel[:, 0]
    y = r_rel[:, 1]
    z = r_rel[:, 2]

    xd = v_rel[:, 0]
    yd = v_rel[:, 1]
    zd = v_rel[:, 2]

    r2 = x * x + y * y + z * z
    rho = np.sqrt(np.maximum(r2, eps))

    rxy2 = x * x + y * y
    rxy = np.sqrt(np.maximum(rxy2, eps))

    # Angles
    ra = np.arctan2(y, x)  # [-pi, pi]
    dec = np.arctan2(z, rxy)  # stable vs asin(z/r)

    # Range rate
    rho_dot = (x * xd + y * yd + z * zd) / rho  # km/s

    # RA rate: (x*yd - y*xd)/(x^2 + y^2)
    ra_dot = (x * yd - y * xd) / np.maximum(rxy2, eps)

    # Dec rate using dec = atan2(z, rxy)
    # rxy_dot = (x*xd + y*yd)/rxy
    rxy_dot = (x * xd + y * yd) / np.maximum(rxy, eps)
    # dec_dot = (zd*rxy - z*rxy_dot) / (rxy^2 + z^2) = (zd*rxy - z*rxy_dot)/rho^2
    dec_dot = (zd * rxy - z * rxy_dot) / np.maximum(r2, eps)

    return ra, dec, rho, ra_dot, dec_dot, rho_dot


def topocentric_rmse(
        final_pos, final_vel,
        true_pos, true_vel,
        sc_positions, sc_velocities,
):
    """
    Returns RMSEs between estimated vs true topocentric observables:
      RA, Dec, Range, RA_dot, Dec_dot, Range_rate.
    """
    ra_f, dec_f, rho_f, ra_dot_f, dec_dot_f, rho_dot_f = topocentric_measurements_and_rates(
        final_pos, final_vel, sc_positions, sc_velocities
    )
    ra_t, dec_t, rho_t, ra_dot_t, dec_dot_t, rho_dot_t = topocentric_measurements_and_rates(
        true_pos, true_vel, sc_positions, sc_velocities
    )

    # Angle differences must be wrapped
    dra = wrap_to_pi(ra_f - ra_t)
    ddec = wrap_to_pi(dec_f - dec_t)  # dec is also an angle; wrapping is safe

    # RMSEs
    ra_rmse = float(np.sqrt(np.mean(dra ** 2)))
    dec_rmse = float(np.sqrt(np.mean(ddec ** 2)))
    rho_rmse = float(np.sqrt(np.mean((rho_f - rho_t) ** 2)))

    ra_dot_rmse = float(np.sqrt(np.mean((ra_dot_f - ra_dot_t) ** 2)))
    dec_dot_rmse = float(np.sqrt(np.mean((dec_dot_f - dec_dot_t) ** 2)))
    rho_dot_rmse = float(np.sqrt(np.mean((rho_dot_f - rho_dot_t) ** 2)))

    return {
        "RA_RMSE_RAD": ra_rmse,
        "DEC_RMSE_RAD": dec_rmse,
        "RHO_RMSE": rho_rmse,
        "RA_DOT_RMSE_RADPS": ra_dot_rmse,
        "DEC_DOT_RMSE_RADPS": dec_dot_rmse,
        "RHO_DOT_RMSE": rho_dot_rmse,
    }


def interpolate_sc_traj(sc_poses, sc_vels, sc_times, num_points=25):
    # -----------------------------------
    # New time grid (1000 samples)
    # -----------------------------------

    sc_pos = sc_poses.detach().cpu().numpy()
    sc_vel = sc_vels.detach().cpu().numpy()

    sc_time = np.array([time.value - sc_times[0].value for time in sc_times]) * 86400

    t_new = np.linspace(sc_time[0], sc_time[-1], num_points)

    # -----------------------------------
    # Build Hermite splines per component
    # -----------------------------------
    splines = []
    for k in range(3):
        # For component k: r_k(t), v_k(t) = dr_k/dt
        spl = CubicHermiteSpline(
            sc_time,
            sc_pos[:, k],
            sc_vel[:, k]
        )
        splines.append(spl)

    # -----------------------------------
    # Evaluate interpolated position and velocity
    # -----------------------------------
    r_new = np.zeros((t_new.size, 3))
    v_new = np.zeros((t_new.size, 3))

    for k in range(3):
        spl = splines[k]
        r_new[:, k] = spl(t_new)  # position component
        v_new[:, k] = spl.derivative()(t_new)  # velocity component (dr/dt)

    # v from hermite wrong
    from scipy.interpolate import interp1d

    def interpolate_velocity_linear(sc_vels, sc_times, num_points=1000):
        """
        Linearly interpolate velocity vectors.

        Parameters
        ----------
        sc_vels : torch.Tensor or np.ndarray, shape (N, 3)
            Velocities (e.g., km/s)
        sc_times : sequence
            Time objects with .value in days
        num_points : int
            Number of output samples

        Returns
        -------
        t_new : np.ndarray, shape (num_points,)
            Interpolated times in seconds (relative to first)
        v_new : np.ndarray, shape (num_points, 3)
            Interpolated velocities (same units as input velocities)
        """

        # Convert inputs
        if hasattr(sc_vels, "detach"):
            v = sc_vels.detach().cpu().numpy()
        else:
            v = np.asarray(sc_vels)

        # Time in seconds relative to first sample
        t = np.array([t_i.value - sc_times[0].value for t_i in sc_times]) * 86400.0

        # New time grid
        t_new = np.linspace(t[0], t[-1], num_points)

        # Interpolate each velocity component
        v_new = np.zeros((num_points, 3))
        for k in range(3):
            f = interp1d(t, v[:, k], kind="linear")
            v_new[:, k] = f(t_new)

        return t_new, v_new

    tnew, vnew = interpolate_velocity_linear(sc_vels, sc_times)

    return r_new, vnew


def generate_iod_file(file_path, final_pos, final_vel, true_pos, true_vel, epochs):
    fx, fy, fz = final_pos[1][1:-1, 0], final_pos[1][1:-1, 1], final_pos[1][1:-1, 2]
    fvx, fvy, fvz = final_vel[1][1:-1, 0], final_vel[1][1:-1, 1], final_vel[1][1:-1, 2]
    fxn, fyn, fzn = final_pos[0][1:-1, 0], final_pos[0][1:-1, 1], final_pos[0][1:-1, 2]
    fvxn, fvyn, fvzn = final_vel[0][1:-1, 0], final_vel[0][1:-1, 1], final_vel[0][1:-1, 2]
    tx, ty, tz = true_pos[:, 0], true_pos[:, 1], true_pos[:, 2]
    tvx, tvy, tvz = true_vel[:, 0], true_vel[:, 1], true_vel[:, 2]

    if len(tx) > len(fx):
        tx, ty, tz = true_pos[:-1, 0], true_pos[:-1, 1], true_pos[:-1, 2]
        tvx, tvy, tvz = true_vel[:-1, 0], true_vel[:-1, 1], true_vel[:-1, 2]
    else:
        pass

    data = {
        "EPOCHS": epochs,
        "IOD_X": fx, "IOD_Y": fy, "IOD_Z": fz,
        "IOD_VX": fvx, "IOD_VY": fvy, "IOD_VZ": fvz,
        "IOD_X_NLLS": fxn, "IOD_Y_NLLS": fyn, "IOD_Z_NLLS": fzn,
        "IOD_VX_NLLS": fvxn, "IOD_VY_NLLS": fvyn, "IOD_VZ_NLLS": fvzn,
        "TRUE_X": tx, "TRUE_Y": ty, "TRUE_Z": tz,
        "TRUE_VX": tvx, "TRUE_VY": tvy, "TRUE_VZ": tvz
    }

    df = pd.DataFrame(data)

    df.to_csv(file_path, index=False)
    return df


def iod_viz(iod_data, results, pred_positions, pred_velocities, nlls_start, config, rmse_df):
    fig = plt.figure()

    # for plotting optimization progress in x, y, z
    total_length = len(results['TRAINING_EPOCH'])
    n = int(total_length / 20)
    indices = list(range(0, total_length, n))
    indices.append(-1)

    positions_filtered = [pred_positions[i] for i in indices]  # shape: (E, len(indices), 3)
    epoch_vals = results['TRAINING_EPOCH'].iloc[indices]
    if config['dynamics'] == 'CR3BP':
        observation_epochs = iod_data["EPOCH(JDTDB)"].values * 5.02189e6 / config['SECONDS_PER_DAY']
        x_vals = [pos[:, 0] * config['AU_TO_M'] / config['KM_TO_M'] for pos in positions_filtered]  # (E, len(indices))
        y_vals = [pos[:, 1] * config['AU_TO_M'] / config['KM_TO_M'] for pos in positions_filtered]
        z_vals = [pos[:, 2] * config['AU_TO_M'] / config['KM_TO_M'] for pos in positions_filtered]
        true_positions = iod_data.loc[:, ["GEO_X(KM)", "GEO_Y(KM)", "GEO_Z(KM)"]].values * config['AU_TO_M'] / config[
            'KM_TO_M']
    else:
        observation_epochs = iod_data["EPOCH(JDTDB)"].values
        x_vals = [pos[:, 0] for pos in positions_filtered]  # (E, len(indices))
        y_vals = [pos[:, 1] for pos in positions_filtered]
        z_vals = [pos[:, 2] for pos in positions_filtered]
        true_positions = iod_data.loc[:, ["GEO_X(KM)", "GEO_Y(KM)", "GEO_Z(KM)"]].values

    true_velocities = iod_data.loc[:, ["GEO_VX(KM/S)", "GEO_VY(KM/S)", "GEO_VZ(KM/S)"]].values

    ### x ###
    lines = []
    colors = []
    for epoch_val, x_val in zip(epoch_vals, x_vals):
        line = np.vstack((observation_epochs, x_val)).T
        lines.append(line)
        colors.append(epoch_val)

    lc = LineCollection(lines, cmap='coolwarm', array=np.array(colors), linewidth=2)
    ax = fig.add_subplot()  # 3D subplot
    ax.plot(observation_epochs, true_positions[:, 0], linestyle='--', color='black', zorder=15)
    ax.add_collection(lc)
    ax.autoscale()  # Auto scale limits to lines
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('X position [km]')

    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label('Training epoch')

    ### y ###
    lines = []
    colors = []
    for epoch_val, y_val in zip(epoch_vals, y_vals):
        line = np.vstack((observation_epochs, y_val)).T
        lines.append(line)
        colors.append(epoch_val)

    fig2 = plt.figure()
    lc2 = LineCollection(lines, cmap='coolwarm', array=np.array(colors), linewidth=2)
    ax2 = fig2.add_subplot()  # 3D subplot
    ax2.plot(observation_epochs, true_positions[:, 1], linestyle='--', color='black', zorder=15)
    ax2.add_collection(lc2)
    ax2.autoscale()  # Auto scale limits to lines
    ax2.set_xlabel('Time [days]')
    ax2.set_ylabel('Y position [km]')

    cbar2 = fig2.colorbar(lc2, ax=ax2)
    cbar2.set_label('Training epoch')

    ### z ###
    lines = []
    colors = []
    for epoch_val, z_val in zip(epoch_vals, z_vals):
        line = np.vstack((observation_epochs, z_val)).T
        lines.append(line)
        colors.append(epoch_val)

    fig3 = plt.figure()
    lc3 = LineCollection(lines, cmap='coolwarm', array=np.array(colors), linewidth=2)
    ax3 = fig3.add_subplot()  # 3D subplot
    ax3.plot(observation_epochs, true_positions[:, 2], linestyle='--', color='black', zorder=15)
    ax3.add_collection(lc3)
    ax3.autoscale()  # Auto scale limits to lines
    ax3.set_xlabel('Time [days]')
    ax3.set_ylabel('Z position [km]')

    cbar3 = fig3.colorbar(lc3, ax=ax3)
    cbar3.set_label('Training epoch')

    """
    # plotting basin hops in x, y, z
    total_length = len(pred_global_pos)
    n = 1
    indices = list(range(0, total_length, n))
    indices.append(-1)

    positions_filtered = [pred_global_pos[i] for i in indices]  # shape: (E, len(indices), 3)
    x_vals = [pos[:, 0] for pos in positions_filtered]  # (E, len(indices))
    y_vals = [pos[:, 1] for pos in positions_filtered]
    z_vals = [pos[:, 2] for pos in positions_filtered]
    epoch_vals = results['TRAINING_EPOCH'].iloc[indices]
    observation_epochs = iod_data["EPOCH(JDTDB)"].values
    true_positions = iod_data.loc[:, ["GEO_X(KM)", "GEO_Y(KM)", "GEO_Z(KM)"]].values
    true_velocities = iod_data.loc[:, ["GEO_VX(KM/S)", "GEO_VY(KM/S)", "GEO_VZ(KM/S)"]].values

    ### x ###
    lines = []
    colors = []
    for idx, x_val in zip(indices, x_vals):
        line = np.vstack((observation_epochs, x_val)).T
        lines.append(line)
        colors.append(idx)

    lc = LineCollection(lines, cmap='viridis', array=np.array(colors), linewidth=2)
    ax = fig.add_subplot(3, 3, 1)  # 3D subplot
    ax.plot(observation_epochs, true_positions[:, 0], linestyle='--', color='black', zorder=15)
    ax.add_collection(lc)
    ax.autoscale()  # Auto scale limits to lines
    ax.set_xlabel('Time ' + str(config['lambda']))
    ax.set_ylabel('X position')

    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label('Training epoch')

    ### y ###
    lines = []
    colors = []
    for idx,y_val in zip(indices, y_vals):
        line = np.vstack((observation_epochs, y_val)).T
        lines.append(line)
        colors.append(idx)

    lc2 = LineCollection(lines, cmap='viridis', array=np.array(colors), linewidth=2)
    ax2 = fig.add_subplot(3, 3, 2)  # 3D subplot
    ax2.plot(observation_epochs, true_positions[:, 1], linestyle='--', color='black', zorder=15)
    ax2.add_collection(lc2)
    ax2.autoscale()  # Auto scale limits to lines
    ax2.set_xlabel('Time ' + str(config['lambda']))
    ax2.set_ylabel('Y position')

    cbar2 = fig.colorbar(lc2, ax=ax2)
    cbar2.set_label('Training epoch')

    ### z ###
    lines = []
    colors = []
    for idx, z_val in zip(indices, z_vals):
        line = np.vstack((observation_epochs, z_val)).T
        lines.append(line)
        colors.append(idx)

    lc3 = LineCollection(lines, cmap='viridis', array=np.array(colors), linewidth=2)
    ax3 = fig.add_subplot(3, 3, 3)  # 3D subplot
    ax3.plot(observation_epochs, true_positions[:, 2], linestyle='--', color='black', zorder=15)
    ax3.add_collection(lc3)
    ax3.autoscale()  # Auto scale limits to lines
    ax3.set_xlabel('Time ' + str(config['lambda']))
    ax3.set_ylabel('Z position')

    cbar3 = fig.colorbar(lc3, ax=ax3)
    cbar3.set_label('Training epoch')
    """

    ###### physics loss ###
    num = 1
    points = np.vstack((results['TRAINING_EPOCH'].values, results['PHYSICS_LOSS'].values)).T
    points = points[::num]
    epoch_points = results['TRAINING_EPOCH'].values
    epoch_points = epoch_points[::num]
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    fig4 = plt.figure()
    lc4 = LineCollection(segments, cmap='coolwarm', array=epoch_points, linewidth=2)
    ax4 = fig4.add_subplot()  # 3D subplot
    ax4.add_collection(lc4)
    ax4.scatter(results['TRAINING_EPOCH'].iloc[nlls_start], results['PHYSICS_LOSS'].iloc[nlls_start])
    ax4.autoscale()  # Auto scale limits to lines
    ax4.set_xlabel('Overall Iteration')
    ax4.set_ylabel('Weighted Physics Loss')
    ax4.set_yscale('log')
    cbar4 = fig4.colorbar(lc4, ax=ax4)
    cbar4.set_label('Overall Iteration')

    ###### data loss ###
    points = np.vstack((results['TRAINING_EPOCH'].values, results['DATA_LOSS'].values)).T
    points = points[::num]
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    fig5 = plt.figure()
    lc5 = LineCollection(segments, cmap='coolwarm', array=epoch_points, linewidth=2)
    ax5 = fig5.add_subplot()  # 3D subplot
    ax5.add_collection(lc5)
    ax5.scatter(results['TRAINING_EPOCH'].iloc[nlls_start], results['DATA_LOSS'].iloc[nlls_start])
    ax5.autoscale()  # Auto scale limits to lines
    ax5.set_xlabel('Overall Iteration')
    ax5.set_ylabel('Observation Loss')
    ax5.set_yscale('log')
    cbar5 = fig5.colorbar(lc5, ax=ax5)
    cbar5.set_label('Overall Iteration')

    ###### data loss ###
    # points = np.vstack((results['TRAINING_EPOCH'].values, results['RANGE_LOSS'].values)).T
    # points = points[::num]
    # segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    # fig6 = plt.figure()
    # lc6 = LineCollection(segments, cmap='coolwarm', array=epoch_points, linewidth=2)
    # ax6 = fig6.add_subplot()  # 3D subplot
    # ax6.add_collection(lc6)
    # ax6.scatter(results['TRAINING_EPOCH'].iloc[nlls_start], results['RANGE_LOSS'].iloc[nlls_start])
    # ax6.autoscale()  # Auto scale limits to lines
    # ax6.set_xlabel('Overall Iteration')
    # ax6.set_ylabel('Weighted Range Loss')
    # ax6.set_yscale('log')
    # cbar6 = fig6.colorbar(lc6, ax=ax6)
    # cbar6.set_label('Overall Iteration')

    fig11 = plt.figure()
    ax21 = fig11.add_subplot()
    if config['optimizer'] == 'NLLS':
        label = 'NLLS'
    elif config['optimizer'] == 'SGD':
        label = 'SGD'
    else:
        label = 'BH+Range'

    # ax21.plot(*pred_positions[-2].T, label='Basin Hopping')

    ax21.plot(*true_positions[:, :2].T, label='True')
    if config['dynamics'] == 'CR3BP':
        ax21.plot(*(pred_positions[-1][:, :2] * config['AU_TO_M'] / config['KM_TO_M']).T, label=label)
        ax21.scatter(
            *(iod_data.loc[:, ["SC_GEO_X(KM)_PHYS", "SC_GEO_Y(KM)_PHYS"]].values * config['AU_TO_M'] / config[
                'KM_TO_M']).T,
            label='Observer Position')
    else:
        ax21.plot(*(pred_positions[-1][:, :2]).T, label=label)
        ax21.scatter(
            *(iod_data.loc[:, ["SC_GEO_X(KM)_PHYS", "SC_GEO_Y(KM)_PHYS"]].values).T,
            label='Observer Position')

    # true_rmse = rmse_df.loc[:, ['TRUE_X', 'TRUE_Y', 'TRUE_Z']].values
    # bh_pos_rmse = rmse_df.loc[:, ['IOD_X', 'IOD_Y', 'IOD_Z']].values
    # nlls_pos_rmse = rmse_df.loc[:, ['IOD_X_NLLS', 'IOD_Y_NLLS', 'IOD_Z_NLLS']].values

    # ax21.plot(*true_rmse.T, linestyle='--', label='RMSE True')
    # ax21.plot(*bh_pos_rmse.T, linestyle='--', label='RMSE BH')
    # ax21.plot(*nlls_pos_rmse.T, linestyle='--', label='RMSE NLLS')

    # ax21.plot(*asteroid_int_geo, label='Integrated', linestyle='--')
    # ax21.plot(*pred_global_pos[0].T, label="Initial")
    # ax21.plot(*pred_global_pos[-1].T, label='Final')

    # for i, pos in enumerate(pred_global_pos):
    #     ax6.plot(*pos.T, label=f"{i}")
    ax21.set_xlabel('X [KM]')
    ax21.set_ylabel('Y [KM]')
    # ax21.set_zlabel('Z [KM]')
    ax21.set_aspect('equal')
    ax21.legend()

    fig7 = plt.figure()
    ax7 = fig7.add_subplot()
    if config['dynamics'] == 'CR3BP':  # i.e. consistently non-dim
        mu = config['SYSTEM_MASS_PARAMETER']
        asteroid_ini_pos = pred_positions[-1][0, :]
        asteroid_ini_vel = pred_velocities[-1][0, :]
        ini_state = np.concatenate([asteroid_ini_pos, asteroid_ini_vel])
        phi_0 = np.eye(6)  # initial Phi (state transition matrix)
        state = np.hstack((np.array(ini_state), phi_0.ravel()))
        res = odeint(nbody.cr3bp, state, iod_data["EPOCH(JDTDB)"].values, args=(mu,))
        asteroid_cr3bp_position = np.array(res[:, :3])
        ax7.plot(*(pred_positions[-1][:, :2] * config['AU_TO_M'] / config['KM_TO_M']).T, label=label, zorder=10)
        ax7.plot(*(asteroid_cr3bp_position[:, :2] * config['AU_TO_M'] / config['KM_TO_M']).T, label='CR3BP Integrated',
                 linestyle='--', linewidth=3, zorder=5)
    else:
        # calc epochs
        num_frames = config['number_of_frames']
        asteroid_epoch = observation_epochs[0]
        step = config['time_between_frames'] / config['SECONDS_PER_DAY']
        total_observation_window = num_frames * step  # epoch is in jd

        # Function to get state vectors (position, velocity) in km & km/s
        epoch_et = spice.unitim(asteroid_epoch, 'JDTDB', 'ET')  # initial epoch

        def get_state(body, reference=10):
            state, _ = spice.spkgeo(body, epoch_et, "ECLIPJ2000", reference)
            return np.array(state)

        earth_state = get_state(399)

        asteroid_ini_pos_geo = pred_positions[-1][0, :]
        asteroid_ini_vel_geo = pred_velocities[-1][0, :]
        asteroid_state_geo = np.concatenate([asteroid_ini_pos_geo, asteroid_ini_vel_geo])
        asteroid_state_helio = eme_to_ecliptic_batch(asteroid_state_geo) + earth_state

        # integrate s/c traj
        asteroid_integrated_states, asteroid_earth_states = nbody.integrate_n_body(asteroid_state_helio,
                                                                                   asteroid_epoch,
                                                                                   total_observation_window *
                                                                                   config['SECONDS_PER_DAY'],
                                                                                   config['time_between_frames'],
                                                                                   type="ASTEROID")  # integrator takes seconds

        asteroid_int_geo = (asteroid_integrated_states - asteroid_earth_states)
        asteroid_eme = ecliptic_to_eme_batch(asteroid_int_geo)
        ast_epoch = Time(observation_epochs[0], format='jd', scale='tdb')
        asteroid_2bd_position, asteroid_2bd_velocity, asteroid_2bd_times = nbody.two_body_integrator(
            asteroid_ini_pos_geo,
            asteroid_ini_vel_geo,
            ast_epoch,
            config['time_between_frames'],
            num_frames)

        ax7.plot(*pred_positions[-1][:, :2].T, label='Predicted Pos')
        ax7.plot(*asteroid_eme[:2, :], label='N-body Integrated', linestyle='--', linewidth=3)
        # ax7.plot(*asteroid_2bd_position.T, label='2-body Integrated', linestyle='--', linewidth=3)

    ax7.set_xlabel('X [KM]')
    ax7.set_ylabel('Y [KM]')
    # ax7.set_zlabel('Z [KM]')
    ax7.set_aspect('equal')
    ax7.legend()

    if config['dynamics'] == 'CR3BP':
        true_v_rmse = rmse_df.loc[:, ['TRUE_VX', 'TRUE_VY', 'TRUE_VZ']].values * 29.8
        nlls_vel_rmse = rmse_df.loc[:, ['IOD_VX_NLLS', 'IOD_VY_NLLS', 'IOD_VZ_NLLS']].values * 29.8
        true_rmse = rmse_df.loc[:, ['TRUE_X', 'TRUE_Y', 'TRUE_Z']].values * config['AU_TO_M'] / config['KM_TO_M']
        pos_rmse = rmse_df.loc[:, ['IOD_X_NLLS', 'IOD_Y_NLLS', 'IOD_Z_NLLS']].values * config['AU_TO_M'] / config[
            'KM_TO_M']
    else:
        true_v_rmse = rmse_df.loc[:, ['TRUE_VX', 'TRUE_VY', 'TRUE_VZ']].values
        nlls_vel_rmse = rmse_df.loc[:, ['IOD_VX_NLLS', 'IOD_VY_NLLS', 'IOD_VZ_NLLS']].values
        true_rmse = rmse_df.loc[:, ['TRUE_X', 'TRUE_Y', 'TRUE_Z']].values
        pos_rmse = rmse_df.loc[:, ['IOD_X_NLLS', 'IOD_Y_NLLS', 'IOD_Z_NLLS']].values

    errors_xyz = np.abs(true_rmse - pos_rmse)
    x = errors_xyz[:, 0]
    y = errors_xyz[:, 1]
    z = errors_xyz[:, 2]

    errors_vxyz = np.abs(true_v_rmse - nlls_vel_rmse)
    vx = errors_vxyz[:, 0]
    vy = errors_vxyz[:, 1]
    vz = errors_vxyz[:, 2]

    bins = 10

    # Compute histograms for positions
    all_data = np.concatenate([x, y, z])
    counts_x, bin_edges = np.histogram(x, bins=bins, range=(all_data.min(), all_data.max()))
    counts_y, _ = np.histogram(y, bins=bin_edges)
    counts_z, _ = np.histogram(z, bins=bin_edges)

    # Compute histograms for velocities
    all_data_v = np.concatenate([vx, vy, vz])
    counts_vx, bin_edges_v = np.histogram(vx, bins=bins, range=(all_data_v.min(), all_data_v.max()))
    counts_vy, _ = np.histogram(vy, bins=bin_edges_v)
    counts_vz, _ = np.histogram(vz, bins=bin_edges_v)

    # Width of each bar
    width = (bin_edges[1] - bin_edges[0]) / 4

    # Subplot 2: Grouped bar chart for positions
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig8 = plt.figure()
    ax8 = fig8.add_subplot()
    ax8.bar(bin_centers - width, counts_x, width=width, label='X', color='r')
    ax8.bar(bin_centers, counts_y, width=width, label='Y', color='g')
    ax8.bar(bin_centers + width, counts_z, width=width, label='Z', color='b')

    # Add markers for the first element of each component
    # first_errors = [x[0], y[0], z[0]]
    # colors = ['r', 'g', 'b']
    # labels = ['x[0]', 'y[0]', 'z[0]']
    # marker_height = max(counts_x.max(), counts_y.max(), counts_z.max()) * 1.05
    #
    # for val, c, lbl in zip(first_errors, colors, labels):
    #     ax8.scatter(val, marker_height, color=c, marker='o', s=50, edgecolors='k', zorder=5, label=f'{lbl} marker')

    # To avoid duplicate legend labels, combine and deduplicate
    handles, labels = ax8.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax8.legend(unique.values(), unique.keys())

    # ax8.set_title('Histogram of Positions (Grouped Bars)')
    ax8.legend()
    ax8.grid(True)

    # Subplot 3: Grouped bar chart for velocities
    # Width of each bar
    width_v = (bin_edges_v[1] - bin_edges_v[0]) / 4
    bin_centers_v = (bin_edges_v[:-1] + bin_edges_v[1:]) / 2
    fig9 = plt.figure()
    ax9 = fig9.add_subplot()
    ax9.bar(bin_centers_v - width_v, counts_vx, width=width_v, label='vx', color='r')
    ax9.bar(bin_centers_v, counts_vy, width=width_v, label='vy', color='g')
    ax9.bar(bin_centers_v + width_v, counts_vz, width=width_v, label='vz', color='b')

    # Add markers for the first element of each component
    # first_errors = [vx[0], vy[0], vz[0]]
    # colors = ['r', 'g', 'b']
    # labels = ['vx[0]', 'vy[0]', 'vz[0]']
    # marker_height = max(counts_vx.max(), counts_vy.max(), counts_vz.max()) * 1.05
    #
    # for val, c, lbl in zip(first_errors, colors, labels):
    #     ax9.scatter(val, marker_height, color=c, marker='o', s=50, edgecolors='k', zorder=5, label=f'{lbl} marker')

    # To avoid duplicate legend labels, combine and deduplicate
    handles, labels = ax9.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax9.legend(unique.values(), unique.keys())

    # ax9.set_title('Histogram of Velocities (Grouped Bars)')
    ax9.legend()
    ax9.grid(True)

    plt.tight_layout()
    plt.show()

    return


def viz(object_pos, minimoon_pos, minimoon, sc_formation, ra_dec, configs):
    # asteroid position
    asteroid_pos = minimoon.orbit.loc[:, ['Synodic x', 'Synodic y', 'Synodic z']].values
    earth_pos = np.zeros_like(asteroid_pos)
    print(minimoon.id)
    moon_pos = minimoon.orbit.loc[:, ['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']].values

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(moon_pos[:, 0], moon_pos[:, 1], moon_pos[:, 2], label='Moon')
    ax.plot(asteroid_pos[:, 0], asteroid_pos[:, 1], asteroid_pos[:, 2], label='Asteroid', color='green', zorder=15)
    ax.scatter(1.5e6, 0, 0, label='L_1', s=20)
    # Create a sphere (Earth model)
    theta = np.linspace(0, np.pi, 30)  # Latitude
    phi = np.linspace(0, 2 * np.pi, 60)  # Longitude
    theta, phi = np.meshgrid(theta, phi)

    # Earth radius (approx. in arbitrary units)
    R = 6378  # Normalize radius

    # Convert spherical to Cartesian coordinates
    x = R * np.sin(theta) * np.cos(phi) / (configs['AU_TO_M'] / 1000)  # km
    y = R * np.sin(theta) * np.sin(phi) / (configs['AU_TO_M'] / 1000)
    z = R * np.cos(theta) / (configs['AU_TO_M'] / 1000)

    # Plot wireframe Earth
    ax.plot_wireframe(x, y, z, color="blue", linewidth=0.5, alpha=0.7)

    # start index is first instance asteroid is FOV of a sc, without occlusion from Earth or moon
    # it is the index in the minimoon trajectory corresponding to this
    sc_visible = []
    colors = ['red', 'blue', 'orange', 'green', 'grey', 'brown', 'black', 'purple', 'yellow', 'pink']
    for i, spacecraft in enumerate(sc_formation.spacecraft):

        sc_pos = spacecraft.matched_trajectory

        # find when the asteroid is in fov and not ocluded by earth or moon
        visible = spacecraft.asteroid_in_fov_batch(asteroid_pos, sc_pos, earth_pos, moon_pos, configs)
        sc_visible.append(visible)

        ######################
        # For generation of spacecrafr fov with asteroid figure
        ###########################################
        is_visible = visible[visible != -1]

        if len(is_visible) == 0:
            pass
        else:
            # for indi in is_visible:
            test_i = int(is_visible[0])
            ax.scatter(*minimoon.get_asteroid_pos(test_i), s=20, color='green', zorder=20)
            fov_corners = plot_fov_projection(spacecraft, minimoon, test_i)
            spacecraft_pos = spacecraft.get_spacecraft_pos(test_i)
            # Plot dotted lines from spacecraft to FOV corners
            for corner in fov_corners:
                ax.plot([spacecraft_pos[0], corner[0]],
                        [spacecraft_pos[1], corner[1]],
                        [spacecraft_pos[2], corner[2]], 'k--', alpha=0.5)

            # Draw FOV projection as a polygon
            fov_poly = Poly3DCollection([fov_corners], color='cyan', alpha=0.3, edgecolor='k')
            ax.add_collection3d(fov_poly)
            triad = [spacecraft_pos, spacecraft_pos - [0.001, 0, 0], spacecraft_pos - [0, 0.001, 0],
                     spacecraft_pos + [0, 0, 0.001]]
            x_axis = np.array([triad[0], triad[1]]).T
            y_axis = np.array([triad[0], triad[2]]).T
            z_axis = np.array([triad[0], triad[3]]).T
            ax.plot(*x_axis, color='black')
            ax.plot(*y_axis, color='black')
            ax.plot(*z_axis, color='black')
            print(np.rad2deg(np.arcsin(ra_dec[0])))
            print(np.rad2deg(np.arcsin(ra_dec[2])))

            for j, spacecraft_j in enumerate(sc_formation.spacecraft):
                spacecraft_pos_j = spacecraft_j.get_spacecraft_pos(test_i)
                sc_pos_j = spacecraft_j.matched_trajectory

                ax.plot(sc_pos_j[:test_i, 0], sc_pos_j[:test_i, 1], sc_pos_j[:test_i, 2], color=colors[j], zorder=15)
                ax.scatter(*spacecraft_j.get_spacecraft_pos(0), s=20, color=colors[j], label='Initial pos sc' + str(j),
                           zorder=20, marker='^')
                ax.scatter(*spacecraft_pos_j, s=20, color=colors[j], label='Detection instant sc ' + str(j), zorder=20)

    ax.scatter(object_pos[0, 0], object_pos[1, 0], object_pos[2, 0], color=colors[-1], s=30,
               label='Integration start sc', zorder=19)
    ax.plot(object_pos[0, :], object_pos[1, :], object_pos[2, :], color=colors[-1], linewidth=5,
            label='Integrated traj sc', zorder=14)
    ax.scatter(-minimoon_pos[0, 0], -minimoon_pos[1, 0], minimoon_pos[2, 0], color=colors[-2], s=30,
               label='Integration start minimoon', zorder=19)
    ax.plot(-minimoon_pos[0, :], -minimoon_pos[1, :], minimoon_pos[2, :], color=colors[-2], linewidth=5,
            label='Integrated traj minimoon', zorder=14)

    ax.plot(sc_pos[:, 0], sc_pos[:, 1], sc_pos[:, 2], color='pink', label='Halo Orbit', zorder=5)
    ax.set_xlabel('X (au)')
    ax.set_ylabel('Y (au)')
    ax.set_zlabel('Z (au)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.legend()
    ax.set_aspect('equal')
    plt.show()

    return sc_visible


def viz_geo_and_secr(object_pos, minimoon_pos, minimoon, sc_formation, ra_dec, minimoon_info, asteroid_geo,
                     spacecraft_geo, sc_eme_states_physically_sound, configs):
    """
    Visualize both in geo and in sun-earth co-rotating frame the detection instant
    :param object_pos:
    :param minimoon_pos:
    :param minimoon:
    :param sc_formation:
    :param ra_dec:
    :param configs:
    :return:
    """
    print(minimoon.id)
    colors = ['red', 'blue', 'orange', 'green', 'grey', 'brown', 'black', 'purple', 'yellow', 'pink']

    ############################
    # SECR Visualization
    ###########################

    # asteroid trajectory - SECR
    asteroid_pos = minimoon.orbit.loc[:, ['Synodic x', 'Synodic y', 'Synodic z']].values * (
            configs['AU_TO_M'] / configs['KM_TO_M'])

    # Moon trajecotry - SECR
    moon_pos = minimoon.orbit.loc[:, ['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']].values * (
            configs['AU_TO_M'] / configs['KM_TO_M'])

    # Create a sphere (Earth model)
    theta = np.linspace(0, np.pi, 30)  # Latitude
    phi = np.linspace(0, 2 * np.pi, 60)  # Longitude
    theta, phi = np.meshgrid(theta, phi)
    R = 6378  # Normalize radius # Earth radius (approx. in arbitrary units)
    x = R * np.sin(theta) * np.cos(phi)  # km # Convert spherical to Cartesian coordinates
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(moon_pos[:, 0], moon_pos[:, 1], moon_pos[:, 2], label='Moon')  # plot moon traj
    ax.plot(asteroid_pos[:, 0], asteroid_pos[:, 1], asteroid_pos[:, 2], label='Asteroid', color='green',
            zorder=15)  # plot asteroid traj
    ax.scatter(0.009, 0, 0, label='L_1', s=20)  # plot L_1
    ax.plot_wireframe(x, y, z, color="blue", linewidth=0.5, alpha=0.7)  # Plot wireframe Earth

    # start index is first instance asteroid is FOV of a sc, without occlusion from Earth or moon
    # it is the index in the minimoon trajectory corresponding to this
    for i, spacecraft in enumerate(sc_formation.spacecraft):

        # index of detection
        traj_index = int(minimoon_info['min_nonnegative'])

        # if this spacecraft detected the minimoon
        if i + 1 == int(minimoon_info.name[2]):

            # positiion of spacecraft i at detection instant
            spacecraft_pos = spacecraft.get_spacecraft_pos(traj_index) * (configs['AU_TO_M'] / configs['KM_TO_M'])

            fov_corners = plot_fov_projection(spacecraft, minimoon, traj_index)
            fov_corners = [fov_corner * (configs['AU_TO_M'] / configs['KM_TO_M']) for fov_corner in fov_corners]

            # plot fov related things
            ax.scatter(*minimoon.get_asteroid_pos(traj_index) * (configs['AU_TO_M'] / configs['KM_TO_M']), s=20,
                       color='green',
                       zorder=20)  # instant of detection on minimoon traj
            # Plot dotted lines from spacecraft to FOV corners
            for corner in fov_corners:
                ax.plot([spacecraft_pos[0], corner[0]],
                        [spacecraft_pos[1], corner[1]],
                        [spacecraft_pos[2], corner[2]], 'k--', alpha=0.5)

            # Draw FOV projection as a polygon
            fov_poly = Poly3DCollection([fov_corners], color='cyan', alpha=0.3, edgecolor='k')
            ax.add_collection3d(fov_poly)

            # print the obtained ra and dec
            ra = np.arctan2(ra_dec[0], ra_dec[1])  # returns radians in [-pi, pi]
            ra_deg = np.degrees(ra) % 360
            # print(ra_deg)
            # print(np.rad2deg(np.arcsin(ra_dec[2])))

        else:
            # non-detecting spacecraft trajectory and position at detection
            spacecraft_pos_j = spacecraft.get_spacecraft_pos(traj_index) * (configs['AU_TO_M'] / configs['KM_TO_M'])

            sc_pos_j = spacecraft.matched_trajectory * (configs['AU_TO_M'] / configs['KM_TO_M'])

            # plot trajectory up until detection instant
            ax.plot(sc_pos_j[:traj_index, 0], sc_pos_j[:traj_index, 1], sc_pos_j[:traj_index, 2], color=colors[i],
                    zorder=15)
            ax.scatter(*spacecraft.get_spacecraft_pos(0) * (configs['AU_TO_M'] / configs['KM_TO_M']), s=20,
                       color=colors[i], label='Initial pos sc' + str(i),
                       zorder=20, marker='^')
            ax.scatter(*spacecraft_pos_j, s=20, color=colors[i], label='Detection instant sc ' + str(i), zorder=20)

    # plot integration results
    ax.scatter(object_pos[0, 0], object_pos[1, 0], object_pos[2, 0], color=colors[-1], s=30,
               label='Integration start sc', zorder=19)  # s/c integration trajectory and initial position
    ax.plot(object_pos[0, :], object_pos[1, :], object_pos[2, :], color=colors[-1], linewidth=5,
            label='Integrated traj sc', zorder=14)
    ax.scatter(minimoon_pos[0, 0], minimoon_pos[1, 0], minimoon_pos[2, 0], color=colors[-2], s=30,
               label='Integration start minimoon', zorder=19)  # minimoon integration trajectory and initial position
    ax.plot(minimoon_pos[0, :], minimoon_pos[1, :], minimoon_pos[2, :], color=colors[-2], linewidth=5,
            label='Integrated traj minimoon', zorder=14)

    ax.set_xlabel('X (KM)')
    ax.set_ylabel('Y (KM)')
    ax.set_zlabel('Z (KM)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax.legend()
    ax.set_aspect('equal')
    # plt.show()

    ####################################
    # GEO Visualization
    ####################################

    # asteroid trajectory - GEO
    asteroid_pos = minimoon.orbit.loc[:, ["Geo x", "Geo y", "Geo z", "Geo vx", "Geo vy", "Geo vz"]].values
    asteroid_pos[:, :3] *= configs['AU_TO_M'] / configs['KM_TO_M']
    asteroid_pos[:, 3:] *= (configs['AU_TO_M'] / configs['KM_TO_M'] / configs['SECONDS_PER_DAY'])

    # Moon trajecotry - SECR
    moon_pos = (minimoon.orbit.loc[:, ["Moon x (Helio)",
                                       "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)",
                                       "Moon vy (Helio)", "Moon vz (Helio)"]].values - minimoon.orbit.loc[:,
                                                                                       ["Earth x (Helio)",
                                                                                        "Earth y (Helio)",
                                                                                        "Earth z (Helio)",
                                                                                        "Earth vx (Helio)",
                                                                                        "Earth vy (Helio)",
                                                                                        "Earth vz (Helio)"]].values)
    moon_pos[:, :3] *= configs['AU_TO_M'] / configs['KM_TO_M']
    moon_pos[:, 3:] *= (configs['AU_TO_M'] / configs['KM_TO_M'] / configs['SECONDS_PER_DAY'])

    asteroid_pos_eme = ecliptic_to_eme_batch(asteroid_pos.T).T
    moon_pos_eme = ecliptic_to_eme_batch(moon_pos.T).T
    asteroid_pos_eme = asteroid_pos_eme[:, :3]
    moon_pos_eme = moon_pos_eme[:, :3]

    # Create a sphere (Earth model)
    theta = np.linspace(0, np.pi, 30)  # Latitude
    phi = np.linspace(0, 2 * np.pi, 60)  # Longitude
    theta, phi = np.meshgrid(theta, phi)
    R = 6378  # Normalize radius # Earth radius (approx. in arbitrary units)
    x = R * np.sin(theta) * np.cos(phi)  # km # Convert spherical to Cartesian coordinates
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    ax2.plot(moon_pos_eme[:, 0], moon_pos_eme[:, 1], moon_pos_eme[:, 2], label='Moon')  # plot moon traj
    ax2.plot(asteroid_pos_eme[:, 0], asteroid_pos_eme[:, 1], asteroid_pos_eme[:, 2], label='Asteroid', color='green',
             zorder=15)  # plot asteroid traj
    ax2.plot_wireframe(x, y, z, color="blue", linewidth=0.5, alpha=0.7)  # Plot wireframe Earth

    # start index is first instance asteroid is FOV of a sc, without occlusion from Earth or moon
    # it is the index in the minimoon trajectory corresponding to this
    for i, spacecraft in enumerate(sc_formation.spacecraft):

        # index of detection
        traj_index = int(minimoon_info['min_nonnegative'])

        # if this spacecraft detected the minimoon
        if i + 1 == int(minimoon_info.name[2]):

            # positiion of spacecraft i at detection instant
            spacecraft_pos = spacecraft_geo[:3, 0]

            # Corotating frame state: shape (6, 1)
            boresight_vec = np.array([-1, 0, 0])  # shape (6, 1)

            # Earth state vector: already 1D (shape (6,))
            earth_state = minimoon.orbit.loc[traj_index, [
                "Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)",
                "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)"
            ]].values * (configs['AU_TO_M'] / configs['KM_TO_M'])  # Reshape to (6, 1)

            # Call the function with correctly shaped inputs
            geo_boresight = sun_earth_corotating_to_geo_eclip_single(boresight_vec, earth_state)
            geo_eme_boresight = ecliptic_to_eme_single(geo_boresight)

            fov_corners = plot_fov_projection_geo(geo_eme_boresight[:3], spacecraft_pos,
                                                  asteroid_pos_eme[traj_index, :],
                                                  spacecraft.fov)
            # fov_corners = [fov_corner * (configs['AU_TO_M'] / configs['KM_TO_M']) for fov_corner in fov_corners]

            # plot fov related things
            ax2.scatter(*asteroid_pos_eme[traj_index, :], s=20,
                        color='green',
                        zorder=20)  # instant of detection on minimoon traj
            # Plot dotted lines from spacecraft to FOV corners
            for corner in fov_corners:
                ax2.plot([spacecraft_pos[0], corner[0]],
                         [spacecraft_pos[1], corner[1]],
                         [spacecraft_pos[2], corner[2]], 'k--', alpha=0.5)

            # Draw FOV projection as a polygon
            fov_poly = Poly3DCollection([fov_corners], color='cyan', alpha=0.3, edgecolor='k')
            ax2.add_collection3d(fov_poly)
            # triad = [spacecraft_pos, spacecraft_pos - [0.001, 0, 0], spacecraft_pos - [0, 0.001, 0],
            #          spacecraft_pos + [0, 0, 0.001]]
            # x_axis = np.array([triad[0], triad[1]]).T
            # y_axis = np.array([triad[0], triad[2]]).T
            # z_axis = np.array([triad[0], triad[3]]).T
            # ax2.plot(*x_axis, color='black')
            # ax2.plot(*y_axis, color='black')
            # ax2.plot(*z_axis, color='black')
            # non-detecting spacecraft trajectory and position at detection
            spacecraft_pos_j = spacecraft.get_spacecraft_pos(traj_index) * (configs['AU_TO_M'] / configs['KM_TO_M'])

            sc_pos_j = spacecraft.matched_trajectory * (configs['AU_TO_M'] / configs['KM_TO_M'])

            # plot trajectory up until detection instant
            ax.plot(sc_pos_j[:traj_index, 0], sc_pos_j[:traj_index, 1], sc_pos_j[:traj_index, 2], color=colors[i],
                    zorder=15)
            ax.scatter(*spacecraft.get_spacecraft_pos(0) * (configs['AU_TO_M'] / configs['KM_TO_M']), s=20,
                       color=colors[i], label='Initial pos sc' + str(i),
                       zorder=20, marker='^')
            ax.scatter(*spacecraft_pos_j, s=20, color=colors[i], label='Detection instant sc ' + str(i), zorder=20)

        else:
            # non-detecting spacecraft trajectory and position at detection
            spacecraft_pos_j = spacecraft.get_spacecraft_pos(traj_index) * (configs['AU_TO_M'] / configs['KM_TO_M'])

            sc_pos_j = spacecraft.matched_trajectory * (configs['AU_TO_M'] / configs['KM_TO_M'])

            # plot trajectory up until detection instant
            ax.plot(sc_pos_j[:traj_index, 0], sc_pos_j[:traj_index, 1], sc_pos_j[:traj_index, 2], color=colors[i],
                    zorder=15)
            ax.scatter(*spacecraft.get_spacecraft_pos(0) * (configs['AU_TO_M'] / configs['KM_TO_M']), s=20,
                       color=colors[i], label='Initial pos sc' + str(i),
                       zorder=20, marker='^')
            ax.scatter(*spacecraft_pos_j, s=20, color=colors[i], label='Detection instant sc ' + str(i), zorder=20)

    # plot integration results
    ax2.scatter(spacecraft_geo[0, 0], spacecraft_geo[1, 0], spacecraft_geo[2, 0], color=colors[-1], s=30,
                label='Integration start sc', zorder=19)  # s/c integration trajectory and initial position
    ax2.scatter(spacecraft_geo[0, :], spacecraft_geo[1, :], spacecraft_geo[2, :], color=colors[-1], linewidth=5,
                label='Integrated traj sc', zorder=14)
    ax2.scatter(sc_eme_states_physically_sound[0, 0], sc_eme_states_physically_sound[1, 0],
                sc_eme_states_physically_sound[2, 0], color=colors[-3], s=50,
                label='Physically sound start', zorder=18)  # s/c integration trajectory and initial position
    ax2.plot(sc_eme_states_physically_sound[0, :], sc_eme_states_physically_sound[1, :],
             sc_eme_states_physically_sound[2, :], color=colors[-3], linewidth=5,
             label='Physically sound', zorder=14)
    ax2.scatter(asteroid_geo[0, 0], asteroid_geo[1, 0], asteroid_geo[2, 0], color=colors[-2], s=30,
                label='Integration start minimoon', zorder=19)  # minimoon integration trajectory and initial position
    ax2.scatter(asteroid_geo[0, :], asteroid_geo[1, :], asteroid_geo[2, :], color=colors[-2], linewidth=5,
                label='Integrated traj minimoon', zorder=14)

    # plot ra and dec lines
    cos_dec = np.sqrt(1 - ra_dec[2] ** 2)
    r_xy = np.sqrt(ra_dec[0] ** 2 + ra_dec[1] ** 2)

    x = ra_dec[1] / r_xy * cos_dec  # cos(RA) * cos(DEC)
    y = ra_dec[0] / r_xy * cos_dec  # sin(RA) * cos(DEC)
    z = ra_dec[2]
    dir_unit = np.stack([x, y, z], axis=0)  # shape (3, N)

    # Step 2: Compute distances to asteroid
    dist = np.linalg.norm(asteroid_geo - spacecraft_geo, axis=0)  # shape (N,)

    # Step 3: Scale directions
    scale = 1.5 * dist  # shape (N,)
    vecs = dir_unit * scale  # shape (3, N)

    # Plot line-of-sight vectors
    for i in range(spacecraft_geo.shape[1]):
        ax2.plot(
            [spacecraft_geo[0, i], spacecraft_geo[0, i] + vecs[0, i]],
            [spacecraft_geo[1, i], spacecraft_geo[1, i] + vecs[1, i]],
            [spacecraft_geo[2, i], spacecraft_geo[2, i] + vecs[2, i]],
            color='blue',
            alpha=0.6
        )

    ax2.set_xlabel('X (KM)')
    ax2.set_ylabel('Y (KM)')
    ax2.set_zlabel('Z (KM)')
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax2.zaxis.set_major_locator(MaxNLocator(nbins=4))  # Adjust nbins for number of ticks
    ax2.legend()
    ax2.set_aspect('equal')

    plt.show()
    return


# Define a converter function
def str_to_tuple(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val  # fallback
    return val


def read_master(file_path, config):
    columns_to_convert = config['visible_file_columns']
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.csv':
        return pd.read_csv(
            file_path,
            sep=',',
            converters={col: str_to_tuple for col in columns_to_convert},
            index_col=config['index_columns']
        )
    elif file_ext == '.parquet':
        df = pd.read_parquet(file_path)

        # Apply conversions manually after reading
        for col in columns_to_convert:
            if col in df.columns:
                df[col] = df[col].apply(str_to_tuple)
        return df
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


def read_IOD_data(file, configuration):
    file_ext = os.path.splitext(file)[1].lower()

    if file_ext == '.csv':
        return pd.read_csv(
            file,
            sep=',',
            header=0,
            names=configuration['IOD_data_columns']
        )
    elif file_ext == '.parquet':
        return pd.read_parquet(file)


def read_IOD_data_geo(file, configuration):
    file_ext = os.path.splitext(file)[1].lower()

    if file_ext == '.csv':
        return pd.read_csv(
            file,
            sep=',',
            header=0,
            names=configuration['IOD_data_columns_geo_and_phys']
        )
    elif file_ext == '.parquet':
        return pd.read_parquet(file)


def add_noise_to_angles(df, std_ra_deg=1.0, std_dec_deg=1.0):
    # Convert sin_ra and cos_ra to RA in degrees
    ra_rad = np.arctan2(df['SIN_RA'], df['COS_RA'])  # range [-π, π]
    ra_deg = np.degrees(ra_rad) % 360  # range [0, 360)

    # Convert sin_dec to Dec in degrees
    dec_rad = np.arcsin(df['SIN_DEC'])  # range [-π/2, π/2]
    dec_deg = np.degrees(dec_rad)  # range [-90, 90]

    # Add Gaussian noise in degrees
    ra_noisy_deg = (ra_deg + np.random.normal(0, std_ra_deg, size=len(df))) % 360
    dec_noisy_deg = np.clip(dec_deg + np.random.normal(0, std_dec_deg, size=len(df)), -90, 90)

    # Convert back to radians
    ra_noisy_rad = np.radians(ra_noisy_deg)
    dec_noisy_rad = np.radians(dec_noisy_deg)

    # Add noisy sin/cos columns
    df['SIN_RA_NOISY'] = np.sin(ra_noisy_rad)
    df['COS_RA_NOISY'] = np.cos(ra_noisy_rad)
    df['SIN_DEC_NOISY'] = np.sin(dec_noisy_rad)

    return df


def helio_eclip_to_geo_secr_generic(obj, earth, eps=1e-12, layout="auto"):
    """
    Convert heliocentric ECLIPJ2000 position(s) or state(s) to Earth-centered
    Sun–Earth co-rotating (SECR) frame with +Z fixed to ecliptic north.

    STRICT RULES:
    - obj dim = 3 → earth may be 3 or 6 (earth velocity ignored)
    - obj dim = 6 → earth MUST be 6 (otherwise ValueError)

    Supported obj shapes:
      Position: (3,), (M,3), (N,3), (3,N), (M,N,3)
      State:    (6,), (M,6), (N,6), (6,N), (M,N,6)

    Supported earth shapes:
      Position: (3,), (3,1), (1,3), (N,3), (3,N)
      State:    (6,), (6,1), (1,6), (N,6), (6,N)

    layout resolves ambiguity when obj is (K,dim):
      - "batch": interpret as (M,dim) objects at one time (N=1)
      - "time" : interpret as (N,dim) time series for one object (M=1)
      - "auto" : infer from earth shape when possible, else default to "batch"

    Returns: same layout as obj.
    """

    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    O = np.asarray(obj, dtype=float)
    E = np.asarray(earth, dtype=float)

    # --- infer dim (3 or 6) ---
    def infer_dim(A, name):
        if A.ndim == 1:
            if A.shape in [(3,), (6,)]:
                return A.shape[0]
            raise ValueError(f"{name} expected (3,) or (6,), got {A.shape}")
        if A.ndim == 2:
            if A.shape[0] in (3, 6):
                return A.shape[0]  # (dim,N)
            if A.shape[1] in (3, 6):
                return A.shape[1]  # (K,dim) or (N,dim)
        if A.ndim == 3:
            if A.shape[2] in (3, 6):
                return A.shape[2]
        raise ValueError(f"Could not infer dim for {name} with shape {A.shape}")

    obj_dim   = infer_dim(O, "obj")
    earth_dim = infer_dim(E, "earth")

    # --- STRICT RULE ---
    if obj_dim == 6 and earth_dim == 3:
        raise ValueError(
            "Invalid input: obj is 6D state but earth is 3D position. "
            "Earth velocity is required to compute omega and the SECR velocity correction."
        )

    # --- normalize earth to (N,earth_dim) ---
    def earth_to_Nd(E, N, dim):
        if E.ndim == 1:
            if E.shape != (dim,):
                raise ValueError(f"earth expected ({dim},), got {E.shape}")
            return np.repeat(E[None, :], N, axis=0)

        if E.ndim == 2:
            if E.shape == (dim, 1):
                return np.repeat(E[:, 0][None, :], N, axis=0)
            if E.shape == (1, dim):
                return np.repeat(E[0, :][None, :], N, axis=0)
            if E.shape[1] == dim:  # (N,dim)
                if E.shape[0] != N:
                    raise ValueError(f"earth has N={E.shape[0]} but obj has N={N}")
                return E
            if E.shape[0] == dim:  # (dim,N)
                if E.shape[1] != N:
                    raise ValueError(f"earth has N={E.shape[1]} but obj has N={N}")
                return E.T

        raise ValueError(f"earth unsupported shape {E.shape} for dim={dim}")

    # --- normalize obj to internal (M,N,obj_dim) and remember return style ---
    if O.ndim == 1:
        if O.shape != (obj_dim,):
            raise ValueError(f"obj expected ({obj_dim},), got {O.shape}")
        O_int = O[None, None, :]
        out_style = ("single",)

    elif O.ndim == 2:
        if O.shape == (obj_dim, 1):
            O_int = O[:, 0][None, None, :]
            out_style = ("single",)

        elif O.shape[0] == obj_dim:  # (dim,N)
            O_int = O.T[None, :, :]  # (1,N,dim)
            out_style = ("dimxN",)

        elif O.shape[1] == obj_dim:  # (K,dim) ambiguous
            K = O.shape[0]

            if layout == "batch":
                O_int = O[:, None, :]  # (M,1,dim)
                out_style = ("Mxdim",)

            elif layout == "time":
                O_int = O[None, :, :]  # (1,N,dim)
                out_style = ("Nxdim_time",)

            else:  # auto
                earth_time_like = (
                    E.ndim == 2 and (
                        (E.shape[1] == earth_dim and E.shape[0] == K) or
                        (E.shape[0] == earth_dim and E.shape[1] == K)
                    )
                )
                if earth_time_like:
                    O_int = O[None, :, :]  # (1,K,dim)
                    out_style = ("Nxdim_time",)
                else:
                    O_int = O[:, None, :]  # (K,1,dim)
                    out_style = ("Mxdim",)
        else:
            raise ValueError(f"obj unsupported shape {O.shape}")

    elif O.ndim == 3:
        if O.shape[2] != obj_dim:
            raise ValueError(f"obj expected (M,N,{obj_dim}), got {O.shape}")
        O_int = O
        out_style = ("MNdim",)

    else:
        raise ValueError(f"obj unsupported ndim={O.ndim}")

    M, N, _ = O_int.shape

    # Earth normalized to (N, earth_dim)
    E_N = earth_to_Nd(E, N, earth_dim)

    # Build Earth vector compatible with obj_dim
    if obj_dim == 3:
        # use only Earth position; ignore Earth velocity even if present
        E_use = E_N[:, :3]                # (N,3)
    else:
        # obj_dim==6 implies earth_dim==6 (strict)
        E_use = E_N                       # (N,6)

    # --- core math ---
    rE = E_use[:, :3]                     # (N,3)
    vE = E_use[:, 3:] if obj_dim == 6 else None

    rO = O_int[:, :, :3]                  # (M,N,3)
    rel_r = rO - rE[None, :, :]           # (M,N,3)

    if obj_dim == 6:
        vO = O_int[:, :, 3:]              # (M,N,3)
        rel_v = vO - vE[None, :, :]       # (M,N,3)

    # rotation angle from Earth->Sun direction (same convention as your original)
    angles = np.arctan2(-rE[:, 1], -rE[:, 0])  # (N,)
    c = np.cos(-angles)
    s = np.sin(-angles)

    R = np.zeros((N, 3, 3), dtype=float)
    R[:, 0, 0] = c;  R[:, 0, 1] = -s
    R[:, 1, 0] = s;  R[:, 1, 1] =  c
    R[:, 2, 2] = 1.0

    # position in rotating frame
    r_prime = np.einsum('nij,mnj->mni', R, rel_r)  # (M,N,3)

    if obj_dim == 3:
        out_int = r_prime
    else:
        # omega magnitude from Earth motion
        rE_norm2 = np.sum(rE * rE, axis=1)
        rE_norm2 = np.maximum(rE_norm2, eps)
        omega_mag = np.linalg.norm(np.cross(rE, vE), axis=1) / rE_norm2  # (N,)

        omega = np.zeros((N, 3), dtype=float)
        omega[:, 2] = omega_mag

        v_rel_rot = np.einsum('nij,mnj->mni', R, rel_v)        # (M,N,3)
        omega_prime = np.einsum('nij,nj->ni', R, omega)        # (N,3)

        v_rot = np.cross(omega_prime[None, :, :], r_prime)     # (M,N,3)
        v_prime = v_rel_rot - v_rot                            # (M,N,3)

        out_int = np.concatenate([r_prime, v_prime], axis=2)   # (M,N,6)

    # --- restore original layout ---
    if out_style[0] == "single":
        return out_int[0, 0, :]
    if out_style[0] == "Mxdim":
        return out_int[:, 0, :]
    if out_style[0] == "dimxN":
        return out_int[0, :, :].T
    if out_style[0] == "Nxdim_time":
        return out_int[0, :, :]
    return out_int


def geo_eclip_to_geo_eme_generic(x, eps=1e-12, layout="auto"):
    """
    Convert geocentric ECLIPJ2000 position(s) or state(s) to geocentric EME/J2000.

    If x has 3 components -> treat as position, return position.
    If x has 6 components -> treat as full state, return full state.

    Supported x shapes:
      Position: (3,), (M,3), (N,3), (3,N), (M,N,3)
      State:    (6,), (M,6), (N,6), (6,N), (M,N,6)

    layout resolves ambiguity when x is (K,3) or (K,6):
      - "batch": interpret as (M,dim) objects at one time (N=1)
      - "time" : interpret as (N,dim) time series for one object (M=1)
      - "auto" : default to "batch" (safer)

    Returns: same layout as x.
    """
    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    X = np.asarray(x, dtype=float)

    # --- infer dim (3 or 6) ---
    def infer_dim(A):
        if A.ndim == 1 and A.shape in [(3,), (6,)]:
            return A.shape[0]
        if A.ndim == 2:
            if A.shape[0] in (3, 6):  # (dim,N)
                return A.shape[0]
            if A.shape[1] in (3, 6):  # (K,dim) or (N,dim)
                return A.shape[1]
        if A.ndim == 3 and A.shape[2] in (3, 6):
            return A.shape[2]
        raise ValueError(f"Input must be position (3) or state (6); got shape {A.shape}")

    dim = infer_dim(X)

    # --- normalize to internal (M,N,dim) and remember output layout ---
    if X.ndim == 1:
        if X.shape != (dim,):
            raise ValueError(f"Expected ({dim},), got {X.shape}")
        X_int = X[None, None, :]
        out_style = ("single",)

    elif X.ndim == 2:
        if X.shape == (dim, 1):
            X_int = X[:, 0][None, None, :]
            out_style = ("single",)

        elif X.shape[0] == dim:              # (dim,N)
            X_int = X.T[None, :, :]          # (1,N,dim)
            out_style = ("dimxN",)

        elif X.shape[1] == dim:              # (K,dim) ambiguous
            if layout == "time":
                X_int = X[None, :, :]        # (1,N,dim)
                out_style = ("Nxdim_time",)
            else:
                X_int = X[:, None, :]        # (M,1,dim)
                out_style = ("Mxdim",)
        else:
            raise ValueError(f"Unsupported shape {X.shape} for dim={dim}")

    elif X.ndim == 3:
        if X.shape[2] != dim:
            raise ValueError(f"Expected (M,N,{dim}), got {X.shape}")
        X_int = X
        out_style = ("MNdim",)

    else:
        raise ValueError(f"Unsupported ndim={X.ndim}")

    # ---- rotation ecliptic -> EME about +x by +eps ----
    eps_deg = 23.439281
    eps_rad = np.deg2rad(eps_deg)
    c, s = np.cos(eps_rad), np.sin(eps_rad)

    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c, -s],
        [0.0,  s,  c]
    ], dtype=float)

    r = X_int[:, :, :3]
    r_eme = np.einsum("ij,mnj->mni", R, r)

    if dim == 3:
        out_int = r_eme
    else:
        v = X_int[:, :, 3:]
        v_eme = np.einsum("ij,mnj->mni", R, v)
        out_int = np.concatenate([r_eme, v_eme], axis=2)  # (M,N,6)

    # ---- restore original layout ----
    if out_style[0] == "single":
        return out_int[0, 0, :]
    if out_style[0] == "Mxdim":
        return out_int[:, 0, :]
    if out_style[0] == "dimxN":
        return out_int[0, :, :].T
    if out_style[0] == "Nxdim_time":
        return out_int[0, :, :]
    return out_int


def geo_eme_to_geo_eclip_generic(x, eps=1e-12, layout="auto"):
    """
    Convert geocentric EME/J2000 position(s) or state(s) to geocentric ECLIPJ2000.

    If x has 3 components -> treat as position, return position.
    If x has 6 components -> treat as full state, return full state.

    Supported x shapes:
      Position: (3,), (M,3), (N,3), (3,N), (M,N,3)
      State:    (6,), (M,6), (N,6), (6,N), (M,N,6)

    layout resolves ambiguity when x is (K,3) or (K,6):
      - "batch": interpret as (M,dim) objects at one time (N=1)
      - "time" : interpret as (N,dim) time series for one object (M=1)
      - "auto" : default to "batch" (safer)

    Returns: same layout as x.
    """
    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    X = np.asarray(x, dtype=float)

    # --- infer dim (3 or 6) ---
    def infer_dim(A):
        if A.ndim == 1 and A.shape in [(3,), (6,)]:
            return A.shape[0]
        if A.ndim == 2:
            if A.shape[0] in (3, 6):  # (dim,N)
                return A.shape[0]
            if A.shape[1] in (3, 6):  # (K,dim) or (N,dim)
                return A.shape[1]
        if A.ndim == 3 and A.shape[2] in (3, 6):
            return A.shape[2]
        raise ValueError(f"Input must be position (3) or state (6); got shape {A.shape}")

    dim = infer_dim(X)

    # --- normalize to internal (M,N,dim) and remember output layout ---
    if X.ndim == 1:
        if X.shape != (dim,):
            raise ValueError(f"Expected ({dim},), got {X.shape}")
        X_int = X[None, None, :]
        out_style = ("single",)

    elif X.ndim == 2:
        if X.shape == (dim, 1):
            X_int = X[:, 0][None, None, :]
            out_style = ("single",)

        elif X.shape[0] == dim:              # (dim,N)
            X_int = X.T[None, :, :]          # (1,N,dim)
            out_style = ("dimxN",)

        elif X.shape[1] == dim:              # (K,dim) ambiguous
            if layout == "time":
                X_int = X[None, :, :]        # (1,N,dim)
                out_style = ("Nxdim_time",)
            else:
                X_int = X[:, None, :]        # (M,1,dim)
                out_style = ("Mxdim",)
        else:
            raise ValueError(f"Unsupported shape {X.shape} for dim={dim}")

    elif X.ndim == 3:
        if X.shape[2] != dim:
            raise ValueError(f"Expected (M,N,{dim}), got {X.shape}")
        X_int = X
        out_style = ("MNdim",)

    else:
        raise ValueError(f"Unsupported ndim={X.ndim}")

    # ---- rotation EME -> ecliptic is inverse of (ecliptic -> EME) ----
    eps_deg = 23.439281
    eps_rad = np.deg2rad(eps_deg)
    c, s = np.cos(eps_rad), np.sin(eps_rad)

    # If ecliptic->EME used:
    # R = [[1,0,0],[0,c,-s],[0,s,c]]
    # then EME->ecliptic is R^T:
    R_T = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  s],
        [0.0, -s,  c]
    ], dtype=float)

    r = X_int[:, :, :3]
    r_ecl = np.einsum("ij,mnj->mni", R_T, r)

    if dim == 3:
        out_int = r_ecl
    else:
        v = X_int[:, :, 3:]
        v_ecl = np.einsum("ij,mnj->mni", R_T, v)
        out_int = np.concatenate([r_ecl, v_ecl], axis=2)  # (M,N,6)

    # ---- restore original layout ----
    if out_style[0] == "single":
        return out_int[0, 0, :]
    if out_style[0] == "Mxdim":
        return out_int[:, 0, :]
    if out_style[0] == "dimxN":
        return out_int[0, :, :].T
    if out_style[0] == "Nxdim_time":
        return out_int[0, :, :]
    return out_int


def helio_eclip_to_geo_eme_generic(obj, earth, eps=1e-12, layout="auto"):
    """
    Convert heliocentric ECLIPJ2000 position(s) or state(s) to geocentric EME/J2000.

    Rules:
    - obj dim = 3 → earth may be 3 or 6 (earth velocity ignored)
    - obj dim = 6 → earth MUST be 6 (otherwise ValueError)

    Supported obj shapes:
      Position: (3,), (M,3), (N,3), (3,N), (M,N,3)
      State:    (6,), (M,6), (N,6), (6,N), (M,N,6)

    Supported earth shapes:
      Position: (3,), (3,1), (1,3), (N,3), (3,N)
      State:    (6,), (6,1), (1,6), (N,6), (6,N)

    layout resolves ambiguity when obj is (K,dim):
      - "batch": interpret as (M,dim) objects at one time (N=1)
      - "time" : interpret as (N,dim) time series for one object (M=1)
      - "auto" : infer from earth shape when possible, else default to "batch"

    Returns: same layout as obj.
    """

    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    O = np.asarray(obj, dtype=float)
    E = np.asarray(earth, dtype=float)

    # ---------- infer dimension (3 or 6) ----------
    def infer_dim(A, name):
        if A.ndim == 1:
            if A.shape in [(3,), (6,)]:
                return A.shape[0]
        elif A.ndim == 2:
            if A.shape[0] in (3, 6):
                return A.shape[0]
            if A.shape[1] in (3, 6):
                return A.shape[1]
        elif A.ndim == 3:
            if A.shape[2] in (3, 6):
                return A.shape[2]
        raise ValueError(f"Could not infer dim for {name} with shape {A.shape}")

    obj_dim   = infer_dim(O, "obj")
    earth_dim = infer_dim(E, "earth")

    # ---------- STRICT RULE ----------
    if obj_dim == 6 and earth_dim == 3:
        raise ValueError(
            "Invalid input: obj is 6D state but earth is 3D position. "
            "Earth velocity is required for 6D transformation."
        )

    # ---------- normalize obj to (M,N,obj_dim) ----------
    if O.ndim == 1:
        O_int = O[None, None, :]
        out_style = ("single",)

    elif O.ndim == 2:
        if O.shape[0] == obj_dim:             # (dim,N)
            O_int = O.T[None, :, :]
            out_style = ("dimxN",)
        elif O.shape[1] == obj_dim:           # (K,dim)
            if layout == "time":
                O_int = O[None, :, :]
                out_style = ("Nxdim_time",)
            else:
                O_int = O[:, None, :]
                out_style = ("Mxdim",)
        else:
            raise ValueError(f"Unsupported obj shape {O.shape}")

    elif O.ndim == 3:
        if O.shape[2] != obj_dim:
            raise ValueError(f"Expected last dim {obj_dim}, got {O.shape}")
        O_int = O
        out_style = ("MNdim",)

    else:
        raise ValueError(f"Unsupported obj ndim {O.ndim}")

    M, N, _ = O_int.shape

    # ---------- normalize earth to (N,earth_dim) ----------
    def earth_to_Nd(E, N, dim):
        if E.ndim == 1:
            return np.repeat(E[None, :], N, axis=0)
        if E.ndim == 2:
            if E.shape[0] == dim:
                return E.T
            if E.shape[1] == dim:
                return E
        raise ValueError(f"Unsupported earth shape {E.shape}")

    E_N = earth_to_Nd(E, N, earth_dim)

    # ---------- build Earth vector compatible with obj_dim ----------
    if obj_dim == 3:
        E_use = E_N[:, :3]
    else:
        E_use = E_N   # earth_dim == 6 guaranteed here

    # ---------- heliocentric → geocentric ----------
    geo_ecl = O_int - E_use[None, :, :]

    # ---------- rotate ecliptic → EME ----------
    eps_rad = np.deg2rad(23.439281)
    c, s = np.cos(eps_rad), np.sin(eps_rad)

    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c, -s],
        [0.0,  s,  c]
    ])

    r = geo_ecl[:, :, :3]
    r_eme = np.einsum("ij,mnj->mni", R, r)

    if obj_dim == 3:
        out_int = r_eme
    else:
        v = geo_ecl[:, :, 3:]
        v_eme = np.einsum("ij,mnj->mni", R, v)
        out_int = np.concatenate([r_eme, v_eme], axis=2)

    # ---------- restore original layout ----------
    if out_style[0] == "single":
        return out_int[0, 0]
    if out_style[0] == "Mxdim":
        return out_int[:, 0]
    if out_style[0] == "dimxN":
        return out_int[0].T
    if out_style[0] == "Nxdim_time":
        return out_int[0]
    return out_int


def geo_secr_to_geo_eclip_generic(obj, earth, eps=1e-12, layout="auto"):
    """
    Convert SECR (Earth-centered rotating) position(s) or state(s) to geocentric ECLIPJ2000.

    STRICT RULES:
    - obj dim = 3 → earth may be 3 or 6 (earth velocity ignored)
    - obj dim = 6 → earth MUST be 6 (otherwise ValueError)

    Supported obj shapes:
      Position: (3,), (M,3), (N,3), (3,N), (M,N,3)
      State:    (6,), (M,6), (N,6), (6,N), (M,N,6)

    Supported earth shapes:
      Position: (3,), (3,1), (1,3), (N,3), (3,N)
      State:    (6,), (6,1), (1,6), (N,6), (6,N)

    layout resolves ambiguity when obj is (K,dim):
      - "batch": interpret as (M,dim) objects at one time (N=1)
      - "time" : interpret as (N,dim) time series for one object (M=1)
      - "auto" : infer from earth shape when possible, else default to "batch"

    Returns: same layout as obj.
    """

    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    O = np.asarray(obj, dtype=float)
    E = np.asarray(earth, dtype=float)

    # --- infer dim (3 vs 6) ---
    def infer_dim(A, name):
        if A.ndim == 1:
            if A.shape in [(3,), (6,)]:
                return A.shape[0]
            raise ValueError(f"{name} expected (3,) or (6,), got {A.shape}")
        if A.ndim == 2:
            if A.shape[0] in (3, 6):
                return A.shape[0]  # (dim,N)
            if A.shape[1] in (3, 6):
                return A.shape[1]  # (K,dim) or (N,dim)
        if A.ndim == 3:
            if A.shape[2] in (3, 6):
                return A.shape[2]
        raise ValueError(f"Could not infer dim for {name} with shape {A.shape}")

    obj_dim   = infer_dim(O, "obj")
    earth_dim = infer_dim(E, "earth")

    # --- STRICT RULE ---
    if obj_dim == 6 and earth_dim == 3:
        raise ValueError(
            "Invalid input: obj is 6D state but earth is 3D position. "
            "Earth velocity is required to compute omega and the inertial velocity correction."
        )

    # --- normalize earth to (N,earth_dim) ---
    def earth_to_Nd(E, N, dim):
        if E.ndim == 1:
            if E.shape != (dim,):
                raise ValueError(f"earth expected ({dim},), got {E.shape}")
            return np.repeat(E[None, :], N, axis=0)

        if E.ndim == 2:
            if E.shape == (dim, 1):
                return np.repeat(E[:, 0][None, :], N, axis=0)
            if E.shape == (1, dim):
                return np.repeat(E[0, :][None, :], N, axis=0)
            if E.shape[1] == dim:  # (N,dim)
                if E.shape[0] != N:
                    raise ValueError(f"earth has N={E.shape[0]} but obj has N={N}")
                return E
            if E.shape[0] == dim:  # (dim,N)
                if E.shape[1] != N:
                    raise ValueError(f"earth has N={E.shape[1]} but obj has N={N}")
                return E.T

        raise ValueError(f"earth unsupported shape {E.shape} for dim={dim}")

    # --- normalize obj to internal (M,N,obj_dim) and remember return style ---
    if O.ndim == 1:
        if O.shape != (obj_dim,):
            raise ValueError(f"obj expected ({obj_dim},), got {O.shape}")
        O_int = O[None, None, :]
        out_style = ("single",)

    elif O.ndim == 2:
        if O.shape == (obj_dim, 1):
            O_int = O[:, 0][None, None, :]
            out_style = ("single",)

        elif O.shape[0] == obj_dim:  # (dim,N)
            O_int = O.T[None, :, :]  # (1,N,dim)
            out_style = ("dimxN",)

        elif O.shape[1] == obj_dim:  # (K,dim) ambiguous
            K = O.shape[0]

            if layout == "batch":
                O_int = O[:, None, :]  # (M,1,dim)
                out_style = ("Mxdim",)

            elif layout == "time":
                O_int = O[None, :, :]  # (1,N,dim)
                out_style = ("Nxdim_time",)

            else:  # auto
                earth_time_like = (
                    E.ndim == 2 and (
                        (E.shape[1] == earth_dim and E.shape[0] == K) or
                        (E.shape[0] == earth_dim and E.shape[1] == K)
                    )
                )
                if earth_time_like:
                    O_int = O[None, :, :]  # (1,K,dim)
                    out_style = ("Nxdim_time",)
                else:
                    O_int = O[:, None, :]  # (K,1,dim)
                    out_style = ("Mxdim",)
        else:
            raise ValueError(f"obj unsupported shape {O.shape}")

    elif O.ndim == 3:
        if O.shape[2] != obj_dim:
            raise ValueError(f"obj expected (M,N,{obj_dim}), got {O.shape}")
        O_int = O
        out_style = ("MNdim",)

    else:
        raise ValueError(f"obj unsupported ndim={O.ndim}")

    M, N, _ = O_int.shape

    # Earth normalized to (N, earth_dim)
    E_N = earth_to_Nd(E, N, earth_dim)

    # For obj_dim==3, allow earth_dim==6 but ignore vel; for obj_dim==6 earth_dim==6 is guaranteed
    if obj_dim == 3:
        h_r_E = E_N[:, :3]          # (N,3)
        h_v_E = None
    else:
        h_r_E = E_N[:, :3]          # (N,3)
        h_v_E = E_N[:, 3:]          # (N,3)

    # ---- unpack SECR obj ----
    r_p = O_int[:, :, :3]          # (M,N,3) in SECR

    if obj_dim == 6:
        v_p = O_int[:, :, 3:]      # (M,N,3) in SECR

    # angles from Earth-Sun direction (same convention as forward function)
    angles = np.arctan2(-h_r_E[:, 1], -h_r_E[:, 0])  # (N,)

    c = np.cos(angles)
    s = np.sin(angles)

    # Rotation SECR -> inertial ecliptic: Rz(+angles)
    R = np.zeros((N, 3, 3), dtype=float)
    R[:, 0, 0] = c;  R[:, 0, 1] = -s
    R[:, 1, 0] = s;  R[:, 1, 1] =  c
    R[:, 2, 2] = 1.0

    # inertial geocentric position: r = R * r'
    geo_r = np.einsum('nij,mnj->mni', R, r_p)  # (M,N,3)

    if obj_dim == 3:
        out_int = geo_r
    else:
        # omega magnitude from Earth motion
        rE_norm2 = np.sum(h_r_E * h_r_E, axis=1)
        rE_norm2 = np.maximum(rE_norm2, eps)
        omega_mag = np.linalg.norm(np.cross(h_r_E, h_v_E), axis=1) / rE_norm2  # (N,)

        omega = np.zeros((N, 3), dtype=float)
        omega[:, 2] = omega_mag

        # omega in SECR coordinates (rotate inertial omega into SECR using same R)
        omega_p = np.einsum('nij,nj->ni', R, omega)  # (N,3)

        # inertial relative velocity: v = R * (v' + omega' x r')
        v_rel_p = v_p + np.cross(omega_p[None, :, :], r_p)  # (M,N,3)
        geo_v = np.einsum('nij,mnj->mni', R, v_rel_p)        # (M,N,3)

        out_int = np.concatenate([geo_r, geo_v], axis=2)     # (M,N,6)

    # ---- restore original layout ----
    if out_style[0] == "single":
        return out_int[0, 0, :]
    if out_style[0] == "Mxdim":
        return out_int[:, 0, :]
    if out_style[0] == "dimxN":
        return out_int[0, :, :].T
    if out_style[0] == "Nxdim_time":
        return out_int[0, :, :]
    return out_int


def helio_eclip_to_sun_earth_corotating_batch_full(states, earth_states):
    """
    Converts a batch of position and velocity state vectors from heliocentric ECLIPJ2000
    to the Earth-centered Sun-Earth co-rotating frame with fixed ecliptic north.

    Parameters:
    - states: (M, 6, N) array of states in heliocentric ECLIPJ2000, for M objects and N timesteps
    - earth_states: (6, N) array of Earth state vectors in the same frame at each timestep

    Returns:
    - states_corotating: (M, 6, N) array of transformed states in the SECR frame
    """

    _, N = states.shape

    h_r_E = earth_states[:3, :].T  # (N, 3)
    h_v_E = earth_states[3:, :].T  # (N, 3)

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angles = np.arctan2(-h_r_E[:, 1], -h_r_E[:, 0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angles = np.cos(-angles)
    sin_angles = np.sin(-angles)

    # Construct rotation matrices (shape: Nx3x3), Z axis is fixed along ecliptic north
    rotation_matrices = np.zeros((N, 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # Z remains unchanged (ecliptic north)

    # Angular velocity vector assuming uniform circular motion in ecliptic plane
    h_omega_mag = np.linalg.norm(np.cross(h_r_E, h_v_E), axis=1) / (np.linalg.norm(h_r_E, axis=1) ** 2)  # (N,)
    h_omega = np.zeros((N, 3))
    h_omega[:, 2] = h_omega_mag  # Only z-component for ecliptic plane rotation

    states_corotating = np.zeros_like(states)

    h_r_O = states[:3, :].T  # (N, 3)
    h_v_O = states[3:, :].T  # (N, 3)

    h_rel_r = h_r_O - h_r_E  # position relative to Earth (in inertial)
    h_rel_v = h_v_O - h_v_E  # velocity relative to Earth (in inertial)

    E_r_o_prime = np.einsum('nij,nj->ni', rotation_matrices, h_rel_r)  # now co-rotating frame
    v_rel_rot = np.einsum('nij,nj->ni', rotation_matrices, h_rel_v)
    E_omega = np.einsum('nij,nj->ni', rotation_matrices, h_omega)
    v_rot = np.cross(E_omega, E_r_o_prime)  # correct: using rotated position
    E_v_o_prime = v_rel_rot - v_rot  # total velocity in rotating frame

    states_corotating[:3, :] = E_r_o_prime.T
    states_corotating[3:, :] = E_v_o_prime.T

    return states_corotating


def sun_earth_corotating_to_geo_eclip_batch_full(states_corotating, earth_states):
    """
    Converts a batch of state vectors from the Sun-Earth co-rotating (SECR) frame
    to geocentric ECLIPJ2000 coordinates.

    Parameters:
    - states_corotating: (6, N) array in SECR frame; first 3 rows = position, last 3 = velocity
    - earth_states: (6, N) array in heliocentric ECLIPJ2000; used only for rotation angle

    Returns:
    - states_geocentric: (6, N) array in geocentric ECLIPJ2000
    """

    _, N = states_corotating.shape

    # Earth's heliocentric position (used only for rotation)
    h_r_E = earth_states[:3, :].T  # (N, 3)
    h_v_E = earth_states[3:, :].T  # (N, 3)

    # Co-rotating frame state
    E_r_o_prime = states_corotating[:3, :].T  # (N, 3)
    E_v_o_prime = states_corotating[3:, :].T  # (N, 3)

    # Compute rotation angles from Earth-Sun vector (negate for SECR to inertial)
    angles = np.arctan2(-h_r_E[:, 1], -h_r_E[:, 0])  # Shape: (N,)

    # Rotation matrices: from SECR to ECLIPJ2000
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    rotation_matrices = np.zeros((N, 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # z-axis (ecliptic north) unchanged

    # Rotate position to ECLIPJ2000 (geocentric)
    geo_r_o = np.einsum('nij,nj->ni', rotation_matrices, E_r_o_prime)  # (N, 3)

    # Angular velocity vector (magnitude from Earth's motion)
    h_omega_mag = np.linalg.norm(np.cross(h_r_E, h_v_E), axis=1) / (np.linalg.norm(h_r_E, axis=1) ** 2)  # (N,)
    h_omega = np.zeros((N, 3))
    h_omega[:, 2] = h_omega_mag  # z-axis angular velocity

    # Rotate angular velocity to SECR frame
    E_omega = np.einsum('nij,nj->ni', rotation_matrices, h_omega)

    # Add Coriolis term to get inertial velocity
    v_rot = np.cross(E_omega, E_r_o_prime)  # (N, 3)
    v_rel_rot = E_v_o_prime + v_rot  # (N, 3)
    geo_v_o = np.einsum('nij,nj->ni', rotation_matrices, v_rel_rot)  # (N, 3)

    # Assemble full state vector
    states_geocentric = np.zeros_like(states_corotating)
    states_geocentric[:3, :] = geo_r_o.T
    states_geocentric[3:, :] = geo_v_o.T

    return states_geocentric


def sun_earth_corotating_to_geo_eclip_single(pos_corot, earth_state_helio):
    """
    Convert a single position from Sun-Earth co-rotating frame to geocentric ECLIPJ2000.

    Parameters:
    - pos_corot: np.array shape (3,), position in Sun-Earth co-rotating frame
    - earth_state_helio: np.array shape (6,), Earth's heliocentric state vector [x,y,z,vx,vy,vz] in ECLIPJ2000

    Returns:
    - pos_geo_eclip: np.array shape (3,), position in geocentric ECLIPJ2000 frame
    """

    # Earth's heliocentric position
    h_r_E = earth_state_helio[:3]

    # Compute rotation angle: angle of Earth relative to Sun in XY plane (negated for SECR to inertial)
    angle = np.arctan2(-h_r_E[1], -h_r_E[0])

    # Rotation matrix about Z-axis by "angle"
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])

    # Rotate position vector from co-rotating to inertial ECLIPJ2000 frame
    pos_inertial = R @ pos_corot

    return pos_inertial


def ecliptic_to_eme_single(state_vectors_ecliptic):
    """
    Transforms a batch of full state vectors from Ecliptic J2000 to EME J2000.

    Parameters:
    - state_vectors_ecliptic (numpy array): 6xN array representing N state vectors
      in Ecliptic J2000 (rows: [x, y, z, vx, vy, vz]).

    Returns:
    - numpy array: 6xN array representing N state vectors in EME J2000.
    """

    # Obliquity of the ecliptic at J2000 (in degrees)
    epsilon = 23.439281
    epsilon_rad = np.radians(epsilon)

    # Rotation matrix about the x-axis (−epsilon for Ecliptic to EME)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(-epsilon_rad), np.sin(-epsilon_rad)],
        [0, -np.sin(-epsilon_rad), np.cos(-epsilon_rad)]
    ])

    # Separate position and velocity (each 3xN)
    pos = state_vectors_ecliptic[0:3]
    vel = state_vectors_ecliptic[3:6]

    # Apply rotation
    pos_eme = R @ pos

    return pos_eme


def get_sc_state_from_sc1_position(detected_pop, config):
    closest_indices = []
    scs_helio = []
    sc_epochs = []

    for kdx, detection in detected_pop.iterrows():
        # create a formation object, it has s/c s randomly placed
        formation = Formation(config)

        # we already had a saved formation, saved according to the s/c 1 position, get the correspoding index in overall orbit file
        sc1_ini_index = formation.get_index_from_pos(detection['spacecraft_1_ini_pos'])

        # re-initialize formation with this index
        formation.recall_formation(sc1_ini_index, config)

        # match the spacecraft trajectories to that of the asteroid in terms of length and sampling (asteroid sampled at one hour)
        formation.match_spacecraft_trajectory(int(detection['total_length']), config)

        # the spacecraft that detected the asteroid
        detecting_spacecraft = formation.spacecraft[detection.name[2] - 1]  # spacecraft id start from 1
        # detecting_spacecraft = formation.spacecraft[0]  # spacecraft id start from 1

        # the position of the detecting spacecraft at the detection instant
        desired_sc_pos = detecting_spacecraft.matched_trajectory[int(detection['min_nonnegative']), :] * (
                config['AU_TO_M'] / 1000)  # now in km sun-earth-syn

        # desired_sc_pos = detecting_spacecraft.matched_trajectory[0, :] * (
        #         config['AU_TO_M'] / 1000)

        # match this position to the overall orbit file
        possible_positions = formation.orbit.loc[:, ['SUN_EARTH_CO_X_(km)',
                                                     'SUN_EARTH_CO_Y_(km)',
                                                     'SUN_EARTH_CO_Z_(km)']]
        distances = np.linalg.norm(possible_positions - desired_sc_pos, axis=1)
        closest_position_index = np.argmin(distances)

        # use the index at the match to query spacecraft state vector
        geo_eme_state = formation.orbit.loc[
            formation.orbit.index[closest_position_index], ["GEO_EME_X_(km)", "GEO_EME_Y_(km)", "GEO_EME_Z_(km)",
                                                            "GEO_EME_Vx_(km/s)", "GEO_EME_Vy_(km/s)",
                                                            "GEO_EME_Vz_(km/s)"]].to_numpy()

        # get the earth's state vector at detection instant
        sc_time = formation.orbit.loc[formation.orbit.index[closest_position_index], "Time"]

        # get the detecting spacecraft state in geo eclip frame
        geo_eclip_state = eme_to_ecliptic_batch(geo_eme_state)
        scs_helio.append(geo_eclip_state)
        closest_indices.append(closest_position_index)
        sc_epochs.append(sc_time.strftime("%Y-%m-%d %H:%M:%S"))

    detected_pop.loc[:, 'detecting_sc_lpf_orbit_index'] = closest_indices
    detected_pop.loc[:, 'sc_epoch'] = sc_epochs
    detected_pop.loc[:,
    ['GEO_ECLIP_X_(km)', 'GEO_ECLIP_Y_(km)', 'GEO_ECLIP_Z_(km)', 'GEO_ECLIP_Vx_(km/s)', 'GEO_ECLIP_Vy_(km/s)',
     'GEO_ECLIP_Vz_(km/s)']] = np.array(scs_helio)

    return detected_pop


def get_scs_initial_states(detected_pop, config):
    """
    For each detection row in `detected_pop`:
      • Rebuild formation from saved SC1 initial position
      • Match the detecting spacecraft position at the first detection instant to formation.orbit
      • (Augment DataFrame) write detecting sc GEO_ECLIP columns, LPF index, epoch
      • Additionally: for *every* spacecraft j, match its position at the same
        detection instant to formation.orbit and read GEO_EME [km, km/s] state.
    Returns:
      (aug_df, all_sc_geo_eme_list, detecting_ids)
        - aug_df: augmented DataFrame (same as before)
        - all_sc_geo_eme_list: list of arrays, each (num_sc, 6), ordered by sc ID (1..num_sc)
        - detecting_ids: list of detecting spacecraft IDs (1-based)
    """

    out_df = detected_pop.copy(deep=True)

    # Accumulators for DataFrame fields
    closest_indices = []
    sc_epochs = []
    det_geo_eclip_states = []

    # Extra returns
    all_sc_geo_eclip_list = []  # per-row: array (num_sc, 6)
    detecting_ids = []
    all_sc_boresights_list = []

    # Convenience: columns we read from formation.orbit for GEO_EME state
    eme_cols = ["GEO_EME_X_(km)", "GEO_EME_Y_(km)", "GEO_EME_Z_(km)",
                "GEO_EME_Vx_(km/s)", "GEO_EME_Vy_(km/s)", "GEO_EME_Vz_(km/s)"]

    for kdx, detection in out_df.iterrows():
        # 1) Recreate formation from saved SC1 initial pos
        formation = Formation(config)
        sc1_ini_index = formation.get_index_from_pos(detection['spacecraft_1_ini_pos'])
        formation.recall_formation(sc1_ini_index, config)
        formation.match_spacecraft_trajectory(int(detection['total_length']), config)

        # 2) Detecting spacecraft ID (1-based, from MultiIndex third level)
        sc_id_detect = int(detection.name[2])
        detecting_ids.append(sc_id_detect)
        detecting_spacecraft = formation.spacecraft[sc_id_detect - 1]

        # sample index of first detection for this row
        idx0 = int(detection['min_nonnegative'])

        # 3) Detecting s/c desired position at detection instant (SECR→km for matching)
        desired_sc_pos_km = detecting_spacecraft.matched_trajectory[idx0, :] * (config['AU_TO_M'] / 1000.0)

        # 4) Match to formation.orbit ‘SUN_EARTH_CO_*’ positions to get the LPF index
        possible_positions = formation.orbit.loc[:, ['SUN_EARTH_CO_X_(km)',
                                                     'SUN_EARTH_CO_Y_(km)',
                                                     'SUN_EARTH_CO_Z_(km)']].to_numpy()
        dists = np.linalg.norm(possible_positions - desired_sc_pos_km, axis=1)
        closest_position_index = int(np.argmin(dists))
        closest_indices.append(closest_position_index)

        # 5) Read detecting s/c GEO_EME state from that row, convert to GEO_ECLIP for DataFrame columns
        geo_eme_state_detect = formation.orbit.loc[
            formation.orbit.index[closest_position_index], eme_cols
        ].to_numpy(dtype=float)

        # Your code put GEO_ECLIP in the DF; keep doing that for compatibility
        geo_eclip_state_detect = eme_to_ecliptic_batch(geo_eme_state_detect)
        det_geo_eclip_states.append(geo_eclip_state_detect)

        # Epoch (string)
        sc_time = formation.orbit.loc[formation.orbit.index[closest_position_index], "Time"]
        sc_epochs.append(sc_time.strftime("%Y-%m-%d %H:%M:%S"))

        # 6) Now expand the *same* matching logic to EVERY spacecraft
        sc_states_geo_eclip = []  # will be (num_sc, 6), ordered by spacecraft ID (1..N)
        sc_boresights = []
        for jdx, sc in enumerate(formation.spacecraft, start=1):
            # position of spacecraft j at the same detection instant idx0
            desired_pos_j_km = sc.matched_trajectory[idx0, :] * (config['AU_TO_M'] / 1000.0)

            # match to orbit table just like above (per spacecraft j)
            dists_j = np.linalg.norm(possible_positions - desired_pos_j_km, axis=1)
            idx_match_j = int(np.argmin(dists_j))

            # read GEO_EME state at that matched row
            geo_eme_state_j = formation.orbit.loc[
                formation.orbit.index[idx_match_j], eme_cols
            ].to_numpy(dtype=float)

            geo_eclip_state_detect_j = eme_to_ecliptic_batch(geo_eme_state_j)

            sc_states_geo_eclip.append(geo_eclip_state_detect_j)
            sc_boresights.append(sc.boresight)

        all_sc_geo_eclip = np.vstack(sc_states_geo_eclip)  # (num_sc, 6)
        all_sc_geo_eclip_list.append(all_sc_geo_eclip)

        all_sc_boresights = np.vstack(sc_boresights)  # (num_sc, 3)
        all_sc_boresights_list.append(all_sc_boresights)

    # 7) Write the detecting s/c info back into the DataFrame (no chained assignment)
    out_df.loc[:, 'detecting_sc_lpf_orbit_index'] = closest_indices
    out_df.loc[:, 'sc_epoch'] = sc_epochs

    det_geo_eclip = np.vstack(det_geo_eclip_states)
    out_df.loc[:, ['GEO_ECLIP_X_(km)', 'GEO_ECLIP_Y_(km)', 'GEO_ECLIP_Z_(km)',
                   'GEO_ECLIP_Vx_(km/s)', 'GEO_ECLIP_Vy_(km/s)', 'GEO_ECLIP_Vz_(km/s)']] = det_geo_eclip

    # Return: augmented DF, per-row all-s/c GEO_EME arrays, and detecting IDs
    return out_df, all_sc_geo_eclip_list, detecting_ids, all_sc_boresights_list


def ms_to_aud(states):
    # Argument parser to get the config file path
    parser = argparse.ArgumentParser(description="Run the spacecraft simulation")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    state_out = np.copy(states)
    state_out[:3] /= (config['AU_TO_M'])
    state_out[3:] /= (config['AU_TO_M'] / config['SECONDS_PER_DAY'])

    return state_out


def plot_fov_projection_geo(geo_boresight, spacecraft_pos, asteroid_pos, fov):
    """
    Visualizes the spacecraft's field of view (FOV) projection along the boresight at the asteroid's distance.

    Parameters:
        spacecraft: the spacecraft object
        asteroid: the asteroid object
            """
    # Convert FOV from degrees to radians
    fov_rad = np.radians(np.sqrt(fov))

    # Compute distance to asteroid
    sc_to_ast = np.array(asteroid_pos) - np.array(spacecraft_pos)
    ast_distance = np.linalg.norm(sc_to_ast)

    # Find the FOV projection center (along boresight at asteroid's distance)
    fov_center = np.array(spacecraft_pos) + ast_distance * geo_boresight

    # Define perpendicular vectors for FOV plane (orthogonal to boresight)
    up = np.array([0, 0, 1]) if abs(geo_boresight[2]) < 0.9 else np.array([1, 0, 0])  # Avoid collinear vector
    right = np.cross(geo_boresight, up)
    new_right = right / np.linalg.norm(right)
    up = np.cross(new_right, geo_boresight)  # Recompute true "up" vector

    # Compute FOV half-width at this distance
    fov_half_width = np.tan(fov_rad / 2) * ast_distance

    # Compute the 4 corners of the FOV projection at the asteroid's distance
    # order is [-1,-1], [1, -1], [-1, 1], [1, 1]
    fov_corners = []
    for dy in [-1, 1]:
        for dx in [-1, 1]:
            corner = fov_center + dx * fov_half_width * right + dy * fov_half_width * up
            fov_corners.append(corner)

    fov_corners[1], fov_corners[2], fov_corners[3] = fov_corners[2], fov_corners[3], fov_corners[1]
    fov_corners.append(fov_corners[0])

    return fov_corners


def plot_fov_projection(spacecraft, asteroid, index):
    """
    Visualizes the spacecraft's field of view (FOV) projection along the boresight at the asteroid's distance.

    Parameters:
        spacecraft: the spacecraft object
        asteroid: the asteroid object
            """
    # Convert FOV from degrees to radians
    fov_rad = np.radians(np.sqrt(spacecraft.fov))

    spacecraft_pos = spacecraft.get_spacecraft_pos(index)
    asteroid_pos = asteroid.get_asteroid_pos(index)

    # Compute distance to asteroid
    sc_to_ast = np.array(asteroid_pos) - np.array(spacecraft_pos)
    ast_distance = np.linalg.norm(sc_to_ast)

    # Find the FOV projection center (along boresight at asteroid's distance)
    fov_center = np.array(spacecraft_pos) + ast_distance * spacecraft.boresight

    # Define perpendicular vectors for FOV plane (orthogonal to boresight)
    up = np.array([0, 0, 1]) if abs(spacecraft.boresight[2]) < 0.9 else np.array([1, 0, 0])  # Avoid collinear vector
    right = np.cross(spacecraft.boresight, up)
    new_right = right / np.linalg.norm(right)
    up = np.cross(new_right, spacecraft.boresight)  # Recompute true "up" vector

    # Compute FOV half-width at this distance
    fov_half_width = np.tan(fov_rad / 2) * ast_distance

    # Compute the 4 corners of the FOV projection at the asteroid's distance
    # order is [-1,-1], [1, -1], [-1, 1], [1, 1]
    fov_corners = []
    for dy in [-1, 1]:
        for dx in [-1, 1]:
            corner = fov_center + dx * fov_half_width * right + dy * fov_half_width * up
            fov_corners.append(corner)

    fov_corners[1], fov_corners[2], fov_corners[3] = fov_corners[2], fov_corners[3], fov_corners[1]
    fov_corners.append(fov_corners[0])

    return fov_corners


def parse_master_new_new_new(file_path):
    """
    function for obtaning master mimimoon data file, with parameters
    'Object id', 'H', 'D', 'Capture Date', 'Helio x at Capture', 'Helio y at Capture', 'Helio z at Capture',
    'Helio vx at Capture', 'Helio vy at Capture', 'Helio vz at Capture', 'Helio q at Capture', 'Helio e at Capture',
    'Helio i at Capture', 'Helio Omega at Capture', 'Helio omega at Capture', 'Helio M at Capture',
    'Geo x at Capture', 'Geo y at Capture', 'Geo z at Capture', 'Geo vx at Capture', 'Geo vy at Capture',
    'Geo vz at Capture', 'Geo q at Capture', 'Geo e at Capture', 'Geo i at Capture', 'Geo Omega at Capture',
    'Geo omega at Capture', 'Geo M at Capture', 'Moon (Helio) x at Capture', 'Moon (Helio) y at Capture',
    'Moon (Helio) z at Capture', 'Moon (Helio) vx at Capture', 'Moon (Helio) vy at Capture',
    'Moon (Helio) vz at Capture', 'Capture Duration', 'Spec. En. Duration', '3 Hill Duration', 'Number of Rev',
    '1 Hill Duration', 'Min. Distance', 'Release Date', 'Helio x at Release', 'Helio y at Release',
    'Helio z at Release', 'Helio vx at Release', 'Helio vy at Release', 'Helio vz at Release', 'Helio q at Release',
    'Helio e at Release', 'Helio i at Release', 'Helio Omega at Release', 'Helio omega at Release',
    'Helio M at Release', 'Geo x at Release', 'Geo y at Release', 'Geo z at Release', 'Geo vx at Release',
    'Geo vy at Release', 'Geo vz at Release', 'Geo q at Release', 'Geo e at Release', 'Geo i at Release',
    'Geo Omega at Release', 'Geo omega at Release', 'Geo M at Release', 'Moon (Helio) x at Release',
     'Moon (Helio) y at Release', 'Moon (Helio) z at Release', 'Moon (Helio) vx at Release',
     'Moon (Helio) vy at Release', 'Moon (Helio) vz at Release', 'Retrograde', 'Became Minimoon', 'Max. Distance',
     'Capture Index', 'Release Index', 'X at Earth Hill', 'Y at Earth Hill', 'Z at Earth Hill', 'Taxonomy', 'STC'
     "EMS Duration", "Periapsides in EMS", "Periapsides in 3 Hill", "Periapsides in 2 Hill", "Periapsides in 1 Hill",
    "STC Start", "STC Start Index", "STC End", "STC End Index", "Helio x at EMS", "Helio y at EMS", "Helio z at EMS",
     "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS", "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
    "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)", "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
     "Moon x at EMS (Helio)", "Moon y at EMS (Helio)", "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
      "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", 'Entry Date to EMS', 'Entry to EMS Index',
     'Exit Date to EMS', 'Exit Index to EMS' "Dimensional Jacobi" "Non-Dimensional Jacobi" Alpha_I Beta_I Theta_M
     "Minimum Energy", "Peri-EM-L2", "Average Geo z", "Average Geo vz", "Winding Difference"
    :return:
    """
    master_data = pd.read_csv(file_path, sep=",", header=0, names=['Object id', 'H', 'D', 'Capture Date',
                                                                   'Helio x at Capture', 'Helio y at Capture',
                                                                   'Helio z at Capture', 'Helio vx at Capture',
                                                                   'Helio vy at Capture', 'Helio vz at Capture',
                                                                   'Helio q at Capture', 'Helio e at Capture',
                                                                   'Helio i at Capture', 'Helio Omega at Capture',
                                                                   'Helio omega at Capture', 'Helio M at Capture',
                                                                   'Geo x at Capture', 'Geo y at Capture',
                                                                   'Geo z at Capture', 'Geo vx at Capture',
                                                                   'Geo vy at Capture', 'Geo vz at Capture',
                                                                   'Geo q at Capture', 'Geo e at Capture',
                                                                   'Geo i at Capture', 'Geo Omega at Capture',
                                                                   'Geo omega at Capture', 'Geo M at Capture',
                                                                   'Moon (Helio) x at Capture',
                                                                   'Moon (Helio) y at Capture',
                                                                   'Moon (Helio) z at Capture',
                                                                   'Moon (Helio) vx at Capture',
                                                                   'Moon (Helio) vy at Capture',
                                                                   'Moon (Helio) vz at Capture',
                                                                   'Capture Duration', 'Spec. En. Duration',
                                                                   '3 Hill Duration', 'Number of Rev',
                                                                   '1 Hill Duration', 'Min. Distance',
                                                                   'Release Date', 'Helio x at Release',
                                                                   'Helio y at Release', 'Helio z at Release',
                                                                   'Helio vx at Release', 'Helio vy at Release',
                                                                   'Helio vz at Release', 'Helio q at Release',
                                                                   'Helio e at Release', 'Helio i at Release',
                                                                   'Helio Omega at Release',
                                                                   'Helio omega at Release',
                                                                   'Helio M at Release', 'Geo x at Release',
                                                                   'Geo y at Release', 'Geo z at Release',
                                                                   'Geo vx at Release', 'Geo vy at Release',
                                                                   'Geo vz at Release', 'Geo q at Release',
                                                                   'Geo e at Release', 'Geo i at Release',
                                                                   'Geo Omega at Release',
                                                                   'Geo omega at Release', 'Geo M at Release',
                                                                   'Moon (Helio) x at Release',
                                                                   'Moon (Helio) y at Release',
                                                                   'Moon (Helio) z at Release',
                                                                   'Moon (Helio) vx at Release',
                                                                   'Moon (Helio) vy at Release',
                                                                   'Moon (Helio) vz at Release', 'Retrograde',
                                                                   'Became Minimoon', 'Max. Distance',
                                                                   'Capture Index',
                                                                   'Release Index', 'X at Earth Hill',
                                                                   'Y at Earth Hill',
                                                                   'Z at Earth Hill', 'Taxonomy', 'STC',
                                                                   "EMS Duration",
                                                                   "Periapsides in EMS", "Periapsides in 3 Hill",
                                                                   "Periapsides in 2 Hill", "Periapsides in 1 Hill",
                                                                   "STC Start", "STC Start Index", "STC End",
                                                                   "STC End Index",
                                                                   "Helio x at EMS", "Helio y at EMS",
                                                                   "Helio z at EMS",
                                                                   "Helio vx at EMS", "Helio vy at EMS",
                                                                   "Helio vz at EMS",
                                                                   "Earth x at EMS (Helio)",
                                                                   "Earth y at EMS (Helio)",
                                                                   "Earth z at EMS (Helio)",
                                                                   "Earth vx at EMS (Helio)",
                                                                   "Earth vy at EMS (Helio)",
                                                                   "Earth vz at EMS (Helio)",
                                                                   "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
                                                                   "Moon z at EMS (Helio)",
                                                                   "Moon vx at EMS (Helio)",
                                                                   "Moon vy at EMS (Helio)",
                                                                   "Moon vz at EMS (Helio)",
                                                                   'Entry Date to EMS', 'Entry to EMS Index',
                                                                   'Exit Date to EMS', 'Exit Index to EMS',
                                                                   "Dimensional Jacobi", "Non-Dimensional Jacobi",
                                                                   'Alpha_I',
                                                                   'Beta_I', 'Theta_M', "Minimum Energy",
                                                                   "Peri-EM-L2", "Average Geo z", "Average Geo vz",
                                                                   "Winding Difference", "Min_SunEarthL1_V",
                                                                   "Min_SunEarthL1_V_index"])

    return master_data


def count_files_in_folder(folder_path):
    num_files = sum(
        1 for entry in os.scandir(folder_path) if entry.is_file()
    )
    return num_files


def get_all_files(folder_path, filetype='csv'):
    assert filetype in ['csv', 'parquet'], "filetype must be 'csv' or 'parquet'"

    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:

            if file.endswith(f'.{filetype}'):
                file_paths.append(os.path.join(root, file))
    return file_paths


def get_all_files_run_number(folder_path, filetype='csv', run_number=None):
    assert filetype in ['csv', 'parquet'], "filetype must be 'csv' or 'parquet'"

    file_paths = []
    run_str = f"run_{run_number}_" if run_number is not None else None

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(f'.{filetype}'):
                if run_str is None or run_str in file:
                    file_paths.append(os.path.join(root, file))

    return file_paths


def get_files_per_folder(parent_folder, filetype):
    subfolders = sorted([
        os.path.join(parent_folder, d)
        for d in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, d))
    ])

    all_files = []
    for folder in subfolders:
        files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and f.endswith('.' + filetype)
        ])
        all_files.append(files)

    return all_files


def eme_to_ecliptic_batch(state_vectors_eme):
    """
    Transforms a batch of full state vectors from EME J2000 to Ecliptic J2000.

    Parameters:
    - state_vectors_eme (numpy array): 6 x N array representing N state vectors
      in EME J2000 (each row: [x, y, z, vx, vy, vz]).

    Returns:
    - numpy array: Nx6 array representing N state vectors in Ecliptic J2000.
    """
    # Obliquity of the ecliptic at J2000 (in degrees)
    epsilon = 23.439281  # Mean obliquity of the ecliptic at J2000 epoch
    epsilon_rad = np.radians(epsilon)

    # Rotation matrix about the x-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(epsilon_rad), np.sin(epsilon_rad)],
        [0, -np.sin(epsilon_rad), np.cos(epsilon_rad)]
    ])

    # Split into position and velocity
    positions = state_vectors_eme[0:3]
    velocities = state_vectors_eme[3:6]

    # Rotate both
    pos_ecliptic = rotation_matrix @ positions
    vel_ecliptic = velocities @ rotation_matrix.T

    # Concatenate position and velocity back
    state_vectors_ecliptic = np.hstack((pos_ecliptic, vel_ecliptic))

    return state_vectors_ecliptic


def eclip_to_sun_earth_corotating_batch_n_body_integrator_output(states, earth_states):
    """
    Converts a batch of position and velocity state vectors from heliocentric ECLIPJ2000
    to the Earth-centered Sun-Earth co-rotating frame (X toward Sun, Z along orbital angular momentum).

    Parameters:
    - states: (M, 6, N) array of states in heliocentric ECLIPJ2000, for M objects and N timesteps
    - earth_states: (6, N) array of Earth state vectors in the same frame at each timestep

    Returns:
    - states_corotating: (M, 6, N) array of transformed states in the SECR frame
    """

    earth_positions = earth_states[:3, :].T

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angles = np.arctan2(earth_positions[:, 1], earth_positions[:, 0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angles = np.cos(-angles)
    sin_angles = np.sin(-angles)

    # Construct rotation matrices (shape: Nx3x3)
    rotation_matrices = np.zeros((len(angles), 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # No rotation in the Z direction

    object_i_pos = states[:3, :].T + earth_positions

    rotation_matrices = np.asarray(rotation_matrices, dtype=np.float64)
    relative_positions = np.asarray(object_i_pos, dtype=np.float64)

    # Apply the rotation to transform positions
    position_corotating = np.einsum("nij,nj->ni", rotation_matrices, relative_positions)

    return position_corotating


def eclip_to_sun_earth_corotating_batch(minimoon_df):
    """
    Converts a batch of positions and velocities from the heliocentric ECLIPJ2000 frame
    to the Sun-Earth co-rotating frame.

    Parameters:
    - positions_eclip (numpy array): Nx3 array of positions in ECLIPJ2000 (AU).
    - et_times (numpy array): N-element array of ephemeris times.

    Returns:
    - positions_corotating (numpy array): Nx3 array of positions in Sun-Earth co-rotating frame (AU).
    """

    moon_pos = minimoon_df.loc[:, ["Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)"]].to_numpy()
    earth_positions = minimoon_df.loc[:, ["Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)"]].to_numpy()

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angles = np.arctan2(earth_positions[:, 1], earth_positions[:, 0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angles = np.cos(-angles)
    sin_angles = np.sin(-angles)

    # Construct rotation matrices (shape: Nx3x3)
    rotation_matrices = np.zeros((len(angles), 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # No rotation in the Z direction

    # Apply the rotation to transform positions
    positions_corotating = np.einsum("nij,nj->ni", rotation_matrices, moon_pos - earth_positions)

    old_file_convention = positions_corotating.copy()
    old_file_convention[:, :2] *= -1

    minimoon_df[['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']] = old_file_convention
    file_path = '/media/aeromec/Seagate Desktop Drive/minimoon_files_oorb/' + str(
        minimoon_df['Object id'].iloc[0]) + '.csv'
    minimoon_df.to_csv(file_path, sep=' ', header=True, index=False)

    return positions_corotating


def eclip_to_sun_earth_corotating_batch_full_original_asteroid(states, earth_states):
    """
    Converts a batch of positions and velocities from the heliocentric ECLIPJ2000 frame
    to the Sun-Earth co-rotating frame.

    Parameters:
    - positions_eclip (numpy array): Nx3 array of positions in ECLIPJ2000 (AU).
    - et_times (numpy array): N-element array of ephemeris times.

    Returns:
    - positions_corotating (numpy array): Nx3 array of positions in Sun-Earth co-rotating frame (AU).
    """

    h_r_E = earth_states[:3, :].T  # (N, 3)
    h_v_E = earth_states[3:, :].T  # (N, 3)
    h_r_o = states[:3, :].T
    h_v_o = states[3:, :].T

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angles = np.arctan2(h_r_E[:, 1], h_r_E[:, 0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angles = np.cos(-angles)
    sin_angles = np.sin(-angles)

    # Construct rotation matrices (shape: Nx3x3)
    rotation_matrices = np.zeros((len(angles), 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # No rotation in the Z direction

    # Apply the rotation to transform positions
    positions_corotating = np.einsum("nij,nj->ni", rotation_matrices, h_r_o - h_r_E)

    return positions_corotating.T


def sun_earth_corotating_to_helio_eclip_batch_full(states_corotating, earth_states):
    """
    Converts a batch of position and velocity state vectors from the Earth-centered
    Sun-Earth co-rotating frame (SECR) to heliocentric ECLIPJ2000, accounting for the Coriolis term.

    Parameters:
    - states_corotating: (6, N) array of states in the SECR frame, where M is the number of objects
      and N is the number of timesteps. The first 3 rows represent positions, and the last 3 rows represent velocities.
    - earth_states: (6, N) array of Earth state vectors in the heliocentric ECLIPJ2000 frame at each timestep.
      The first 3 rows represent Earth's position, and the last 3 rows represent Earth's velocity.

    Returns:
    - states_heliocentric: (6, N) array of transformed states in heliocentric ECLIPJ2000 frame.
      The first 3 rows represent positions, and the last 3 rows represent velocities.
    """

    _, N = states_corotating.shape

    # Extract Earth's position and velocity from heliocentric ECLIPJ2000
    h_r_E = earth_states[:3, :].T  # (N, 3)
    h_v_E = earth_states[3:, :].T  # (N, 3)
    E_r_o_prime = states_corotating[:3, :].T
    E_v_o_prime = states_corotating[3:, :].T

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angles = np.arctan2(-h_r_E[:, 1], -h_r_E[:, 0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Construct rotation matrices (shape: Nx3x3), Z axis is fixed along ecliptic north
    rotation_matrices = np.zeros((N, 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # Z remains unchanged (ecliptic north)

    states_heliocentric = np.zeros_like(states_corotating)

    h_rel_o = np.einsum('nij,nj->ni', rotation_matrices, E_r_o_prime)  # now co-rotating frame
    h_r_o = h_rel_o + h_r_E

    # Angular velocity vector assuming uniform circular motion in ecliptic plane
    h_omega_mag = np.linalg.norm(np.cross(h_r_E, h_v_E), axis=1) / (np.linalg.norm(h_r_E, axis=1) ** 2)  # (N,)
    h_omega = np.zeros((N, 3))
    h_omega[:, 2] = h_omega_mag  # Only z-component for ecliptic plane rotation

    E_omega = np.einsum('nij,nj->ni', rotation_matrices.transpose(0, 2, 1), h_omega)
    v_rot = np.cross(E_omega, E_r_o_prime)  # correct: using rotated position
    v_rel_rot = E_v_o_prime + v_rot
    h_rel_v = np.einsum('nij,nj->ni', rotation_matrices, v_rel_rot)
    h_v_o = h_rel_v + h_v_E

    states_heliocentric[:3, :] = h_r_o.T
    states_heliocentric[3:, :] = h_v_o.T

    return states_heliocentric


def sun_earth_corotating_to_helio_eclip_single(state_corotating, earth_state):
    """
    Converts a batch of position and velocity state vectors from the Earth-centered
    Sun-Earth co-rotating frame (SECR) to heliocentric ECLIPJ2000, accounting for the Coriolis term.

    Parameters:
    - states_corotating: (1, N) array of states in the SECR frame, where M is the number of objects
      and N is the number of timesteps. The first 3 rows represent positions, and the last 3 rows represent velocities.
    - earth_states: (1, N) array of Earth state vectors in the heliocentric ECLIPJ2000 frame at each timestep.
      The first 3 rows represent Earth's position, and the last 3 rows represent Earth's velocity.

    Returns:
    - states_heliocentric: (1, N) array of transformed states in heliocentric ECLIPJ2000 frame.
      The first 3 rows represent positions, and the last 3 rows represent velocities.
    """

    # Extract Earth's position and velocity from heliocentric ECLIPJ2000
    h_r_E = earth_state[:3]  # ( 3)
    h_v_E = earth_state[3:]  # (3)
    E_r_o_prime = state_corotating[:3]
    E_v_o_prime = state_corotating[3:]

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angle = np.arctan2(-h_r_E[1], -h_r_E[0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Construct rotation matrices (shape: Nx3x3), Z axis is fixed along ecliptic north
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, 0] = cos_angle
    rotation_matrix[0, 1] = -sin_angle
    rotation_matrix[1, 0] = sin_angle
    rotation_matrix[1, 1] = cos_angle
    rotation_matrix[2, 2] = 1  # Z remains unchanged (ecliptic north)

    state_heliocentric = np.zeros_like(state_corotating)

    h_rel_o = rotation_matrix @ E_r_o_prime  # now co-rotating frame
    h_r_o = h_rel_o + h_r_E

    # Angular velocity vector assuming uniform circular motion in ecliptic plane
    h_omega_mag = np.linalg.norm(np.cross(h_r_E, h_v_E)) / (np.linalg.norm(h_r_E) ** 2)  # (N,)
    h_omega = np.zeros(3, )
    h_omega[2] = h_omega_mag  # Only z-component for ecliptic plane rotation

    E_omega = rotation_matrix.T @ h_omega
    v_rot = np.cross(E_omega, E_r_o_prime)  # correct: using rotated position
    v_rel_rot = E_v_o_prime + v_rot
    h_rel_v = rotation_matrix @ v_rel_rot
    h_v_o = h_rel_v + h_v_E

    state_heliocentric[:3] = h_r_o
    state_heliocentric[3:] = h_v_o

    return state_heliocentric


def ecliptic_to_eme_batch(state_vectors_ecliptic):
    """
    Transform state vector(s) from Ecliptic J2000 to EME (Equatorial) J2000.

    Accepts:
      - shape (6,)      -> returns (6,)
      - shape (6, N)    -> returns (6, N)
      - shape (N, 6)    -> returns (N, 6)

    State ordering: [x, y, z, vx, vy, vz]
    """

    X = np.asarray(state_vectors_ecliptic, dtype=float)

    # Remember input shape style
    if X.ndim == 1:
        if X.shape != (6,):
            raise ValueError(f"Expected shape (6,), got {X.shape}")
        X2 = X.reshape(6, 1)  # (6,1)
        out_style = "vec"
    elif X.ndim == 2:
        if X.shape[0] == 6:
            X2 = X  # (6,N)
            out_style = "6xN"
        elif X.shape[1] == 6:
            X2 = X.T  # (6,N)
            out_style = "Nx6"
        else:
            raise ValueError(f"Expected (6,N) or (N,6), got {X.shape}")
    else:
        raise ValueError(f"Expected 1D or 2D array, got ndim={X.ndim}")

    # J2000 mean obliquity (deg)
    epsilon_deg = 23.439281
    eps = np.deg2rad(epsilon_deg)

    # Rotation about +x by +eps (Ecliptic -> Equatorial/EME)
    c, s = np.cos(eps), np.sin(eps)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ])

    pos = X2[0:3, :]
    vel = X2[3:6, :]

    pos_eme = R @ pos
    vel_eme = R @ vel

    Y2 = np.vstack((pos_eme, vel_eme))  # (6,N)

    # Return in the same shape style as input
    if out_style == "vec":
        return Y2[:, 0]
    elif out_style == "6xN":
        return Y2
    else:  # "Nx6"
        return Y2.T


def helio_eclip_to_geo_eme_batch(states_helio_eclip, earth_helio_eclip):
    """
    Convert heliocentric-ecliptic state(s) to geocentric-EME state(s).

    Accepts:
      states_helio_eclip: (6,), (6,N), or (N,6)
      earth_helio_eclip : (6,) or same batch shape as states

    Returns in same shape style as states_helio_eclip.
    """
    S = np.asarray(states_helio_eclip, dtype=float)
    E = np.asarray(earth_helio_eclip, dtype=float)

    # Normalize shapes into (6,N) for subtraction
    def to_6xN(A, name):
        if A.ndim == 1:
            if A.shape != (6,):
                raise ValueError(f"{name}: expected (6,), got {A.shape}")
            return A.reshape(6, 1), "vec"
        if A.ndim == 2:
            if A.shape[0] == 6:
                return A, "6xN"
            if A.shape[1] == 6:
                return A.T, "Nx6"
        raise ValueError(f"{name}: expected (6,), (6,N) or (N,6), got {A.shape}")

    S2, style = to_6xN(S, "states_helio_eclip")

    # Earth can be (6,) or batch
    if E.ndim == 1 and E.shape == (6,):
        E2 = E.reshape(6, 1)  # will broadcast across N
    else:
        E2, _ = to_6xN(E, "earth_helio_eclip")
        if E2.shape[1] not in (1, S2.shape[1]):
            raise ValueError(f"earth batch length {E2.shape[1]} doesn't match states {S2.shape[1]}")

    states_geo_eclip = S2 - E2  # broadcasting ok
    states_geo_eme = ecliptic_to_eme_batch(states_geo_eclip)  # returns (6,N)

    # Return in original style of states input
    if style == "vec":
        return states_geo_eme[:, 0]
    elif style == "6xN":
        return states_geo_eme
    else:  # "Nx6"
        return states_geo_eme.T


def helio_eclip_from_geo_eme(eme_vectors, earth_helio_state):
    ###########################
    # convert geo eme to geo elcip
    ###########################

    # Obliquity of the ecliptic at J2000 (in degrees)
    epsilon = 23.439281  # Mean obliquity of the ecliptic at J2000 epoch

    # Convert epsilon to radians
    epsilon_rad = np.radians(epsilon)

    # Rotation matrix for transformation about the x-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(epsilon_rad), np.sin(epsilon_rad)],
        [0, -np.sin(epsilon_rad), np.cos(epsilon_rad)]
    ])

    # Apply the rotation to each vector using matrix multiplication
    ecliptic_positions = np.dot(eme_vectors[:3], rotation_matrix.T)
    ecliptic_velocities = np.dot(eme_vectors[3:], rotation_matrix.T)

    ##############################
    # convert geo eclip to helio
    ##########################
    helio_eclip_position = ecliptic_positions + earth_helio_state[:3]
    helio_eclip_velocities = ecliptic_velocities + earth_helio_state[3:]

    return np.hstack((helio_eclip_position, helio_eclip_velocities))


def ecliptic_to_eme_single_posvel(state_vectors_ecliptic):
    """
    Transforms a batch of full state vectors from Ecliptic J2000 to EME J2000.

    Parameters:
    - state_vectors_ecliptic (numpy array): 6xN array representing N state vectors
      in Ecliptic J2000 (rows: [x, y, z, vx, vy, vz]).

    Returns:
    - numpy array: 6xN array representing N state vectors in EME J2000.
    """

    # Obliquity of the ecliptic at J2000 (in degrees)
    epsilon = 23.439281
    epsilon_rad = np.radians(epsilon)

    # Rotation matrix about the x-axis (−epsilon for Ecliptic to EME)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(-epsilon_rad), np.sin(-epsilon_rad)],
        [0, -np.sin(-epsilon_rad), np.cos(-epsilon_rad)]
    ])

    # Separate position and velocity (each 3xN)
    pos = state_vectors_ecliptic[0:3]
    vel = state_vectors_ecliptic[3:6]

    # Apply rotation
    pos_eme = R @ pos
    vel_eme = R @ vel

    # Stack back into 6×N
    state_vectors_eme = np.stack((pos_eme, vel_eme)).reshape(-1)
    return state_vectors_eme


def _visible_dir(config):
    num_sc = int(config['num_spacecraft'])
    root = os.path.abspath(config['visible_files_folder'])
    return os.path.join(root, f"spacecraft_{num_sc}")


def _iod_dir(config):
    num_sc = int(config['num_spacecraft'])
    root = os.path.abspath(config['IOD_folder_path'])
    return os.path.join(root, f"spacecraft_{num_sc}")


def _non_hidden_entries(path):
    return [e for e in os.scandir(path) if not e.name.startswith('.')]


def _source_basenames_in_visible(vis_dir, save_format):
    # Prefer your util if available; otherwise glob by format(s)
    try:
        files = get_all_files(vis_dir, save_format)
    except Exception:
        files = []
        if save_format in ('csv', 'both'):
            files += glob.glob(os.path.join(vis_dir, '*.csv'))
        if save_format in ('parquet', 'both'):
            files += glob.glob(os.path.join(vis_dir, '*.parquet'))
    files = sorted(files)
    bases = [os.path.splitext(os.path.basename(p))[0] for p in files]
    return bases
