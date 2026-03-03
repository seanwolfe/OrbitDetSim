import yaml
import os
import pandas as pd
from Asteroid import Asteroid
from Formation import Formation
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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


# -------------------------
# Geometry helpers (3D)
# -------------------------
def _normalize_rows(X, eps=1e-15):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def orthonormal_basis(u):
    """
    Build (e1,e2) orthonormal basis spanning the plane perpendicular to unit vector u.
    """
    u = np.asarray(u, dtype=float)
    u = u / (np.linalg.norm(u) + 1e-15)

    # Pick a vector not parallel to u
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, a)
    e1 = e1 / (np.linalg.norm(e1) + 1e-15)
    e2 = np.cross(u, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-15)
    return e1, e2


def cone_boundary_circle(apex, u, theta_h, length, n_circle=60):
    """
    Return circle points (n_circle,3) at the cone cap and the cap center.
    """
    apex = np.asarray(apex, dtype=float).reshape(3,)
    u = np.asarray(u, dtype=float).reshape(3,)
    u = u / (np.linalg.norm(u) + 1e-15)

    e1, e2 = orthonormal_basis(u)
    center = apex + length * u
    radius = length * np.tan(float(theta_h))

    phi = np.linspace(0, 2*np.pi, int(n_circle), endpoint=True)
    circle = center + radius * (np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :])
    return circle, center


def plot_fov_cone(ax, apex, u, theta_h, length,
                  n_rays=12, n_circle=80,
                  alpha=0.18, lw=1.2, color="tab:blue", label=None):
    """
    Draw a cone as:
      - a handful of boundary rays
      - a circle connecting their endpoints
    """
    apex = np.asarray(apex, dtype=float).reshape(3,)
    u = np.asarray(u, dtype=float).reshape(3,)
    u = u / (np.linalg.norm(u) + 1e-15)

    circle, _ = cone_boundary_circle(apex, u, theta_h, length, n_circle=n_circle)

    # Circle (cap)
    ax.plot(circle[:, 0], circle[:, 1], circle[:, 2],
            lw=lw, alpha=alpha, color=color, label=label)

    # Boundary rays
    n_rays = int(max(3, n_rays))
    idx = np.linspace(0, circle.shape[0] - 1, n_rays, dtype=int)
    for k in idx:
        p = circle[k]
        ax.plot([apex[0], p[0]], [apex[1], p[1]], [apex[2], p[2]],
                lw=lw, alpha=alpha, color=color)



def set_axes_equal_3d(ax):
    """
    Make 3D axes have equal scale (matplotlib doesn't do this automatically).
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


# -------------------------
# Coverage in 3D (point cloud)
# -------------------------
def coverage_count_3d(points_xyz, agents_xyz, u_opt_agents_xyz, theta_h_rad, max_range):
    """
    points_xyz: (N,3)
    agents_xyz: (M,3)
    u_opt_agents_xyz: (M,3) unit-ish boresight vectors in same frame
    Returns:
      cov: (N,) integer coverage count (how many cones contain each point)
    """
    P = np.asarray(points_xyz, dtype=float)
    A = np.asarray(agents_xyz, dtype=float)
    U = np.asarray(u_opt_agents_xyz, dtype=float)

    M = A.shape[0]
    N = P.shape[0]

    U = _normalize_rows(U)
    theta_h = float(theta_h_rad)
    cos_th = np.cos(theta_h)
    max_range = float(max_range)

    cov = np.zeros(N, dtype=int)

    # Vectorized per-agent loop (good balance of clarity + speed for typical sizes)
    for i in range(M):
        v = P - A[i]  # (N,3)
        d = np.linalg.norm(v, axis=1)  # (N,)
        in_range = d <= max_range

        # Avoid division by zero for points at the apex
        vhat = v / (d[:, None] + 1e-15)
        cosang = vhat @ U[i]  # (N,)

        in_cone = in_range & (cosang >= cos_th)
        cov += in_cone.astype(int)

    return cov


# -------------------------
# Uncertainty ellipsoid (3D)
# -------------------------
def mahalanobis_ellipsoid_mesh(mu, Sigma, d_mahal=3.0, n_u=40, n_v=20):
    """
    Mesh points on the ellipsoid:
      (x-mu)^T Sigma^{-1} (x-mu) = d_mahal^2

    Returns X,Y,Z each (n_v, n_u) suitable for ax.plot_wireframe / ax.plot_surface.
    """
    mu = np.asarray(mu, dtype=float).reshape(3,)
    Sigma = np.asarray(Sigma, dtype=float).reshape(3, 3)

    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-12)

    # Unit sphere parameterization
    u = np.linspace(0, 2*np.pi, int(n_u))
    v = np.linspace(0, np.pi, int(n_v))
    uu, vv = np.meshgrid(u, v)

    xs = np.cos(uu) * np.sin(vv)
    ys = np.sin(uu) * np.sin(vv)
    zs = np.cos(vv)

    sphere = np.stack([xs, ys, zs], axis=0).reshape(3, -1)  # (3, n_u*n_v)

    axes_lengths = float(d_mahal) * np.sqrt(eigvals)  # (3,)
    ell_local = (np.diag(axes_lengths) @ sphere)       # (3, npts)
    ell_world = (eigvecs @ ell_local).T + mu           # (npts, 3)

    X = ell_world[:, 0].reshape(vv.shape)
    Y = ell_world[:, 1].reshape(vv.shape)
    Z = ell_world[:, 2].reshape(vv.shape)
    return X, Y, Z


# -------------------------
# EMS sphere mesh
# -------------------------
def sphere_mesh(center, radius, n_u=60, n_v=30):
    center = np.asarray(center, dtype=float).reshape(3,)
    R = float(radius)

    u = np.linspace(0, 2*np.pi, int(n_u))
    v = np.linspace(0, np.pi, int(n_v))
    uu, vv = np.meshgrid(u, v)

    X = center[0] + R * np.cos(uu) * np.sin(vv)
    Y = center[1] + R * np.sin(uu) * np.sin(vv)
    Z = center[2] + R * np.cos(vv)
    return X, Y, Z


def plot_theta_phi_over_history(history, M, *, deg=True, title=None):
    """
    Plot theta (solid) and phi (dashed) vs iteration index for each agent.

    - One figure
    - One color per agent
    - Solid line = theta_i
    - Dashed line = phi_i
    - Handles mixed history entries:
        * original mode: entry["x"] is length 2*M
        * fixed mode: entry["x"] may contain NaNs for fixed agent, and entry may include:
            - entry["idx_fix"] and entry["use_fixed_agent"]
            - entry["x_free"] containing only free agents in order of increasing id (common)
    Assumption for fixed mode:
        free agents are all agents except idx_fix, ordered ascending.
    """

    # build per-agent time series (lists, then arrays)
    thetas = [[] for _ in range(M)]
    phis   = [[] for _ in range(M)]

    for t, entry in enumerate(history):
        use_fixed = bool(entry.get("use_fixed_agent", False))
        idx_fix = entry.get("idx_fix", None)

        x = np.asarray(entry.get("x", []), dtype=float).ravel()
        x_free = entry.get("x_free", None)
        if x_free is not None:
            x_free = np.asarray(x_free, dtype=float).ravel()

        if (not use_fixed) or (idx_fix is None):
            # original mode: expect full x length 2*M
            if x.size < 2*M:
                raise ValueError(f"history[{t}] has x of size {x.size}, expected >= {2*M}")
            for i in range(M):
                thetas[i].append(float(x[2*i]))
                phis[i].append(float(x[2*i + 1]))
            continue

        # fixed mode
        if not (0 <= int(idx_fix) < M):
            raise ValueError(f"history[{t}] idx_fix out of range: {idx_fix}")
        idx_fix = int(idx_fix)

        # try to read theta/phi from full x if present and not NaN
        for i in range(M):
            th = np.nan
            ph = np.nan
            if x.size >= 2*M:
                th = float(x[2*i])
                ph = float(x[2*i + 1])
            thetas[i].append(th)
            phis[i].append(ph)

        # if x had NaNs for fixed-mode, fill from x_free for free agents
        if x_free is not None:
            free_idx = [i for i in range(M) if i != idx_fix]  # assumed ordering
            if x_free.size != 2*len(free_idx):
                # if it doesn't match, we still keep whatever was in x
                continue

            for k, i in enumerate(free_idx):
                thetas[i][-1] = float(x_free[2*k])
                phis[i][-1]   = float(x_free[2*k + 1])

        # if fixed agent is NaN in x, keep as NaN (or you can hold it constant if you want)

    # convert to arrays
    thetas = [np.asarray(v, dtype=float) for v in thetas]
    phis   = [np.asarray(v, dtype=float) for v in phis]

    # radians -> degrees if requested
    if deg:
        thetas = [np.rad2deg(v) for v in thetas]
        phis   = [np.rad2deg(v) for v in phis]
        ylab = "Angle (deg)"
    else:
        ylab = "Angle (rad)"

    # plot
    plt.figure(figsize=(10, 5.5))
    for i in range(M):
        plt.plot(thetas[i], linestyle='-',  linewidth=2.0, label=f"θ A{i}")
        plt.plot(phis[i],   linestyle='--', linewidth=2.0, label=f"φ A{i}")

    plt.xlabel("History step")
    plt.ylabel(ylab)
    plt.title(title if title is not None else "Theta (solid) and Phi (dashed) over history")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    return thetas, phis



# -------------------------
# Main plotting function (3D)
# -------------------------
def plot_od_scenario_3d_new(
        *,
        # epoch/meta
        t_label=None,

        # agents
        agents_xyz,            # (M,3)
        u_opt_agents_xyz,      # (M,3) boresight unit-ish vectors in this frame
        theta_h_rad,           # scalar half-angle of cone
        ray_length=10.0,

        # optional: current boresight vectors for dotted line + slew text
        u_curr_agents_xyz=None,   # (M,3)
        boresight_line_len=3.0,

        # optimizer initial pointing vectors (solid long black lines)
        u_init_agents_xyz=None,   # (M,3)
        init_boresight_line_len=None,   # scalar; defaults to ray_length if None

        # optional: orbit tracks / quasi-halo projections
        agent_orbit_tracks_xyz=None,  # list length M; each entry (K,3)

        # coverage extents + sampling
        xlim=None, ylim=None, zlim=None,
        Nx=40, Ny=40, Nz=20,
        max_points_for_scatter=120_000,

        # target uncertainty + truth (current instant)
        target_mean_xyz=None,      # (3,)
        target_cov_xyz=None,       # (3,3)
        d_mahal=2.0,
        true_target_xyz=None,      # (3,)

        # trajectories (full history / window) to plot as lines
        target_mean_traj_xyz=None,   # (K,3)
        true_target_traj_xyz=None,   # (K,3)
        true_target_traj_xyz_2=None,


        # EMS sphere
        ems_center_xyz=None,       # (3,)
        ems_radius=None,           # scalar

        # styling toggles
        show_coverage=True,
        show_uncertainty=True,
        show_truth=True,
        show_ems=True,
        show_fov_cones=True,
        title=None,

        # declutter + styling knobs
        label_fontsize=9,
        label_offset_px=10,
        slew_label_offset_px=16,
        fill_alpha=0.10,
        sparse_wire=True,
        fov_n_rays=8,
        fov_n_circle=48,
        coverage_dot_size=1.0,
        coverage_dot_alpha=0.75,

        # initial pointing style
        init_boresight_lw=1.5,
        init_boresight_alpha=0.95,

        # ------------------------------------------------------------
        # NEW: slew history boresight visualization (color progression)
        # ------------------------------------------------------------
        # Provide either:
        #   - dict[int, array_like(K,3)]  mapping spacecraft id -> boresight unit vectors over time
        #   - list[tuple[int, array_like(K,3)]]  [(sc_id, U_hist), ...]
        slew_history=None,
        slew_history_line_len=None,      # defaults to boresight_line_len
        slew_history_lw=1.8,
        slew_history_alpha=0.85,
        slew_history_cmap="viridis",     # any matplotlib cmap name
        slew_history_every=1,            # plot every Nth history sample
        slew_history_colorbar=True,
        slew_history_colorbar_label="Slew history step",
        slew_history_norm_mode="per_agent",  # "per_agent" or "global"
):
    """
    Pure 3D visualization: everything is passed in (positions, vectors, cov, tracks).
    Returns (fig, ax).
    """

    def _as_3(x):
        if x is None:
            return None
        return np.asarray(x, dtype=float).reshape(3,)

    def _annotate3d(ax_, text, xyz, *, dx=0, dy=0, fontsize=9, color="black"):
        x, y, z = map(float, xyz)
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax_.get_proj())
        ax_.annotate(
            text, xy=(x2, y2), xytext=(dx, dy), textcoords="offset points",
            ha="left", va="bottom", fontsize=fontsize, color=color,
        )

    def _sphere_surface(ax_, c, R, *, alpha=0.10, n_u=26, n_v=13, color="orange", label=None):
        u = np.linspace(0, 2*np.pi, n_u)
        v = np.linspace(0, np.pi, n_v)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        X = c[0] + R*np.cos(uu)*np.sin(vv)
        Y = c[1] + R*np.sin(uu)*np.sin(vv)
        Z = c[2] + R*np.cos(vv)
        ax_.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=alpha, color=color)
        if label is not None:
            proxy = plt.Line2D([0], [0], linestyle="none", marker="s", color=color, alpha=alpha, label=label)
            ax_.add_artist(ax_.legend(handles=[proxy], loc="upper right"))

    def _plot_two_orth_circles(ax_, c, A3x3, *, n=240, color="orange", lw=1.0, alpha=0.6, label=None):
        t = np.linspace(0, 2*np.pi, n)
        q_xy = np.stack([np.cos(t), np.sin(t), 0*t], axis=1)
        q_xz = np.stack([np.cos(t), 0*t, np.sin(t)], axis=1)
        P1 = c.reshape(1, 3) + (q_xy @ A3x3.T)
        P2 = c.reshape(1, 3) + (q_xz @ A3x3.T)
        ax_.plot(P1[:, 0], P1[:, 1], P1[:, 2], color=color, lw=lw, alpha=alpha, label=label)
        ax_.plot(P2[:, 0], P2[:, 1], P2[:, 2], color=color, lw=lw, alpha=alpha)

    def _ellipsoid_surface(ax_, mu, P, d, *, alpha=0.10, n_u=26, n_v=13, color="tab:red"):
        w, V = np.linalg.eigh(P)
        w = np.clip(w, 0.0, None)
        Aell = (V * (np.sqrt(w) * float(d))) @ V.T  # 3x3

        u = np.linspace(0, 2*np.pi, n_u)
        v = np.linspace(0, np.pi, n_v)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        qx = np.cos(uu) * np.sin(vv)
        qy = np.sin(uu) * np.sin(vv)
        qz = np.cos(vv)
        Q = np.stack([qx, qy, qz], axis=-1)

        X = mu[0] + (Aell[0, 0]*Q[..., 0] + Aell[0, 1]*Q[..., 1] + Aell[0, 2]*Q[..., 2])
        Y = mu[1] + (Aell[1, 0]*Q[..., 0] + Aell[1, 1]*Q[..., 1] + Aell[1, 2]*Q[..., 2])
        Z = mu[2] + (Aell[2, 0]*Q[..., 0] + Aell[2, 1]*Q[..., 1] + Aell[2, 2]*Q[..., 2])

        ax_.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=alpha, color=color)
        return Aell

    def _normalize_rows(U, eps=1e-12):
        U = np.asarray(U, dtype=float)
        n = np.linalg.norm(U, axis=1, keepdims=True)
        n = np.maximum(n, eps)
        return U / n

    # ---- inputs ----
    A = np.asarray(agents_xyz, dtype=float)
    M = A.shape[0]
    Uopt = _normalize_rows(np.asarray(u_opt_agents_xyz, dtype=float).reshape(M, 3))

    # ---- bounds ----
    if (xlim is None) or (ylim is None) or (zlim is None):
        xs = [A[:, 0]]
        ys = [A[:, 1]]
        zs = [A[:, 2]]

        mu = _as_3(target_mean_xyz)
        tr = _as_3(true_target_xyz)
        ec = _as_3(ems_center_xyz)

        if mu is not None:
            xs.append([mu[0]]); ys.append([mu[1]]); zs.append([mu[2]])
        if tr is not None:
            xs.append([tr[0]]); ys.append([tr[1]]); zs.append([tr[2]])
        if ec is not None:
            xs.append([ec[0]]); ys.append([ec[1]]); zs.append([ec[2]])

        if target_mean_traj_xyz is not None:
            Tm = np.asarray(target_mean_traj_xyz, dtype=float)
            if Tm.ndim == 2 and Tm.shape[1] == 3 and Tm.size > 0:
                xs.append(Tm[:, 0]); ys.append(Tm[:, 1]); zs.append(Tm[:, 2])
        if true_target_traj_xyz is not None:
            Tt = np.asarray(true_target_traj_xyz, dtype=float)
            if Tt.ndim == 2 and Tt.shape[1] == 3 and Tt.size > 0:
                xs.append(Tt[:, 0]); ys.append(Tt[:, 1]); zs.append(Tt[:, 2])
        if true_target_traj_xyz_2 is not None:
            Tt2 = np.asarray(true_target_traj_xyz_2, dtype=float)
            if Tt2.ndim == 2 and Tt2.shape[1] == 3 and Tt2.size > 0:
                xs.append(Tt2[:, 0]); ys.append(Tt2[:, 1]); zs.append(Tt2[:, 2])

        xall = np.concatenate([np.asarray(v).ravel() for v in xs])
        yall = np.concatenate([np.asarray(v).ravel() for v in ys])
        zall = np.concatenate([np.asarray(v).ravel() for v in zs])

        pad = 2.0
        if xlim is None: xlim = (float(np.min(xall) - pad), float(np.max(xall) + pad))
        if ylim is None: ylim = (float(np.min(yall) - pad), float(np.max(yall) + pad))
        if zlim is None: zlim = (float(np.min(zall) - pad), float(np.max(zall) + pad))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # ---------------------------------------------------------------------
    # NEW: Slew history boresight lines w/ color progression + optional cbar
    # ---------------------------------------------------------------------
    if slew_history is not None:
        # normalize input to dict[int, (K,3)]
        if isinstance(slew_history, dict):
            hist_map = dict(slew_history)
        else:
            # assume iterable of (id, hist)
            hist_map = {int(k): v for (k, v) in slew_history}

        Lhist = float(boresight_line_len) if (slew_history_line_len is None) else float(slew_history_line_len)
        cmap = cm.get_cmap(slew_history_cmap)

        # decide normalization mode
        if slew_history_norm_mode not in ("per_agent", "global"):
            raise ValueError("slew_history_norm_mode must be 'per_agent' or 'global'")

        if slew_history_norm_mode == "global":
            # global max steps across provided agents (after decimation)
            maxK = 0
            for sc_id, Uh in hist_map.items():
                Uh = np.asarray(Uh, dtype=float)
                if Uh.ndim != 2 or Uh.shape[1] != 3:
                    raise ValueError(f"slew_history[{sc_id}] must be (K,3)")
                K = Uh.shape[0]
                Kd = (K + (slew_history_every - 1)) // max(int(slew_history_every), 1)
                maxK = max(maxK, Kd)
            norm_global = mcolors.Normalize(vmin=0, vmax=max(0, maxK - 1))

        # plot each requested agent history
        for sc_id, Uh in hist_map.items():
            sc_id = int(sc_id)
            if not (0 <= sc_id < M):
                # silently skip invalid IDs (or raise if you prefer)
                continue

            Uh = np.asarray(Uh, dtype=float)
            if Uh.ndim != 2 or Uh.shape[1] != 3:
                raise ValueError(f"slew_history[{sc_id}] must be (K,3)")

            # decimate
            step = max(int(slew_history_every), 1)
            Uh = Uh[::step]
            if Uh.shape[0] == 0:
                continue

            Uh = _normalize_rows(Uh)

            if slew_history_norm_mode == "per_agent":
                norm = mcolors.Normalize(vmin=0, vmax=max(0, Uh.shape[0] - 1))
            else:
                norm = norm_global

            # draw segments/lines from spacecraft position
            p0 = A[sc_id]
            for k in range(Uh.shape[0]):
                col = cmap(norm(k))
                pend = p0 + Uh[k] * Lhist
                ax.plot([p0[0], pend[0]], [p0[1], pend[1]], [p0[2], pend[2]],
                        color=col, lw=float(slew_history_lw), alpha=float(slew_history_alpha),
                        label=f"Slew history A{sc_id}" if (k == 0) else None)

        # one colorbar for the whole figure (optional)
        if bool(slew_history_colorbar):
            sm = cm.ScalarMappable(
                norm=(norm_global if slew_history_norm_mode == "global"
                      else mcolors.Normalize(vmin=0, vmax=1)),
                cmap=cmap
            )
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
            cbar.set_label(str(slew_history_colorbar_label))

            if slew_history_norm_mode == "per_agent":
                # explain that color is "early -> late" per agent
                cbar.set_ticks([0.0, 1.0])
                cbar.set_ticklabels(["early", "late"])

    # ---- coverage point cloud (ONLY double + 3+) ----
    if show_coverage:
        if (target_mean_xyz is not None) and (target_cov_xyz is not None):
            mu = np.asarray(target_mean_xyz, dtype=float).reshape(3,)
            P = np.asarray(target_cov_xyz, dtype=float).reshape(3, 3)
            w, V = np.linalg.eigh(P)
            w = np.clip(w, 0.0, None)
            Aell = (V * (np.sqrt(w) * float(d_mahal))) @ V.T
            half = np.sum(np.abs(Aell), axis=1)

            xlim_cov = (float(mu[0] - half[0]), float(mu[0] + half[0]))
            ylim_cov = (float(mu[1] - half[1]), float(mu[1] + half[1]))
            zlim_cov = (float(mu[2] - half[2]), float(mu[2] + half[2]))

            xg = np.linspace(xlim_cov[0], xlim_cov[1], int(Nx))
            yg = np.linspace(ylim_cov[0], ylim_cov[1], int(Ny))
            zg = np.linspace(zlim_cov[0], zlim_cov[1], int(Nz))
        else:
            xg = np.linspace(xlim[0], xlim[1], int(Nx))
            yg = np.linspace(ylim[0], ylim[1], int(Ny))
            zg = np.linspace(zlim[0], zlim[1], int(Nz))

        XX, YY, ZZ = np.meshgrid(xg, yg, zg, indexing="xy")
        grid = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

        if grid.shape[0] > int(max_points_for_scatter):
            rng = np.random.default_rng(0)
            idx = rng.choice(grid.shape[0], size=int(max_points_for_scatter), replace=False)
            grid = grid[idx]

        cov = coverage_count_3d(grid, A, Uopt, theta_h_rad, max_range=ray_length)

        mask = cov >= 2
        if np.any(mask):
            Pcov = grid[mask]
            Ccov = cov[mask]
            m2 = (Ccov == 2)
            m3 = (Ccov >= 3)

            if np.any(m2):
                ax.scatter(Pcov[m2, 0], Pcov[m2, 1], Pcov[m2, 2],
                           s=float(coverage_dot_size), alpha=float(coverage_dot_alpha),
                           color="tab:green", label="Double coverage")
            if np.any(m3):
                ax.scatter(Pcov[m3, 0], Pcov[m3, 1], Pcov[m3, 2],
                           s=float(coverage_dot_size), alpha=float(coverage_dot_alpha),
                           color="red", label="3+ coverage")

    # ---- orbit tracks ----
    if agent_orbit_tracks_xyz is not None:
        for i, trk in enumerate(agent_orbit_tracks_xyz):
            if trk is None:
                continue
            trk = np.asarray(trk, dtype=float)
            if trk.ndim != 2 or trk.shape[1] != 3:
                raise ValueError(f"agent_orbit_tracks_xyz[{i}] must be (K,3)")
            ax.plot(trk[:, 0], trk[:, 1], trk[:, 2], lw=1.2, alpha=0.7, color='blue',
                    label="Agent orbit" if i == 0 else None)

    # ---- trajectories (mean + true) ----
    if target_mean_traj_xyz is not None:
        Tm = np.asarray(target_mean_traj_xyz, dtype=float)
        if Tm.ndim != 2 or Tm.shape[1] != 3:
            raise ValueError("target_mean_traj_xyz must be (K,3)")
        ax.plot(Tm[:, 0], Tm[:, 1], Tm[:, 2], lw=1.6, alpha=0.8, color="tab:red",
                label="IOD mean trajectory")

    if true_target_traj_xyz is not None:
        Tt = np.asarray(true_target_traj_xyz, dtype=float)
        if Tt.ndim != 2 or Tt.shape[1] != 3:
            raise ValueError("true_target_traj_xyz must be (K,3)")
        ax.plot(Tt[:, 0], Tt[:, 1], Tt[:, 2], lw=1.6, alpha=0.8, color="green",
                label="True trajectory")
    if true_target_traj_xyz_2 is not None:
        Tt2 = np.asarray(true_target_traj_xyz_2, dtype=float)
        if Tt2.ndim != 2 or Tt2.shape[1] != 3:
            raise ValueError("true_target_traj_xyz_2 must be (K,3)")
        print("here")
        ax.plot(Tt2[:, 0], Tt2[:, 1], Tt2[:, 2], lw=1.6, alpha=0.8, color="green",
                label="True trajectory 2", linestyle='--')

    # ---- initial optimizer pointing (solid black lines) ----
    if u_init_agents_xyz is not None:
        Ui = np.asarray(u_init_agents_xyz, dtype=float)
        if Ui.shape != (M, 3):
            raise ValueError("u_init_agents_xyz must be (M,3) matching agents")
        Ui = _normalize_rows(Ui)
        Linit = float(ray_length) if (init_boresight_line_len is None) else float(init_boresight_line_len)

        for i in range(M):
            p_end = A[i] + Ui[i] * Linit
            ax.plot([A[i, 0], p_end[0]], [A[i, 1], p_end[1]], [A[i, 2], p_end[2]],
                    linestyle="-", color="black", lw=float(init_boresight_lw),
                    alpha=float(init_boresight_alpha),
                    label="Initial optimizer pointing" if i == 0 else None)

    # ---- FOV cones + agent markers + labels ----
    if show_fov_cones:
        for i in range(M):
            plot_fov_cone(ax, A[i], Uopt[i], theta_h_rad, float(ray_length),
                          n_rays=int(fov_n_rays), n_circle=int(fov_n_circle),
                          alpha=0.75, lw=1.0, color="tab:blue",
                          label="Agent FOV" if i == 0 else None)

            ax.scatter(A[i, 0], A[i, 1], A[i, 2], s=40, color="tab:blue",
                       label="Agent position" if i == 0 else None)

    if u_curr_agents_xyz is not None:
        Uc = np.asarray(u_curr_agents_xyz, dtype=float)
        if Uc.shape != (M, 3):
            raise ValueError("u_curr_agents_xyz must be (M,3) matching agents")
        Uc = _normalize_rows(Uc)

        for i in range(M):
            p_end = A[i] + Uc[i] * float(boresight_line_len)
            ax.plot([A[i, 0], p_end[0]], [A[i, 1], p_end[1]], [A[i, 2], p_end[2]],
                    linestyle=":", color="black", lw=1.2,
                    label="Initial boresight" if i == 0 else None)

            dot = float(np.clip(np.dot(Uc[i], Uopt[i]), -1.0, 1.0))
            slew_deg = float(np.degrees(np.arccos(dot)))

            dx = int(slew_label_offset_px * (1 if (i % 2 == 0) else -1))
            dy = int(slew_label_offset_px * (1 if ((i // 2) % 2 == 0) else -1))
            _annotate3d(ax, f"{slew_deg:.1f}°", p_end, dx=dx, dy=dy,
                        fontsize=label_fontsize, color="black")

    for i in range(M):
        dx = int(label_offset_px * (1 if (i % 2 == 0) else -1))
        dy = int(label_offset_px * (1 if ((i // 2) % 2 == 0) else -1))
        _annotate3d(ax, f"A{i}", A[i], dx=dx, dy=dy, fontsize=label_fontsize, color="tab:blue")

    # ---- uncertainty ellipsoid ----
    if show_uncertainty and (target_mean_xyz is not None) and (target_cov_xyz is not None):
        mu = np.asarray(target_mean_xyz, dtype=float).reshape(3,)
        P = np.asarray(target_cov_xyz, dtype=float).reshape(3, 3)
        ax.scatter(mu[0], mu[1], mu[2], marker="x", s=50, linewidths=2,
                   label="Target mean (current)", color="tab:red")
        Aell = _ellipsoid_surface(ax, mu, P, float(d_mahal), alpha=float(fill_alpha), color="tab:red")
        if sparse_wire:
            _plot_two_orth_circles(ax, mu, Aell, color="tab:red", lw=1.1, alpha=0.65,
                                   label="Uncertainty (2 wires)")

    # ---- true marker ----
    if show_truth and (true_target_xyz is not None):
        tr = np.asarray(true_target_xyz, dtype=float).reshape(3,)
        ax.scatter(tr[0], tr[1], tr[2], s=40, color="green", marker="o",
                   label="True position (current)")

    # ---- EMS sphere ----
    if show_ems and (ems_center_xyz is not None) and (ems_radius is not None) and (float(ems_radius) > 0):
        c = np.asarray(ems_center_xyz, dtype=float).reshape(3,)
        R = float(ems_radius)

        _sphere_surface(ax, c, R, alpha=float(fill_alpha), n_u=26, n_v=13, color="orange")

        if sparse_wire:
            A_sphere = np.eye(3) * R
            _plot_two_orth_circles(ax, c, A_sphere, color="orange", lw=1.1, alpha=0.65,
                                   label="EMS (2 wires)")
        else:
            Xs, Ys, Zs = sphere_mesh(c, R, n_u=60, n_v=30)
            ax.plot_wireframe(Xs, Ys, Zs, rstride=2, cstride=2, linewidth=0.7, alpha=0.35,
                              label="EMS sphere", color="orange")

    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    if title is None:
        title = f"Scenario @ {t_label}" if t_label is not None else "Scenario (3D)"
    ax.set_title(title)

    ax.grid(alpha=0.25)
    set_axes_equal_3d(ax)
    # ax.legend(loc="upper right")
    return fig, ax






def plot_od_scenario_3d(
        *,
        # epoch/meta
        t_label=None,

        # agents
        agents_xyz,            # (M,3)
        u_opt_agents_xyz,      # (M,3) boresight unit-ish vectors in this frame
        theta_h_rad,           # scalar half-angle of cone
        ray_length=10.0,

        # optional: current boresight vectors for dotted line + slew text
        u_curr_agents_xyz=None,   # (M,3)
        boresight_line_len=3.0,

        # optional: orbit tracks / quasi-halo projections
        agent_orbit_tracks_xyz=None,  # list length M; each entry (K,3)

        # coverage extents + sampling
        xlim=None, ylim=None, zlim=None,
        Nx=40, Ny=40, Nz=20,
        max_points_for_scatter=120_000,

        # target uncertainty + truth
        target_mean_xyz=None,      # (3,)
        target_cov_xyz=None,       # (3,3)
        d_mahal=2.0,
        true_target_xyz=None,      # (3,)

        # EMS sphere
        ems_center_xyz=None,       # (3,)
        ems_radius=None,           # scalar

        # styling toggles
        show_coverage=True,
        show_uncertainty=True,
        show_truth=True,
        show_ems=True,
        title=None,
):
    """
    Pure 3D visualization: everything is passed in (positions, vectors, cov, tracks).
    Returns (fig, ax).
    """
    A = np.asarray(agents_xyz, dtype=float)
    M = A.shape[0]
    Uopt = _normalize_rows(np.asarray(u_opt_agents_xyz, dtype=float).reshape(M, 3))

    # Determine plot bounds if not provided
    if (xlim is None) or (ylim is None) or (zlim is None):
        xs = [A[:, 0]]
        ys = [A[:, 1]]
        zs = [A[:, 2]]

        if target_mean_xyz is not None:
            mu = np.asarray(target_mean_xyz, dtype=float).reshape(3,)
            xs.append([mu[0]]); ys.append([mu[1]]); zs.append([mu[2]])
        if true_target_xyz is not None:
            tr = np.asarray(true_target_xyz, dtype=float).reshape(3,)
            xs.append([tr[0]]); ys.append([tr[1]]); zs.append([tr[2]])
        if ems_center_xyz is not None:
            ec = np.asarray(ems_center_xyz, dtype=float).reshape(3,)
            xs.append([ec[0]]); ys.append([ec[1]]); zs.append([ec[2]])

        xall = np.concatenate([np.asarray(v).ravel() for v in xs])
        yall = np.concatenate([np.asarray(v).ravel() for v in ys])
        zall = np.concatenate([np.asarray(v).ravel() for v in zs])

        pad = 2.0
        if xlim is None:
            xlim = (np.min(xall) - pad, np.max(xall) + pad)
        if ylim is None:
            ylim = (np.min(yall) - pad, np.max(yall) + pad)
        if zlim is None:
            zlim = (np.min(zall) - pad, np.max(zall) + pad)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Coverage point cloud
    # Coverage point cloud (dense, discrete colors: 1=blue, 2=green, 3+=red)
    if show_coverage:
        # Make it denser by default (override via function args if you want)
        Nx_i, Ny_i, Nz_i = int(Nx), int(Ny), int(Nz)

        xg = np.linspace(xlim[0], xlim[1], Nx_i)
        yg = np.linspace(ylim[0], ylim[1], Ny_i)
        zg = np.linspace(zlim[0], zlim[1], Nz_i)
        XX, YY, ZZ = np.meshgrid(xg, yg, zg, indexing="xy")
        grid = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

        # Optional downsample cap (raise this a lot for denser clouds)
        if grid.shape[0] > int(max_points_for_scatter):
            rng = np.random.default_rng(0)
            idx = rng.choice(grid.shape[0], size=int(max_points_for_scatter), replace=False)
            grid = grid[idx]

        cov = coverage_count_3d(grid, A, Uopt, theta_h_rad, max_range=ray_length)

        # Keep only covered points
        mask_any = cov > 0
        if np.any(mask_any):
            Pcov = grid[mask_any]
            Ccov = cov[mask_any]

            m1 = (Ccov == 1)
            m2 = (Ccov == 2)
            m3 = (Ccov >= 3)

            # Much smaller dots + dense cloud
            dot_size = 1.0  # try 0.5 if you want even smaller
            dot_alpha = 0.35  # lower if it looks too saturated

            # Scatter by class (colors fixed)
            h1 = None
            h2 = None
            h3 = None

            if np.any(m1):
                h1 = ax.scatter(Pcov[m1, 0], Pcov[m1, 1], Pcov[m1, 2],
                                s=dot_size, alpha=dot_alpha, color="tab:blue",
                                label="Single coverage")
            if np.any(m2):
                h2 = ax.scatter(Pcov[m2, 0], Pcov[m2, 1], Pcov[m2, 2],
                                s=dot_size, alpha=dot_alpha, color="tab:green",
                                label="Double coverage")
            if np.any(m3):
                h3 = ax.scatter(Pcov[m3, 0], Pcov[m3, 1], Pcov[m3, 2],
                                s=dot_size, alpha=dot_alpha, color="red",
                                label="3+ coverage")

    # Orbit tracks / projections (3D)
    if agent_orbit_tracks_xyz is not None:
        for i, trk in enumerate(agent_orbit_tracks_xyz):
            if trk is None:
                continue
            trk = np.asarray(trk, dtype=float)
            if trk.ndim != 2 or trk.shape[1] != 3:
                raise ValueError(f"agent_orbit_tracks_xyz[{i}] must be (K,3)")
            ax.plot(trk[:, 0], trk[:, 1], trk[:, 2], lw=1.2, alpha=0.7,
                    label="Agent orbit" if i == 0 else None)

    # FOV cones + agent markers
    for i in range(M):
        plot_fov_cone(ax, A[i], Uopt[i], theta_h_rad, float(ray_length),
                      n_rays=12, n_circle=80, alpha=0.18, lw=1.0,
                      color="tab:blue",
                      label="Agent FOV" if i == 0 else None)

        ax.scatter(A[i, 0], A[i, 1], A[i, 2],
                   s=40, color="tab:blue",
                   label="Agent position" if i == 0 else None)

        ax.text(A[i, 0], A[i, 1], A[i, 2], f"  A{i}", fontsize=9)

    # Optional: current boresight dotted line + slew annotation
    if u_curr_agents_xyz is not None:
        Uc = np.asarray(u_curr_agents_xyz, dtype=float)
        if Uc.shape != (M, 3):
            raise ValueError("u_curr_agents_xyz must be (M,3) matching agents")
        Uc = _normalize_rows(Uc)

        for i in range(M):
            p_end = A[i] + Uc[i] * float(boresight_line_len)
            ax.plot([A[i, 0], p_end[0]], [A[i, 1], p_end[1]], [A[i, 2], p_end[2]],
                    linestyle=":", color="black", lw=1.2,
                    label="Initial boresight" if i == 0 else None)

            dot = float(np.clip(np.dot(Uc[i], Uopt[i]), -1.0, 1.0))
            slew_deg = float(np.degrees(np.arccos(dot)))

            ax.text(A[i, 0], A[i, 1], A[i, 2] + 0.6, f"{slew_deg:.1f}°",
                    fontsize=9, color="black")

    # Target mean + uncertainty ellipsoid (wireframe)
    if show_uncertainty and (target_mean_xyz is not None) and (target_cov_xyz is not None):
        mu = np.asarray(target_mean_xyz, dtype=float).reshape(3,)
        P = np.asarray(target_cov_xyz, dtype=float).reshape(3, 3)

        ax.scatter(mu[0], mu[1], mu[2], marker="x", s=50, linewidths=2, label="Target mean", color='red')

        X, Y, Z = mahalanobis_ellipsoid_mesh(mu, P, d_mahal=float(d_mahal), n_u=44, n_v=22)
        ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2,
                          linewidth=0.9, alpha=0.45,
                          color="tab:red",
                          label="Uncertainty ellipsoid")

    # True target
    if show_truth and (true_target_xyz is not None):
        tr = np.asarray(true_target_xyz, dtype=float).reshape(3, )
        ax.scatter(tr[0], tr[1], tr[2],
                   s=40, color="green", marker="o",
                   label="True position")

    # EMS sphere
    if show_ems and (ems_center_xyz is not None) and (ems_radius is not None) and (float(ems_radius) > 0):
        c = np.asarray(ems_center_xyz, dtype=float).reshape(3,)
        R = float(ems_radius)

        Xs, Ys, Zs = sphere_mesh(c, R, n_u=60, n_v=30)
        ax.plot_wireframe(Xs, Ys, Zs, rstride=2, cstride=2, linewidth=0.7, alpha=0.35,
                          label="EMS sphere", color='orange')

    # Labels, limits, title
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    if title is None:
        title = f"Scenario @ {t_label}" if t_label is not None else "Scenario (3D)"
    ax.set_title(title)

    ax.grid(alpha=0.25)
    set_axes_equal_3d(ax)
    ax.legend(loc="upper right")

    return fig, ax


def clamp_point_into_fov_cone(point_xyz,
                              sc_pos_xyz,
                              boresight_u_xyz,
                              theta_h_rad,
                              max_range=None,
                              eps=1e-15):
    """
    Ensure point_xyz lies within the spacecraft FOV cone defined by:
      apex = sc_pos_xyz
      axis = boresight_u_xyz (need not be unit; we normalize)
      half-angle = theta_h_rad
      optional max_range (distance from apex)

    Returns:
      point_clamped_xyz, inside_bool, info_dict
    """
    p = np.asarray(point_xyz, dtype=float).reshape(3,)
    A = np.asarray(sc_pos_xyz, dtype=float).reshape(3,)
    u = np.asarray(boresight_u_xyz, dtype=float).reshape(3,)

    nu = np.linalg.norm(u)
    if nu < eps:
        raise ValueError("boresight_u_xyz has near-zero norm")
    u = u / nu

    v = p - A
    r = np.linalg.norm(v)
    if r < eps:
        # Point at apex -> treat as inside (degenerate)
        return p.copy(), True, {"reason": "at_apex"}

    h = float(np.dot(v, u))           # axial distance along cone axis
    w = v - h * u                     # perpendicular component
    s = float(np.linalg.norm(w))      # radial distance from axis

    # Must be in front of spacecraft: h >= 0
    # Must satisfy angle constraint: s <= h*tan(theta)
    tan_th = float(np.tan(theta_h_rad))
    cos_th = float(np.cos(theta_h_rad))
    inside = (h >= 0.0) and (s <= h * tan_th + 1e-12)

    # If inside angle but beyond max_range, clamp along same ray
    if inside and (max_range is not None) and (r > float(max_range)):
        p2 = A + (float(max_range) / r) * v
        return p2, False, {"reason": "range_clamp_only", "old_r": r, "new_r": float(max_range)}

    if inside:
        return p.copy(), True, {"reason": "inside"}

    # --- Outside: compute closest point in (solid) cone ---
    # If behind the apex, the closest point in the cone is the apex (or along axis at h=0)
    if h <= 0.0:
        p2 = A.copy()
        if max_range is not None:
            # apex already within any range
            pass
        return p2, False, {"reason": "behind_apex_clamp_to_apex"}

    # For closest point on the cone surface (same azimuth as w):
    if s > eps:
        w_hat = w / s
    else:
        # v almost on axis but not inside only happens if h<0, handled above
        # still provide a stable perpendicular direction
        tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        w_hat = np.cross(u, tmp)
        w_hat /= (np.linalg.norm(w_hat) + eps)

    # Minimize ||(h' u + (h' tanθ) w_hat) - (h u + s w_hat)||^2 over h' >= 0
    # Closed form:
    #   h' = cos^2θ (h + tanθ s)
    h_prime = (cos_th * cos_th) * (h + tan_th * s)
    h_prime = max(0.0, float(h_prime))

    v_prime = h_prime * u + (h_prime * tan_th) * w_hat
    p2 = A + v_prime

    # Optional max_range clamp (keep direction from apex to p2)
    if max_range is not None:
        r2 = float(np.linalg.norm(v_prime))
        L = float(max_range)
        if r2 > L and r2 > eps:
            p2 = A + (L / r2) * v_prime
            return p2, False, {"reason": "angle_and_range_clamp", "old_r": r, "new_r": L}

    return p2, False, {"reason": "angle_clamp", "old_r": r}


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
        target_mean_xy_traj=None,
        target_cov_xy=None,  # (2,2)
        d_mahal=2.0,
        true_target_xy=None,  # (2,)
        true_target_xy_traj=None,

        # EMS zone (2D cross-section you provide)
        ems_center_xy=None,  # (2,)
        ems_radius=None,  # scalar

        # styling toggles
        show_coverage=True,
        show_uncertainty=True,
        show_truth=True,
        show_ems=True,
        title=None,
        ax=None,
):
    """
    Pure visualization function: does not infer any states.
    Everything (positions, angles, mean/cov, tracks) is passed in.

    Returns (fig, ax).
    """
    # --- NEW: allow plotting into an existing axis ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure
        ax.cla()  # clear only this subplot

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


    # Coverage grid
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
            u_opt = np.array([np.cos(ang[i]), np.sin(ang[i])], dtype=float)
            dot = float(np.clip(np.dot(u, u_opt), -1.0, 1.0))
            slew_deg = float(np.degrees(np.arccos(dot)))

            ax.text(A[i, 0], A[i, 1] + 0.5, f"{slew_deg:.1f}°",
                    ha='center', va='bottom', fontsize=9, color='black')

    # Target mean + uncertainty ellipse
    unc_mean_sc = None
    ellipse_line = None
    unc_mean_traj = None
    if show_uncertainty and (target_mean_xy is not None) and (target_cov_xy is not None):
        mu = np.asarray(target_mean_xy, dtype=float).reshape(2, )
        P = np.asarray(target_cov_xy, dtype=float).reshape(2, 2)

        unc_mean_sc = ax.scatter(mu[0], mu[1], color='tab:red', s=80,
                                 marker='x', linewidths=2,
                                 label='Target mean')
        if target_mean_xy_traj is not None:
            unc_mean_traj, = ax.plot(target_mean_xy_traj[:, 0], target_mean_xy_traj[:, 1], color='tab:red',
                                    linewidth=2, label='Target predicted trajectory')
        ellipse_pts = mahalanobis_ellipse_points(mu, P, d_mahal=float(d_mahal), n=250)
        ellipse_line, = ax.plot(ellipse_pts[:, 0], ellipse_pts[:, 1],
                                color='tab:red', lw=2,
                                label='Uncertainty ellipse')
        ax.fill(ellipse_pts[:, 0], ellipse_pts[:, 1], color='tab:red', alpha=0.10)

    # True target position
    true_sc = None
    true_traj = None
    if show_truth and (true_target_xy is not None):
        tr = np.asarray(true_target_xy, dtype=float).reshape(2, )
        true_sc = ax.scatter(tr[0], tr[1], s=60, facecolors='none',
                             edgecolors='green', linewidths=2,
                             label='True position')
        if true_target_xy_traj is not None:
            true_traj, = ax.plot(true_target_xy_traj[:, 0], true_target_xy_traj[:, 1], color='tab:green',
                                linewidth=2, label='True trajetory')

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
    if unc_mean_traj is not None:
        handles.append(unc_mean_traj)
    if true_traj is not None:
        handles.append(true_traj)

    if show_coverage:
        single_cov_patch = Patch(facecolor=cmap_colors[1], alpha=0.25, label='Single coverage')
        double_cov_patch = Patch(facecolor=cmap_colors[2], alpha=0.25, label='Double coverage')
        triple_cov_patch = Patch(facecolor=cmap_colors[3], alpha=0.25, label='Triple+ coverage')
        handles.extend([single_cov_patch, double_cov_patch, triple_cov_patch])

    # ax.legend(handles=handles, loc='upper right')
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
        asteroid_state_helio = geo_eme_to_geo_eclip_generic(asteroid_state_geo) + earth_state

        # integrate s/c traj
        asteroid_integrated_states, asteroid_earth_states = nbody.integrate_n_body(asteroid_state_helio,
                                                                                   asteroid_epoch,
                                                                                   total_observation_window *
                                                                                   config['SECONDS_PER_DAY'],
                                                                                   config['time_between_frames'],
                                                                                   type="ASTEROID")  # integrator takes seconds

        asteroid_int_geo = (asteroid_integrated_states - asteroid_earth_states)
        asteroid_eme = geo_eclip_to_geo_eme_generic(asteroid_int_geo, layout="time")
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


def _cov_ellipse_2d(P2, n_std=1.0, n_pts=200):
    P2 = np.asarray(P2, dtype=float).reshape(2, 2)
    w, V = np.linalg.eigh(P2)
    w = np.maximum(w, 0.0)
    t = np.linspace(0.0, 2.0 * np.pi, n_pts)
    circle = np.vstack([np.cos(t), np.sin(t)])  # (2,n)
    A = V @ np.diag(np.sqrt(w)) * float(n_std)
    pts = A @ circle
    return pts[0], pts[1]


def plot_priors_positions_and_cov_2d(
    X_pred_km,
    P_pred_km2,
    *,
    sc_trajs_km=None,
    sc_trajs_km2=None,          # <-- NEW: second spacecraft set
    sc_mark_every=1,
    planes=("xy", "xz", "yz"),
    stride=5,
    n_std=1.0,
    show_path=True,
    show_sc_paths=True,
    title_prefix="Priors",
    equal_aspect=True,
):
    """
    Visualize mean position trajectory and covariance ellipses in 2D planes,
    optionally overlaid with spacecraft trajectories.

    Inputs:
      - X_pred_km: (K,6) or (6,) from ukf.propagate_priors(...)
      - P_pred_km2: (K,6,6) or (6,6) position covariance is top-left 3x3 (km^2)

      - sc_trajs_km: optional spacecraft trajectories (SET 1):
          * (K,M,6) or (K,M,3) or (M,6) or (M,3)
            - If (M,*) assumed static at one epoch (K=1)
            - If (K,M,*) time-varying; uses first 3 components as position

      - sc_trajs_km2: optional second spacecraft trajectories (SET 2), same shape rules as sc_trajs_km.
          Plotted with the SAME corresponding color as set 1, but with triangle markers.

      - sc_mark_every: plot markers for s/c every this many steps (for clarity)

    Plots:
      - One figure per plane (xy, xz, yz)
      - Mean asteroid trajectory (optional)
      - Covariance ellipses every `stride` steps at `n_std` sigma
      - Spacecraft trajectories (optional) including a second set with matching colors
    """
    X = np.asarray(X_pred_km, dtype=float)
    P = np.asarray(P_pred_km2, dtype=float)

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if P.ndim == 2:
        P = P.reshape(1, 6, 6)

    K = X.shape[0]
    if X.shape[1] < 3:
        raise ValueError(f"X_pred must have at least 3 components, got {X.shape}")
    if P.shape != (K, 6, 6):
        raise ValueError(f"P_pred must be (K,6,6) matching X_pred, got {P.shape}")

    def _normalize_sc_trajs(sc_trajs, K_expected, name="sc_trajs_km"):
        """Normalize spacecraft trajectories to (K, M, 3) or None."""
        if sc_trajs is None:
            return None

        sc = np.asarray(sc_trajs, dtype=float)

        if sc.ndim == 2:
            # (M,3) or (M,6) -> treat as static (K=1)
            if sc.shape[1] not in (3, 6):
                raise ValueError(f"{name} (2D) must be (M,3) or (M,6), got {sc.shape}")
            return sc[:, :3].reshape(1, sc.shape[0], 3)

        if sc.ndim == 3:
            # (K,M,3) or (K,M,6)
            if sc.shape[2] not in (3, 6):
                raise ValueError(f"{name} (3D) must be (K,M,3) or (K,M,6), got {sc.shape}")
            if sc.shape[0] != K_expected:
                raise ValueError(f"{name} K mismatch: got {sc.shape[0]}, expected {K_expected}")
            return sc[:, :, :3]

        raise ValueError(f"{name} must be 2D or 3D array, got ndim={sc.ndim}")

    # Normalize spacecraft trajectories
    sc_pos1 = _normalize_sc_trajs(sc_trajs_km, K, name="sc_trajs_km")
    sc_pos2 = _normalize_sc_trajs(sc_trajs_km2, K, name="sc_trajs_km2")

    # If both provided, enforce same number of spacecraft M (so "corresponding color" is well-defined)
    if sc_pos1 is not None and sc_pos2 is not None:
        if sc_pos1.shape[1] != sc_pos2.shape[1]:
            raise ValueError(
                f"sc_trajs_km and sc_trajs_km2 must have same M. "
                f"Got M1={sc_pos1.shape[1]} vs M2={sc_pos2.shape[1]}."
            )

    idx_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}

    for pl in planes:
        if pl not in idx_map:
            raise ValueError(f"Unknown plane '{pl}'. Use one of {list(idx_map.keys())}.")
        a, b = idx_map[pl]

        plt.figure()
        ax = plt.gca()

        # Spacecraft trajectories (set 1 + set 2 with matching colors)
        if show_sc_paths and (sc_pos1 is not None or sc_pos2 is not None):
            mark_every = max(int(sc_mark_every), 1)

            # Determine M across whichever is present
            if sc_pos1 is not None:
                K1, M, _ = sc_pos1.shape
            else:
                K2, M, _ = sc_pos2.shape

            for m in range(M):
                color_m = None

                # --- Set 1: default markers (circles) ---
                if sc_pos1 is not None:
                    K1, _, _ = sc_pos1.shape
                    if K1 == 1 and K > 1:
                        xs = sc_pos1[0, m, a]
                        ys = sc_pos1[0, m, b]
                        line1 = ax.plot([xs], [ys], marker="o", linestyle="None")[0]
                    else:
                        line1 = ax.plot(sc_pos1[:, m, a], sc_pos1[:, m, b])[0]
                        ax.plot(
                            sc_pos1[::mark_every, m, a],
                            sc_pos1[::mark_every, m, b],
                            marker="o",
                            linestyle="None",
                            color=line1.get_color(),
                        )
                    color_m = line1.get_color()

                # --- Set 2: same color, triangle markers ---
                if sc_pos2 is not None:
                    K2, _, _ = sc_pos2.shape
                    if K2 == 1 and K > 1:
                        xs = sc_pos2[0, m, a]
                        ys = sc_pos2[0, m, b]
                        ax.plot([xs], [ys], marker="^", linestyle="None", color=color_m)
                    else:
                        line2 = ax.plot(
                            sc_pos2[:, m, a],
                            sc_pos2[:, m, b],
                            color=color_m,
                        )[0]
                        # If set 1 wasn't present, color_m comes from set 2 line
                        if color_m is None:
                            color_m = line2.get_color()
                        ax.plot(
                            sc_pos2[::mark_every, m, a],
                            sc_pos2[::mark_every, m, b],
                            marker="^",
                            linestyle="None",
                            color=color_m,
                        )

        # Asteroid mean path
        if show_path and K > 1:
            ax.plot(X[:, a], X[:, b])

        # Covariance ellipses
        for k in range(0, K, max(int(stride), 1)):
            mu = X[k, :3]
            Ppos = P[k, :3, :3]
            P2 = Ppos[np.ix_([a, b], [a, b])]

            ex, ey = _cov_ellipse_2d(P2, n_std=float(n_std))
            ax.plot(mu[a] + ex, mu[b] + ey)

        ax.set_xlabel(f"{pl[0]} (km)")
        ax.set_ylabel(f"{pl[1]} (km)")
        ax.set_title(f"{title_prefix}: mean + {n_std}σ ellipses in {pl.upper()} plane")

        if equal_aspect:
            ax.set_aspect("equal", adjustable="datalim")

        ax.grid(True)

    plt.show()



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

    asteroid_pos_eme = geo_eclip_to_geo_eme_generic(asteroid_pos.T, layout='time').T
    moon_pos_eme = geo_eclip_to_geo_eme_generic(moon_pos.T, layout="time").T
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
            geo_boresight = geo_secr_to_geo_eclip_generic(boresight_vec, earth_state)
            geo_eme_boresight = geo_eclip_to_geo_eme_generic(geo_boresight)

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


def helio_eclip_to_geo_secr_generic(obj, earth, eps=1e-12, layout="auto",
                                    obj_hint=None, earth_hint=None):
    """
    Convert heliocentric ECLIPJ2000 position(s) or state(s) to Earth-centered
    Sun–Earth co-rotating (SECR) frame with +Z fixed to ecliptic north.

    STRICT RULES:
    - obj dim = 3 → earth may be 3 or 6 (earth velocity ignored)
    - obj dim = 6 → earth MUST be 6 (otherwise ValueError)

    Supported shapes (fallback, when no explicit hint):
      obj Position: (3,), (M,3), (N,3), (3,N), (M,N,3)
      obj State:    (6,), (M,6), (N,6), (6,N), (M,N,6)

      earth Position: (3,), (3,1), (1,3), (N,3), (3,N)
      earth State:    (6,), (6,1), (1,6), (N,6), (6,N)

    layout resolves ambiguity when obj is (K,dim) (fallback mode):
      - "batch": interpret as (M,dim) objects at one time (N=1)
      - "time" : interpret as (N,dim) time series for one object (M=1)
      - "auto" : infer from earth shape when possible, else default to "batch"

    NEW (explicit structure hints):
      You can explicitly define the axis order using tokens (any permutation):
        - 'batch'   : M axis (multiple objects / spacecraft)
        - 'time'    : N axis (time series)
        - 'position': dim=3
        - 'state'   : dim=6
        - 3 or 6    : dim override (optional)

      Examples (obj):
        obj_hint=('batch','position')     -> expects obj shape (M,3)
        obj_hint=('position','batch')     -> expects obj shape (3,M)
        obj_hint=('batch','state')        -> expects obj shape (M,6)
        obj_hint=('time','state')         -> expects obj shape (N,6)
        obj_hint=('state','time')         -> expects obj shape (6,N)
        obj_hint=('time','state','batch') -> expects obj shape (N,6,M)
        obj_hint=('batch','time','state') -> expects obj shape (M,N,6)
        obj_hint=('state','batch','time') -> expects obj shape (6,M,N)
        etc.

      Examples (earth) — IMPORTANT:
        In this function Earth is TIME-indexed only (no 'batch' axis supported).
        So earth_hint may include 'time' and 'position'/'state' (or 3/6) in any permutation.

    Returns: same layout as obj; if obj_hint was used, returns in that hinted axis order.
    """
    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    O = np.asarray(obj, dtype=float)
    E = np.asarray(earth, dtype=float)

    # -----------------------
    # Hint parsing: explicit axis order
    # -----------------------
    def _parse_struct_hint(h, name, allow_batch=True):
        """
        Returns (order, dim) where:
          order: list like ['batch','dim'] or ['time','dim','batch'] (axis order)
          dim: 3 or 6
        """
        if h is None:
            return None, None

        if isinstance(h, (tuple, list, set)):
            tokens = list(h)
        elif isinstance(h, str):
            s = h.strip()
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            tokens = [t.strip() for t in s.split(",") if t.strip()]
        else:
            raise ValueError(f"{name}_hint must be None, a tuple/list/set, or a string like '(time,state,batch)'")

        order = []
        dim = None

        def _add_axis(ax):
            if ax in order:
                raise ValueError(f"{name}_hint repeats axis '{ax}'. Got {h}.")
            order.append(ax)

        for t in tokens:
            if isinstance(t, (int, np.integer)):
                iv = int(t)
                if iv in (3, 6):
                    dim = iv
                    if "dim" not in order:
                        _add_axis("dim")
                else:
                    raise ValueError(f"{name}_hint invalid dim {t} (use 3 or 6)")
                continue

            ts = str(t).strip().lower()
            if ts == "batch":
                if not allow_batch:
                    raise ValueError(f"{name}_hint may not include 'batch' (earth cannot be batch-indexed here).")
                _add_axis("batch")
            elif ts == "time":
                _add_axis("time")
            elif ts in ("position", "pos"):
                if dim is not None and dim != 3:
                    raise ValueError(f"{name}_hint conflicts: both state(6) and position(3) implied.")
                dim = 3
                _add_axis("dim")
            elif ts in ("state", "st"):
                if dim is not None and dim != 6:
                    raise ValueError(f"{name}_hint conflicts: both position(3) and state(6) implied.")
                dim = 6
                _add_axis("dim")
            elif ts in ("3", "6"):
                dim = int(ts)
                if "dim" not in order:
                    _add_axis("dim")
            elif ts == "":
                continue
            else:
                raise ValueError(
                    f"{name}_hint token '{t}' unrecognized. Use 'batch','time','position','state',3,6."
                )

        if "dim" not in order:
            raise ValueError(f"{name}_hint must include 'position'/'state' (or 3/6). Got {h}.")
        if dim not in (3, 6):
            raise ValueError(f"{name}_hint must resolve dim to 3 or 6. Got {h}.")

        return order, dim

    obj_order_hint, obj_dim_hint = _parse_struct_hint(obj_hint, "obj", allow_batch=True)
    earth_order_hint, earth_dim_hint = _parse_struct_hint(earth_hint, "earth", allow_batch=False) if earth_hint is not None else (None, None)

    # -----------------------
    # Dim inference (fallback when no explicit hint)
    # -----------------------
    def infer_dim_fallback(A, name):
        if A.ndim == 1:
            if A.shape in [(3,), (6,)]:
                return A.shape[0]
            raise ValueError(f"{name} expected (3,) or (6,), got {A.shape}")

        if A.ndim == 2:
            r, c = A.shape
            r_is = r in (3, 6)
            c_is = c in (3, 6)

            if r_is and not c_is:
                return r
            if c_is and not r_is:
                return c
            if r_is and c_is:
                return 6 if (r == 6 or c == 6) else 3

        if A.ndim == 3 and A.shape[-1] in (3, 6):
            return A.shape[-1]

        raise ValueError(f"Could not infer dim for {name} with shape {A.shape}")

    obj_dim = obj_dim_hint if obj_dim_hint is not None else infer_dim_fallback(O, "obj")
    earth_dim = earth_dim_hint if earth_dim_hint is not None else infer_dim_fallback(E, "earth")

    # --- STRICT RULE ---
    if obj_dim == 6 and earth_dim == 3:
        raise ValueError(
            "Invalid input: obj is 6D state but earth is 3D position. "
            "Earth velocity is required to compute omega and the SECR velocity correction."
        )

    # -----------------------
    # Normalize obj to internal (M,N,obj_dim) and remember how to restore
    # -----------------------
    def _normalize_obj_with_hint(A, order, dim):
        """
        Strictly interpret obj using explicit axis order -> return (M,N,dim).
        Missing 'batch' or 'time' axes are treated as singleton.
        Also returns restore_info to put output back into the same hinted axis order.
        """
        if A.ndim == 1:
            if A.shape != (dim,):
                raise ValueError(f"obj expected ({dim},), got {A.shape}")
            A_int = A[None, None, :]
            restore = {"used_hint": True, "order": ["dim"], "orig_ndim": 1}
            return A_int, restore

        if A.ndim != len(order):
            raise ValueError(f"obj_hint implies {len(order)}D but obj has ndim={A.ndim}, shape={A.shape}")

        ax_dim = order.index("dim")
        ax_batch = order.index("batch") if "batch" in order else None
        ax_time  = order.index("time")  if "time"  in order else None

        if A.shape[ax_dim] != dim:
            raise ValueError(f"obj dim axis length {A.shape[ax_dim]} does not match hinted dim={dim}.")

        axes = []
        if ax_batch is not None:
            axes.append(ax_batch)
        if ax_time is not None:
            axes.append(ax_time)
        axes.append(ax_dim)

        A_perm = np.transpose(A, axes=axes)

        if ax_batch is not None and ax_time is not None:
            A_int = A_perm                  # (M,N,dim)
        elif ax_batch is not None and ax_time is None:
            A_int = A_perm[:, None, :]      # (M,1,dim)
        elif ax_batch is None and ax_time is not None:
            A_int = A_perm[None, :, :]      # (1,N,dim)
        else:
            A_int = A_perm[None, None, :]   # (1,1,dim)

        restore = {"used_hint": True, "order": order, "orig_ndim": A.ndim}
        return A_int, restore

    def _choose_mode_for_Kdim(K):
        # fallback for (K,dim) inputs without hints
        if layout in ("batch", "time"):
            return layout

        # layout == auto: infer from earth shape if possible
        earth_time_like = (
            E.ndim == 2 and (
                (E.shape[1] == earth_dim and E.shape[0] == K) or
                (E.shape[0] == earth_dim and E.shape[1] == K)
            )
        )
        return "time" if earth_time_like else "batch"

    if obj_order_hint is not None:
        O_int, obj_restore = _normalize_obj_with_hint(O, obj_order_hint, obj_dim)
        out_style = ("hinted", obj_restore)
    else:
        # ---- fallback normalization (your original behavior) ----
        if O.ndim == 1:
            if O.shape != (obj_dim,):
                raise ValueError(f"obj expected ({obj_dim},), got {O.shape}")
            O_int = O[None, None, :]
            out_style = ("single",)

        elif O.ndim == 2:
            if O.shape == (obj_dim, 1):
                O_int = O[:, 0][None, None, :]
                out_style = ("single",)

            elif O.shape == (1, obj_dim):
                O_int = O[0, :][None, None, :]
                out_style = ("single",)

            elif O.shape[0] == obj_dim and O.shape[1] != obj_dim:  # (dim,N)
                O_int = O.T[None, :, :]
                out_style = ("dimxN",)

            elif O.shape[1] == obj_dim and O.shape[0] != obj_dim:  # (K,dim)
                K = O.shape[0]
                mode = _choose_mode_for_Kdim(K)
                if mode == "batch":
                    O_int = O[:, None, :]
                    out_style = ("Mxdim",)
                else:
                    O_int = O[None, :, :]
                    out_style = ("Nxdim_time",)

            elif O.shape[0] == obj_dim and O.shape[1] == obj_dim:
                raise ValueError(
                    f"Ambiguous obj shape {O.shape}. Reshape explicitly or pass obj_hint."
                )
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

    # -----------------------
    # Normalize earth to (N,earth_dim) (time-indexed only)
    # -----------------------
    def _normalize_earth_to_Nd(Earr, N, dim, order_hint=None):
        """
        Earth is time-indexed only: output is (N,dim).
        If order_hint is provided, interpret strictly by that axis order (no 'batch').
        """
        if order_hint is not None:
            if "batch" in order_hint:
                raise ValueError("earth_hint may not include 'batch' in this function.")

            if Earr.ndim == 1:
                if Earr.shape != (dim,):
                    raise ValueError(f"earth expected ({dim},), got {Earr.shape}")
                return np.repeat(Earr[None, :], N, axis=0)

            if Earr.ndim != len(order_hint):
                raise ValueError(f"earth_hint implies {len(order_hint)}D but earth has ndim={Earr.ndim}, shape={Earr.shape}")

            ax_dim = order_hint.index("dim")
            ax_time = order_hint.index("time") if "time" in order_hint else None

            if Earr.shape[ax_dim] != dim:
                raise ValueError(f"earth dim axis length {Earr.shape[ax_dim]} does not match hinted dim={dim}.")

            if ax_time is None:
                squeezed = np.squeeze(Earr)
                if squeezed.shape != (dim,):
                    raise ValueError(f"earth constant form must squeeze to ({dim},), got {squeezed.shape}")
                return np.repeat(squeezed[None, :], N, axis=0)

            # transpose to (time,dim)
            E_td = np.transpose(Earr, axes=[ax_time, ax_dim])
            if E_td.shape[0] != N:
                raise ValueError(f"earth has N={E_td.shape[0]} but obj has N={N}")
            return E_td

        # ---- fallback (your original earth_to_Nd) ----
        if Earr.ndim == 1:
            if Earr.shape != (dim,):
                raise ValueError(f"earth expected ({dim},), got {Earr.shape}")
            return np.repeat(Earr[None, :], N, axis=0)

        if Earr.ndim == 2:
            if Earr.shape == (dim, 1):
                return np.repeat(Earr[:, 0][None, :], N, axis=0)
            if Earr.shape == (1, dim):
                return np.repeat(Earr[0, :][None, :], N, axis=0)
            if Earr.shape[1] == dim:
                if Earr.shape[0] != N:
                    raise ValueError(f"earth has N={Earr.shape[0]} but obj has N={N}")
                return Earr
            if Earr.shape[0] == dim:
                if Earr.shape[1] != N:
                    raise ValueError(f"earth has N={Earr.shape[1]} but obj has N={N}")
                return Earr.T

        raise ValueError(f"earth unsupported shape {Earr.shape} for dim={dim}")

    E_N = _normalize_earth_to_Nd(E, N, earth_dim, earth_order_hint)

    # For obj_dim==3, allow earth_dim==6 but ignore vel; for obj_dim==6 earth_dim==6 guaranteed
    if obj_dim == 3:
        E_use = E_N[:, :3]
    else:
        E_use = E_N

    rE = E_use[:, :3]                    # (N,3)
    vE = E_use[:, 3:] if obj_dim == 6 else None

    rO = O_int[:, :, :3]                 # (M,N,3)
    rel_r = rO - rE[None, :, :]          # (M,N,3)

    if obj_dim == 6:
        vO = O_int[:, :, 3:]
        rel_v = vO - vE[None, :, :]

    # rotation angle from Earth->Sun direction (same convention as your original)
    angles = np.arctan2(-rE[:, 1], -rE[:, 0])  # (N,)
    c = np.cos(-angles)
    s = np.sin(-angles)

    R = np.zeros((N, 3, 3), dtype=float)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0

    r_prime = np.einsum("nij,mnj->mni", R, rel_r)  # (M,N,3)

    if obj_dim == 3:
        out_int = r_prime
    else:
        rE_norm2 = np.sum(rE * rE, axis=1)
        rE_norm2 = np.maximum(rE_norm2, eps)
        omega_mag = np.linalg.norm(np.cross(rE, vE), axis=1) / rE_norm2  # (N,)

        omega = np.zeros((N, 3), dtype=float)
        omega[:, 2] = omega_mag

        v_rel_rot = np.einsum("nij,mnj->mni", R, rel_v)         # (M,N,3)
        omega_prime = np.einsum("nij,nj->ni", R, omega)         # (N,3)

        v_rot = np.cross(omega_prime[None, :, :], r_prime)      # (M,N,3)
        v_prime = v_rel_rot - v_rot                              # (M,N,3)

        out_int = np.concatenate([r_prime, v_prime], axis=2)     # (M,N,6)

    # -----------------------
    # Restore original obj layout
    # -----------------------
    def _restore_obj_from_hint(out_int_MNdim, restore):
        order = restore["order"]
        orig_ndim = restore["orig_ndim"]

        if orig_ndim == 1:
            return out_int_MNdim[0, 0, :]

        have_batch = "batch" in order
        have_time  = "time"  in order

        A = out_int_MNdim  # (M,N,dim)

        if not have_batch:
            A = A[0, :, :]           # (N,dim)
        if not have_time:
            if have_batch:
                A = A[:, 0, :]       # (M,dim)
            else:
                A = A[0, :]          # (dim,)

        present = []
        if have_batch:
            present.append("batch")
        if have_time:
            present.append("time")
        present.append("dim")

        if A.ndim == 1:
            return A

        if set(present) != set(order):
            raise RuntimeError(f"Internal restore mismatch: present={present}, order={order}")

        perm = [present.index(ax) for ax in order]
        return np.transpose(A, axes=perm)

    if out_style[0] == "hinted":
        return _restore_obj_from_hint(out_int, out_style[1])

    # fallback restore (your original)
    if out_style[0] == "single":
        return out_int[0, 0, :]
    if out_style[0] == "Mxdim":
        return out_int[:, 0, :]
    if out_style[0] == "dimxN":
        return out_int[0, :, :].T
    if out_style[0] == "Nxdim_time":
        return out_int[0, :, :]
    return out_int


def geo_secr_to_helio_eclip_generic(obj, earth, eps=1e-12, layout="auto",
                                   obj_hint=None, earth_hint=None):
    """
    Convert Earth-centered Sun–Earth co-rotating (SECR) position(s) or state(s)
    to heliocentric ECLIPJ2000.

    This is the inverse of `helio_eclip_to_geo_secr_generic` in the sense:
      helio_eclip_to_geo_secr:  r' = R(-θ) (r_obj - r_E)
                               v' = R(-θ) (v_obj - v_E) - ω' x r'
    Here we do:
      r_rel = R(+θ) r'
      v_rel = R(+θ) (v' + ω' x r')
      r_obj = r_E + r_rel
      v_obj = v_E + v_rel

    STRICT RULES:
    - obj dim = 3 → earth may be 3 or 6 (earth velocity ignored)
    - obj dim = 6 → earth MUST be 6 (otherwise ValueError)

    Supported shapes (fallback, when no explicit hint):
      obj Position: (3,), (M,3), (N,3), (3,N), (M,N,3)
      obj State:    (6,), (M,6), (N,6), (6,N), (M,N,6)

      earth Position: (3,), (3,1), (1,3), (N,3), (3,N)
      earth State:    (6,), (6,1), (1,6), (N,6), (6,N)

    layout resolves ambiguity when obj is (K,dim) (fallback mode):
      - "batch": interpret as (M,dim) objects at one time (N=1)
      - "time" : interpret as (N,dim) time series for one object (M=1)
      - "auto" : infer from earth shape when possible, else default to "batch"

    NEW (explicit structure hints):
      You can explicitly define the axis order using tokens (any permutation):
        - 'batch'   : M axis (multiple objects / spacecraft)
        - 'time'    : N axis (time series)
        - 'position': dim=3
        - 'state'   : dim=6
        - 3 or 6    : dim override (optional)

      Examples (obj):
        obj_hint=('batch','position')     -> expects obj shape (M,3)
        obj_hint=('position','batch')     -> expects obj shape (3,M)
        obj_hint=('batch','state')        -> expects obj shape (M,6)
        obj_hint=('time','state')         -> expects obj shape (N,6)
        obj_hint=('state','time')         -> expects obj shape (6,N)
        obj_hint=('time','state','batch') -> expects obj shape (N,6,M)
        obj_hint=('batch','time','state') -> expects obj shape (M,N,6)
        obj_hint=('state','batch','time') -> expects obj shape (6,M,N)
        etc.

      Examples (earth) — IMPORTANT:
        Earth is treated as TIME-indexed only in this function (no 'batch' axis supported here).
        So earth_hint may include 'time' and 'position'/'state' (or 3/6) in any permutation.

    Returns: same layout as obj; if obj_hint was used, returns in that hinted axis order.
    """
    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    O = np.asarray(obj, dtype=float)
    E = np.asarray(earth, dtype=float)

    # -----------------------
    # Hint parsing: explicit axis order
    # -----------------------
    def _parse_struct_hint(h, name, allow_batch=True):
        """
        Returns (order, dim) where:
          order: list like ['batch','dim'] or ['time','dim','batch'] (axis order)
          dim: 3 or 6
        """
        if h is None:
            return None, None

        if isinstance(h, (tuple, list, set)):
            tokens = list(h)
        elif isinstance(h, str):
            s = h.strip()
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            tokens = [t.strip() for t in s.split(",") if t.strip()]
        else:
            raise ValueError(f"{name}_hint must be None, a tuple/list/set, or a string like '(time,state,batch)'")

        order = []
        dim = None

        def _add_axis(ax):
            if ax in order:
                raise ValueError(f"{name}_hint repeats axis '{ax}'. Got {h}.")
            order.append(ax)

        for t in tokens:
            if isinstance(t, (int, np.integer)):
                iv = int(t)
                if iv in (3, 6):
                    dim = iv
                    if "dim" not in order:
                        _add_axis("dim")
                else:
                    raise ValueError(f"{name}_hint invalid dim {t} (use 3 or 6)")
                continue

            ts = str(t).strip().lower()
            if ts == "batch":
                if not allow_batch:
                    raise ValueError(f"{name}_hint may not include 'batch' (earth cannot be batch-indexed here).")
                _add_axis("batch")
            elif ts == "time":
                _add_axis("time")
            elif ts in ("position", "pos"):
                if dim is not None and dim != 3:
                    raise ValueError(f"{name}_hint conflicts: both state(6) and position(3) implied.")
                dim = 3
                _add_axis("dim")
            elif ts in ("state", "st"):
                if dim is not None and dim != 6:
                    raise ValueError(f"{name}_hint conflicts: both position(3) and state(6) implied.")
                dim = 6
                _add_axis("dim")
            elif ts in ("3", "6"):
                dim = int(ts)
                if "dim" not in order:
                    _add_axis("dim")
            elif ts == "":
                continue
            else:
                raise ValueError(
                    f"{name}_hint token '{t}' unrecognized. Use 'batch','time','position','state',3,6."
                )

        if "dim" not in order:
            raise ValueError(f"{name}_hint must include 'position'/'state' (or 3/6). Got {h}.")
        if dim not in (3, 6):
            raise ValueError(f"{name}_hint must resolve dim to 3 or 6. Got {h}.")

        return order, dim

    obj_order_hint, obj_dim_hint = _parse_struct_hint(obj_hint, "obj", allow_batch=True)
    earth_order_hint, earth_dim_hint = _parse_struct_hint(earth_hint, "earth", allow_batch=False) if earth_hint is not None else (None, None)

    # -----------------------
    # Dim inference (fallback when no explicit hint)
    # -----------------------
    def infer_dim_fallback(A, name):
        if A.ndim == 1:
            if A.shape in [(3,), (6,)]:
                return A.shape[0]
            raise ValueError(f"{name} expected (3,) or (6,), got {A.shape}")

        if A.ndim == 2:
            r, c = A.shape
            r_is = r in (3, 6)
            c_is = c in (3, 6)

            if r_is and not c_is:
                return r
            if c_is and not r_is:
                return c
            if r_is and c_is:
                return 6 if (r == 6 or c == 6) else 3

        if A.ndim == 3 and A.shape[-1] in (3, 6):
            return A.shape[-1]

        raise ValueError(f"Could not infer dim for {name} with shape {A.shape}")

    obj_dim = obj_dim_hint if obj_dim_hint is not None else infer_dim_fallback(O, "obj")
    earth_dim = earth_dim_hint if earth_dim_hint is not None else infer_dim_fallback(E, "earth")

    # --- STRICT RULE ---
    if obj_dim == 6 and earth_dim == 3:
        raise ValueError(
            "Invalid input: obj is 6D state but earth is 3D position. "
            "Earth velocity is required to compute omega and the inertial velocity."
        )

    # -----------------------
    # Normalize obj (SECR) to internal (M,N,obj_dim) and remember how to restore
    # -----------------------
    def _normalize_obj_with_hint(A, order, dim):
        """
        Strictly interpret obj using explicit axis order -> return (M,N,dim).
        Missing 'batch' or 'time' axes are treated as singleton.
        Also returns restore_info to put output back into the same hinted axis order.
        """
        if A.ndim == 1:
            if A.shape != (dim,):
                raise ValueError(f"obj expected ({dim},), got {A.shape}")
            A_int = A[None, None, :]
            restore = {"used_hint": True, "order": ["dim"], "orig_ndim": 1}
            return A_int, restore

        if A.ndim != len(order):
            raise ValueError(f"obj_hint implies {len(order)}D but obj has ndim={A.ndim}, shape={A.shape}")

        ax_dim = order.index("dim")
        ax_batch = order.index("batch") if "batch" in order else None
        ax_time  = order.index("time")  if "time"  in order else None

        if A.shape[ax_dim] != dim:
            raise ValueError(f"obj dim axis length {A.shape[ax_dim]} does not match hinted dim={dim}.")

        axes = []
        if ax_batch is not None:
            axes.append(ax_batch)
        if ax_time is not None:
            axes.append(ax_time)
        axes.append(ax_dim)

        A_perm = np.transpose(A, axes=axes)

        if ax_batch is not None and ax_time is not None:
            A_int = A_perm                  # (M,N,dim)
        elif ax_batch is not None and ax_time is None:
            A_int = A_perm[:, None, :]      # (M,1,dim)
        elif ax_batch is None and ax_time is not None:
            A_int = A_perm[None, :, :]      # (1,N,dim)
        else:
            A_int = A_perm[None, None, :]   # (1,1,dim)

        restore = {"used_hint": True, "order": order, "orig_ndim": A.ndim}
        return A_int, restore

    def _choose_mode_for_Kdim(K):
        # fallback for (K,dim) inputs without hints
        if layout in ("batch", "time"):
            return layout

        # layout == auto: infer from earth shape if possible
        earth_time_like = (
            E.ndim == 2 and (
                (E.shape[1] == earth_dim and E.shape[0] == K) or
                (E.shape[0] == earth_dim and E.shape[1] == K)
            )
        )
        return "time" if earth_time_like else "batch"

    if obj_order_hint is not None:
        O_int, obj_restore = _normalize_obj_with_hint(O, obj_order_hint, obj_dim)
        out_style = ("hinted", obj_restore)
    else:
        # ---- fallback normalization (your original behavior) ----
        if O.ndim == 1:
            if O.shape != (obj_dim,):
                raise ValueError(f"obj expected ({obj_dim},), got {O.shape}")
            O_int = O[None, None, :]
            out_style = ("single",)

        elif O.ndim == 2:
            if O.shape == (obj_dim, 1):
                O_int = O[:, 0][None, None, :]
                out_style = ("single",)

            elif O.shape == (1, obj_dim):
                O_int = O[0, :][None, None, :]
                out_style = ("single",)

            elif O.shape[0] == obj_dim and O.shape[1] != obj_dim:  # (dim,N)
                O_int = O.T[None, :, :]
                out_style = ("dimxN",)

            elif O.shape[1] == obj_dim and O.shape[0] != obj_dim:  # (K,dim)
                K = O.shape[0]
                mode = _choose_mode_for_Kdim(K)
                if mode == "batch":
                    O_int = O[:, None, :]
                    out_style = ("Mxdim",)
                else:
                    O_int = O[None, :, :]
                    out_style = ("Nxdim_time",)

            elif O.shape[0] == obj_dim and O.shape[1] == obj_dim:
                raise ValueError(
                    f"Ambiguous obj shape {O.shape}. Reshape explicitly or pass obj_hint."
                )
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

    # -----------------------
    # Normalize earth to (N,earth_dim) (time-indexed only)
    # -----------------------
    def _normalize_earth_to_Nd(Earr, N, dim, order_hint=None):
        """
        Earth is time-indexed only: output is (N,dim).
        If order_hint is provided, interpret strictly by that axis order (no 'batch').
        """
        if order_hint is not None:
            if "batch" in order_hint:
                raise ValueError("earth_hint may not include 'batch' in this function.")

            if Earr.ndim == 1:
                if Earr.shape != (dim,):
                    raise ValueError(f"earth expected ({dim},), got {Earr.shape}")
                return np.repeat(Earr[None, :], N, axis=0)

            if Earr.ndim != len(order_hint):
                raise ValueError(f"earth_hint implies {len(order_hint)}D but earth has ndim={Earr.ndim}, shape={Earr.shape}")

            ax_dim = order_hint.index("dim")
            ax_time = order_hint.index("time") if "time" in order_hint else None

            if Earr.shape[ax_dim] != dim:
                raise ValueError(f"earth dim axis length {Earr.shape[ax_dim]} does not match hinted dim={dim}.")

            if ax_time is None:
                squeezed = np.squeeze(Earr)
                if squeezed.shape != (dim,):
                    raise ValueError(f"earth constant form must squeeze to ({dim},), got {squeezed.shape}")
                return np.repeat(squeezed[None, :], N, axis=0)

            # transpose to (time,dim)
            E_td = np.transpose(Earr, axes=[ax_time, ax_dim])
            if E_td.shape[0] != N:
                raise ValueError(f"earth has N={E_td.shape[0]} but obj has N={N}")
            return E_td

        # ---- fallback (your original earth_to_Nd) ----
        if Earr.ndim == 1:
            if Earr.shape != (dim,):
                raise ValueError(f"earth expected ({dim},), got {Earr.shape}")
            return np.repeat(Earr[None, :], N, axis=0)

        if Earr.ndim == 2:
            if Earr.shape == (dim, 1):
                return np.repeat(Earr[:, 0][None, :], N, axis=0)
            if Earr.shape == (1, dim):
                return np.repeat(Earr[0, :][None, :], N, axis=0)
            if Earr.shape[1] == dim:
                if Earr.shape[0] != N:
                    raise ValueError(f"earth has N={Earr.shape[0]} but obj has N={N}")
                return Earr
            if Earr.shape[0] == dim:
                if Earr.shape[1] != N:
                    raise ValueError(f"earth has N={Earr.shape[1]} but obj has N={N}")
                return Earr.T

        raise ValueError(f"earth unsupported shape {Earr.shape} for dim={dim}")

    E_N = _normalize_earth_to_Nd(E, N, earth_dim, earth_order_hint)

    # Build Earth vector compatible with obj_dim
    if obj_dim == 3:
        E_use = E_N[:, :3]  # ignore Earth velocity if present
    else:
        E_use = E_N         # earth_dim == 6 guaranteed

    rE = E_use[:, :3]                      # (N,3)
    vE = E_use[:, 3:] if obj_dim == 6 else None

    # ---- unpack SECR obj ----
    r_p = O_int[:, :, :3]                  # (M,N,3) SECR
    v_p = O_int[:, :, 3:] if obj_dim == 6 else None

    # SECR convention (same as forward): angles = atan2(-yE, -xE)
    angles = np.arctan2(-rE[:, 1], -rE[:, 0])  # (N,)

    # Inverse uses rel_r = R(+θ) r'
    c = np.cos(angles)
    s = np.sin(angles)

    R = np.zeros((N, 3, 3), dtype=float)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0

    rel_r = np.einsum("nij,mnj->mni", R, r_p)  # (M,N,3)

    if obj_dim == 3:
        out_int = rel_r + rE[None, :, :]       # heliocentric position
    else:
        rE_norm2 = np.sum(rE * rE, axis=1)
        rE_norm2 = np.maximum(rE_norm2, eps)
        omega_mag = np.linalg.norm(np.cross(rE, vE), axis=1) / rE_norm2  # (N,)

        omega = np.zeros((N, 3), dtype=float)
        omega[:, 2] = omega_mag

        # omega' in SECR coordinates
        omega_p = np.einsum("nij,nj->ni", R, omega)  # (N,3)

        v_rel_rot = v_p + np.cross(omega_p[None, :, :], r_p)     # (M,N,3)
        rel_v = np.einsum("nij,mnj->mni", R, v_rel_rot)          # (M,N,3)

        helio_r = rel_r + rE[None, :, :]
        helio_v = rel_v + vE[None, :, :]

        out_int = np.concatenate([helio_r, helio_v], axis=2)     # (M,N,6)

    # -----------------------
    # Restore original obj layout
    # -----------------------
    def _restore_obj_from_hint(out_int_MNdim, restore):
        order = restore["order"]
        orig_ndim = restore["orig_ndim"]

        if orig_ndim == 1:
            return out_int_MNdim[0, 0, :]

        have_batch = "batch" in order
        have_time  = "time"  in order

        A = out_int_MNdim  # (M,N,dim)

        if not have_batch:
            A = A[0, :, :]           # (N,dim)
        if not have_time:
            if have_batch:
                A = A[:, 0, :]       # (M,dim)
            else:
                A = A[0, :]          # (dim,)

        present = []
        if have_batch:
            present.append("batch")
        if have_time:
            present.append("time")
        present.append("dim")

        if A.ndim == 1:
            return A

        if set(present) != set(order):
            raise RuntimeError(f"Internal restore mismatch: present={present}, order={order}")

        perm = [present.index(ax) for ax in order]
        return np.transpose(A, axes=perm)

    if out_style[0] == "hinted":
        return _restore_obj_from_hint(out_int, out_style[1])

    # fallback restore (your original)
    if out_style[0] == "single":
        return out_int[0, 0, :]
    if out_style[0] == "Mxdim":
        return out_int[:, 0, :]
    if out_style[0] == "dimxN":
        return out_int[0, :, :].T
    if out_style[0] == "Nxdim_time":
        return out_int[0, :, :]
    return out_int



def geo_eclip_to_geo_eme_generic(x, eps=1e-12, layout="auto", hint=None):
    """
    Convert geocentric ECLIPJ2000 position(s) or state(s) to geocentric EME/J2000.

    If x has 3 components -> treat as position, return position.
    If x has 6 components -> treat as full state, return full state.

    Supported x shapes (fallback, when no explicit hint):
      Position: (3,), (M,3), (N,3), (3,N), (M,N,3)
      State:    (6,), (M,6), (N,6), (6,N), (M,N,6)

    layout resolves ambiguity when x is (K,3) or (K,6) (fallback mode):
      - "batch": interpret as (M,dim) objects at one time (N=1)
      - "time" : interpret as (N,dim) time series for one object (M=1)
      - "auto" : default to "batch" (safer)

    NEW (explicit structure hint):
      You can explicitly define the axis order using tokens:
        - 'batch'   : M axis (multiple objects)
        - 'time'    : N axis (time series)
        - 'position': dim=3
        - 'state'   : dim=6
        - 3 or 6    : dim override (optional)

      Any permutation is accepted, and missing axes are treated as singleton.

      Examples:
        hint=('batch','position')   -> expects x shape (M,3)
        hint=('position','batch')   -> expects x shape (3,M)

        hint=('time','state')       -> expects x shape (N,6)
        hint=('state','time')       -> expects x shape (6,N)

        hint=('time','state','batch')  -> expects x shape (N,6,M)
        hint=('batch','time','state')  -> expects x shape (M,N,6)
        hint=('state','batch','time')  -> expects x shape (6,M,N)
        etc.

      Notes:
        - If you provide a hint, it is used strictly (shape must match its rank and dim axis size).
        - If you do NOT provide a hint, behavior matches your previous function:
            infer dim, then use layout/auto heuristics for (K,dim) 2D inputs.

    Returns: same layout as input x; if a hint was used, it returns in that hinted axis order.
    """
    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    X = np.asarray(x, dtype=float)

    # -----------------------
    # Hint parsing: explicit axis order
    # -----------------------
    def _parse_struct_hint(h, name="hint"):
        """
        Returns (order, dim) where:
          order: list like ['batch','dim'] or ['time','dim','batch'] (axis order)
          dim: 3 or 6
        """
        if h is None:
            return None, None

        if isinstance(h, (tuple, list, set)):
            tokens = list(h)
        elif isinstance(h, str):
            s = h.strip()
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            tokens = [t.strip() for t in s.split(",") if t.strip()]
        else:
            raise ValueError(f"{name} must be None, a tuple/list/set, or a string like '(time,state,batch)'")

        order = []
        dim = None

        def _add_axis(ax):
            if ax in order:
                raise ValueError(f"{name} repeats axis '{ax}'. Got {h}.")
            order.append(ax)

        for t in tokens:
            if isinstance(t, (int, np.integer)):
                iv = int(t)
                if iv in (3, 6):
                    dim = iv
                    if "dim" not in order:
                        _add_axis("dim")
                else:
                    raise ValueError(f"{name} invalid dim {t} (use 3 or 6)")
                continue

            ts = str(t).strip().lower()

            if ts == "batch":
                _add_axis("batch")
            elif ts == "time":
                _add_axis("time")
            elif ts in ("position", "pos"):
                if dim is not None and dim != 3:
                    raise ValueError(f"{name} conflicts: both state(6) and position(3) implied.")
                dim = 3
                _add_axis("dim")
            elif ts in ("state", "st"):
                if dim is not None and dim != 6:
                    raise ValueError(f"{name} conflicts: both position(3) and state(6) implied.")
                dim = 6
                _add_axis("dim")
            elif ts in ("3", "6"):
                dim = int(ts)
                if "dim" not in order:
                    _add_axis("dim")
            elif ts == "":
                continue
            else:
                raise ValueError(
                    f"{name} token '{t}' unrecognized. Use 'batch','time','position','state',3,6."
                )

        if "dim" not in order:
            raise ValueError(f"{name} must include 'position'/'state' (or 3/6). Got {h}.")
        if dim not in (3, 6):
            raise ValueError(f"{name} must resolve dim to 3 or 6. Got {h}.")

        return order, dim

    hint_order, hint_dim = _parse_struct_hint(hint, "hint")

    # -----------------------
    # Dim inference (fallback when no explicit hint)
    # -----------------------
    def infer_dim_fallback(A):
        if A.ndim == 1 and A.shape in [(3,), (6,)]:
            return A.shape[0]

        if A.ndim == 2:
            r, c = A.shape
            r_is = r in (3, 6)
            c_is = c in (3, 6)
            if r_is and not c_is:
                return r
            if c_is and not r_is:
                return c
            if r_is and c_is:
                return 6 if (r == 6 or c == 6) else 3

        if A.ndim == 3 and A.shape[-1] in (3, 6):
            return A.shape[-1]

        raise ValueError(f"Input must be position (3) or state (6); got shape {A.shape}")

    dim = hint_dim if hint_dim is not None else infer_dim_fallback(X)

    # -----------------------
    # Normalize X to internal (M,N,dim) and remember how to restore
    # -----------------------
    def _normalize_with_hint(A, order, dim):
        """
        Strictly interpret x using explicit axis order -> return (M,N,dim).
        Missing 'batch' or 'time' axes are treated as singleton.
        Also returns restore_info to put output back into the same hinted order.
        """
        if A.ndim == 1:
            if A.shape != (dim,):
                raise ValueError(f"x expected ({dim},), got {A.shape}")
            A_int = A[None, None, :]
            restore = {"used_hint": True, "order": ["dim"], "orig_ndim": 1}
            return A_int, restore

        if A.ndim != len(order):
            raise ValueError(f"hint implies {len(order)}D but x has ndim={A.ndim}, shape={A.shape}")

        ax_dim = order.index("dim")
        ax_batch = order.index("batch") if "batch" in order else None
        ax_time  = order.index("time")  if "time"  in order else None

        if A.shape[ax_dim] != dim:
            raise ValueError(f"x dim axis length {A.shape[ax_dim]} does not match hinted dim={dim}.")

        axes = []
        if ax_batch is not None:
            axes.append(ax_batch)
        if ax_time is not None:
            axes.append(ax_time)
        axes.append(ax_dim)

        A_perm = np.transpose(A, axes=axes)

        if ax_batch is not None and ax_time is not None:
            A_int = A_perm                  # (M,N,dim)
        elif ax_batch is not None and ax_time is None:
            A_int = A_perm[:, None, :]      # (M,1,dim)
        elif ax_batch is None and ax_time is not None:
            A_int = A_perm[None, :, :]      # (1,N,dim)
        else:
            A_int = A_perm[None, None, :]   # (1,1,dim)

        restore = {"used_hint": True, "order": order, "orig_ndim": A.ndim}
        return A_int, restore

    def _choose_mode_for_Kdim(K):
        # fallback for (K,dim) inputs without hints
        if layout in ("batch", "time"):
            return layout
        return "batch"  # auto default

    if hint_order is not None:
        X_int, restore_info = _normalize_with_hint(X, hint_order, dim)
        out_style = ("hinted", restore_info)
    else:
        # ---- fallback normalization (your original behavior) ----
        if X.ndim == 1:
            if X.shape != (dim,):
                raise ValueError(f"Expected ({dim},), got {X.shape}")
            X_int = X[None, None, :]
            out_style = ("single",)

        elif X.ndim == 2:
            if X.shape == (dim, 1):
                X_int = X[:, 0][None, None, :]
                out_style = ("single",)

            elif X.shape == (1, dim):
                X_int = X[0, :][None, None, :]
                out_style = ("single",)

            elif X.shape[0] == dim and X.shape[1] != dim:      # (dim,N)
                X_int = X.T[None, :, :]                         # (1,N,dim)
                out_style = ("dimxN",)

            elif X.shape[1] == dim and X.shape[0] != dim:      # (K,dim)
                K = X.shape[0]
                mode = _choose_mode_for_Kdim(K)
                if mode == "time":
                    X_int = X[None, :, :]                       # (1,N,dim)
                    out_style = ("Nxdim_time",)
                else:
                    X_int = X[:, None, :]                       # (M,1,dim)
                    out_style = ("Mxdim",)

            elif X.shape[0] == dim and X.shape[1] == dim:
                raise ValueError(
                    f"Ambiguous input shape {X.shape}. Reshape explicitly or pass hint."
                )
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

    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )

    r = X_int[:, :, :3]
    r_eme = np.einsum("ij,mnj->mni", R, r)

    if dim == 3:
        out_int = r_eme
    else:
        v = X_int[:, :, 3:]
        v_eme = np.einsum("ij,mnj->mni", R, v)
        out_int = np.concatenate([r_eme, v_eme], axis=2)  # (M,N,6)

    # -----------------------
    # Restore output layout
    # -----------------------
    def _restore_from_hint(out_int_MNdim, restore):
        order = restore["order"]
        orig_ndim = restore["orig_ndim"]

        if orig_ndim == 1:
            return out_int_MNdim[0, 0, :]

        have_batch = "batch" in order
        have_time  = "time"  in order

        A = out_int_MNdim  # (M,N,dim)

        if not have_batch:
            A = A[0, :, :]           # (N,dim)
        if not have_time:
            if have_batch:
                A = A[:, 0, :]       # (M,dim)
            else:
                A = A[0, :]          # (dim,)

        present = []
        if have_batch:
            present.append("batch")
        if have_time:
            present.append("time")
        present.append("dim")

        if A.ndim == 1:
            return A

        if set(present) != set(order):
            raise RuntimeError(f"Internal restore mismatch: present={present}, order={order}")

        perm = [present.index(ax) for ax in order]
        return np.transpose(A, axes=perm)

    if out_style[0] == "hinted":
        return _restore_from_hint(out_int, out_style[1])

    # fallback restore (your original)
    if out_style[0] == "single":
        return out_int[0, 0, :]
    if out_style[0] == "Mxdim":
        return out_int[:, 0, :]
    if out_style[0] == "dimxN":
        return out_int[0, :, :].T
    if out_style[0] == "Nxdim_time":
        return out_int[0, :, :]
    return out_int


def helio_eclip_to_geo_eme_generic(obj, earth, eps=1e-12, layout="auto",
                                  obj_hint=None, earth_hint=None):
    """
    Convert heliocentric ECLIPJ2000 position(s) or state(s) to geocentric EME/J2000.

    Rules:
    - obj dim = 3 → earth may be 3 or 6 (earth velocity ignored)
    - obj dim = 6 → earth MUST be 6 (otherwise ValueError)

    Supported shapes (fallback, when no explicit hint):
      obj Position: (3,), (M,3), (N,3), (3,N), (M,N,3)
      obj State:    (6,), (M,6), (N,6), (6,N), (M,N,6)

      earth Position: (3,), (3,1), (1,3), (N,3), (3,N)
      earth State:    (6,), (6,1), (1,6), (N,6), (6,N)

    layout resolves ambiguity when obj is (K,dim) (fallback mode):
      - "batch": interpret as (M,dim) objects at one time (N=1)
      - "time" : interpret as (N,dim) time series for one object (M=1)
      - "auto" : infer from earth shape when possible, else default to "batch"

    NEW (explicit structure hints):
      You can explicitly define the axis order using tokens (any permutation):
        - 'batch'   : M axis (multiple objects / spacecraft)
        - 'time'    : N axis (time series)
        - 'position': dim=3
        - 'state'   : dim=6
        - 3 or 6    : dim override (optional)

      Examples (obj):
        obj_hint=('batch','position')     -> expects obj shape (M,3)
        obj_hint=('position','batch')     -> expects obj shape (3,M)
        obj_hint=('batch','state')        -> expects obj shape (M,6)
        obj_hint=('time','state')         -> expects obj shape (N,6)
        obj_hint=('state','time')         -> expects obj shape (6,N)
        obj_hint=('time','state','batch') -> expects obj shape (N,6,M)
        obj_hint=('batch','time','state') -> expects obj shape (M,N,6)
        obj_hint=('state','batch','time') -> expects obj shape (6,M,N)
        etc.

      Examples (earth) — IMPORTANT:
        Earth is treated as TIME-indexed only in this function (no 'batch' axis here),
        because the transform uses a single Earth state per time index.
        So earth_hint may include:
          - 'time' and 'position'/'state' (or 3/6)
          - in any permutation for 2D/3D (but no 'batch')

        e.g. earth_hint=('time','state') -> (N,6)
             earth_hint=('state','time') -> (6,N)

      Missing 'batch' or 'time' axes in a hint are treated as singleton.

    Returns: same layout as obj; if obj_hint was used, returns in that hinted axis order.
    """

    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    O = np.asarray(obj, dtype=float)
    E = np.asarray(earth, dtype=float)

    # -----------------------
    # Hint parsing: explicit axis order
    # -----------------------
    def _parse_struct_hint(h, name, allow_batch=True):
        """
        Returns (order, dim) where:
          order: list like ['batch','dim'] or ['time','dim','batch'] (axis order)
          dim: 3 or 6
        """
        if h is None:
            return None, None

        if isinstance(h, (tuple, list, set)):
            tokens = list(h)
        elif isinstance(h, str):
            s = h.strip()
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            tokens = [t.strip() for t in s.split(",") if t.strip()]
        else:
            raise ValueError(f"{name}_hint must be None, a tuple/list/set, or a string like '(time,state,batch)'")

        order = []
        dim = None

        def _add_axis(ax):
            if ax in order:
                raise ValueError(f"{name}_hint repeats axis '{ax}'. Got {h}.")
            order.append(ax)

        for t in tokens:
            if isinstance(t, (int, np.integer)):
                iv = int(t)
                if iv in (3, 6):
                    dim = iv
                    if "dim" not in order:
                        _add_axis("dim")
                else:
                    raise ValueError(f"{name}_hint invalid dim {t} (use 3 or 6)")
                continue

            ts = str(t).strip().lower()
            if ts == "batch":
                if not allow_batch:
                    raise ValueError(f"{name}_hint may not include 'batch' (earth cannot be batch-indexed here).")
                _add_axis("batch")
            elif ts == "time":
                _add_axis("time")
            elif ts in ("position", "pos"):
                if dim is not None and dim != 3:
                    raise ValueError(f"{name}_hint conflicts: both state(6) and position(3) implied.")
                dim = 3
                _add_axis("dim")
            elif ts in ("state", "st"):
                if dim is not None and dim != 6:
                    raise ValueError(f"{name}_hint conflicts: both position(3) and state(6) implied.")
                dim = 6
                _add_axis("dim")
            elif ts in ("3", "6"):
                dim = int(ts)
                if "dim" not in order:
                    _add_axis("dim")
            elif ts == "":
                continue
            else:
                raise ValueError(
                    f"{name}_hint token '{t}' unrecognized. Use 'batch','time','position','state',3,6."
                )

        if "dim" not in order:
            raise ValueError(f"{name}_hint must include 'position'/'state' (or 3/6). Got {h}.")
        if dim not in (3, 6):
            raise ValueError(f"{name}_hint must resolve dim to 3 or 6. Got {h}.")

        return order, dim

    obj_order_hint, obj_dim_hint = _parse_struct_hint(obj_hint, "obj", allow_batch=True)
    earth_order_hint, earth_dim_hint = _parse_struct_hint(earth_hint, "earth", allow_batch=False) if earth_hint is not None else (None, None)

    # -----------------------
    # Dim inference (fallback when no explicit hint)
    # -----------------------
    def infer_dim_fallback(A, name):
        if A.ndim == 1:
            if A.shape in [(3,), (6,)]:
                return A.shape[0]
            raise ValueError(f"{name} expected (3,) or (6,), got {A.shape}")

        if A.ndim == 2:
            r, c = A.shape
            r_is = r in (3, 6)
            c_is = c in (3, 6)

            if r_is and not c_is:
                return r
            if c_is and not r_is:
                return c
            if r_is and c_is:
                return 6 if (r == 6 or c == 6) else 3

        if A.ndim == 3 and A.shape[-1] in (3, 6):
            return A.shape[-1]

        raise ValueError(f"Could not infer dim for {name} with shape {A.shape}")

    obj_dim = obj_dim_hint if obj_dim_hint is not None else infer_dim_fallback(O, "obj")
    earth_dim = earth_dim_hint if earth_dim_hint is not None else infer_dim_fallback(E, "earth")

    # -----------------------
    # STRICT RULE
    # -----------------------
    if obj_dim == 6 and earth_dim == 3:
        raise ValueError(
            "Invalid input: obj is 6D state but earth is 3D position. "
            "Earth velocity is required for 6D transformation."
        )

    # -----------------------
    # Normalize obj to internal (M,N,obj_dim) and remember how to restore
    # -----------------------
    def _normalize_obj_with_hint(A, order, dim):
        """
        Strictly interpret obj using explicit axis order -> return (M,N,dim).
        Missing 'batch' or 'time' axes in the hint are treated as singleton.
        Also returns restore_info to put output back into the same hinted order.
        """
        if A.ndim == 1:
            if A.shape != (dim,):
                raise ValueError(f"obj expected ({dim},), got {A.shape}")
            A_int = A[None, None, :]
            restore = {"used_hint": True, "order": ["dim"], "orig_ndim": 1}
            return A_int, restore

        if A.ndim != len(order):
            raise ValueError(f"obj_hint implies {len(order)}D but obj has ndim={A.ndim}, shape={A.shape}")

        ax_dim = order.index("dim")
        ax_batch = order.index("batch") if "batch" in order else None
        ax_time  = order.index("time")  if "time"  in order else None

        if A.shape[ax_dim] != dim:
            raise ValueError(f"obj dim axis length {A.shape[ax_dim]} does not match hinted dim={dim}.")

        axes = []
        if ax_batch is not None:
            axes.append(ax_batch)
        if ax_time is not None:
            axes.append(ax_time)
        axes.append(ax_dim)

        A_perm = np.transpose(A, axes=axes)

        if ax_batch is not None and ax_time is not None:
            A_int = A_perm                  # (M,N,dim)
        elif ax_batch is not None and ax_time is None:
            A_int = A_perm[:, None, :]      # (M,1,dim)
        elif ax_batch is None and ax_time is not None:
            A_int = A_perm[None, :, :]      # (1,N,dim)
        else:
            A_int = A_perm[None, None, :]   # (1,1,dim)

        restore = {"used_hint": True, "order": order, "orig_ndim": A.ndim}
        return A_int, restore

    def _choose_obj_mode_for_Kdim(K):
        # Priority (fallback): layout if not auto; else auto heuristic using earth; else batch default
        if layout in ("batch", "time"):
            return layout
        if layout == "auto":
            earth_time_like = (
                E.ndim == 2 and (
                    (E.shape[1] == earth_dim and E.shape[0] == K) or
                    (E.shape[0] == earth_dim and E.shape[1] == K)
                )
            )
            return "time" if earth_time_like else "batch"
        return "batch"

    if obj_order_hint is not None:
        O_int, obj_restore = _normalize_obj_with_hint(O, obj_order_hint, obj_dim)
        out_style = ("hinted", obj_restore)
    else:
        # ---- fallback normalization (your original behavior) ----
        if O.ndim == 1:
            if O.shape != (obj_dim,):
                raise ValueError(f"obj expected ({obj_dim},), got {O.shape}")
            O_int = O[None, None, :]
            out_style = ("single",)

        elif O.ndim == 2:
            if O.shape == (obj_dim, 1):
                O_int = O[:, 0][None, None, :]
                out_style = ("single",)

            elif O.shape == (1, obj_dim):
                O_int = O[0, :][None, None, :]
                out_style = ("single",)

            elif O.shape[0] == obj_dim and O.shape[1] != obj_dim:  # (dim,N)
                O_int = O.T[None, :, :]
                out_style = ("dimxN",)

            elif O.shape[1] == obj_dim and O.shape[0] != obj_dim:  # (K,dim)
                K = O.shape[0]
                mode = _choose_obj_mode_for_Kdim(K)
                if mode == "time":
                    O_int = O[None, :, :]
                    out_style = ("Nxdim_time",)
                else:
                    O_int = O[:, None, :]
                    out_style = ("Mxdim",)

            elif O.shape[0] == obj_dim and O.shape[1] == obj_dim:
                raise ValueError(
                    f"Ambiguous obj shape {O.shape}. Reshape explicitly or pass obj_hint."
                )
            else:
                raise ValueError(f"Unsupported obj shape {O.shape} for dim={obj_dim}")

        elif O.ndim == 3:
            if O.shape[2] != obj_dim:
                raise ValueError(f"Expected last dim {obj_dim}, got {O.shape}")
            O_int = O
            out_style = ("MNdim",)

        else:
            raise ValueError(f"Unsupported obj ndim {O.ndim}")

    M, N, _ = O_int.shape

    # -----------------------
    # Normalize earth to (N,earth_dim) (time-indexed only)
    # -----------------------
    def _normalize_earth_to_Nd(Earr, N, dim, order_hint=None):
        """
        Earth is time-indexed only: output is (N,dim).
        If order_hint is provided, interpret strictly by that axis order (no 'batch').
        """
        if order_hint is not None:
            if "batch" in order_hint:
                raise ValueError("earth_hint may not include 'batch' in this function.")

            if Earr.ndim == 1:
                if Earr.shape != (dim,):
                    raise ValueError(f"earth expected ({dim},), got {Earr.shape}")
                return np.repeat(Earr[None, :], N, axis=0)

            if Earr.ndim != len(order_hint):
                raise ValueError(f"earth_hint implies {len(order_hint)}D but earth has ndim={Earr.ndim}, shape={Earr.shape}")

            ax_dim = order_hint.index("dim")
            ax_time = order_hint.index("time") if "time" in order_hint else None

            if Earr.shape[ax_dim] != dim:
                raise ValueError(f"earth dim axis length {Earr.shape[ax_dim]} does not match hinted dim={dim}.")

            if ax_time is None:
                # no time axis -> constant
                squeezed = np.squeeze(Earr)
                if squeezed.shape != (dim,):
                    raise ValueError(f"earth constant form must squeeze to ({dim},), got {squeezed.shape}")
                return np.repeat(squeezed[None, :], N, axis=0)

            # transpose to (time,dim)
            E_td = np.transpose(Earr, axes=[ax_time, ax_dim])
            if E_td.shape[0] != N:
                raise ValueError(f"earth has N={E_td.shape[0]} but obj has N={N}")
            return E_td

        # ---- fallback (your original earth_to_Nd) ----
        if Earr.ndim == 1:
            if Earr.shape != (dim,):
                raise ValueError(f"earth expected ({dim},), got {Earr.shape}")
            return np.repeat(Earr[None, :], N, axis=0)

        if Earr.ndim == 2:
            if Earr.shape == (dim, 1):
                return np.repeat(Earr[:, 0][None, :], N, axis=0)
            if Earr.shape == (1, dim):
                return np.repeat(Earr[0, :][None, :], N, axis=0)

            if Earr.shape[0] == dim and Earr.shape[1] == N:  # (dim,N)
                return Earr.T
            if Earr.shape[1] == dim and Earr.shape[0] == N:  # (N,dim)
                return Earr

            if (Earr.shape[0] == dim) and (Earr.shape[1] == dim):
                raise ValueError(
                    f"Ambiguous earth shape {Earr.shape} with dim={dim} (could be (dim,N) or (N,dim)). "
                    f"Reshape explicitly or pass earth_hint."
                )

        raise ValueError(f"Unsupported earth shape {Earr.shape} for dim={dim} and N={N}")

    E_N = _normalize_earth_to_Nd(E, N, earth_dim, earth_order_hint)

    # -----------------------
    # Build Earth vector compatible with obj_dim
    # -----------------------
    if obj_dim == 3:
        E_use = E_N[:, :3]     # ignore earth velocity even if provided
    else:
        E_use = E_N            # earth_dim == 6 guaranteed here

    # -----------------------
    # Heliocentric → geocentric (still in ecliptic frame)
    # -----------------------
    geo_ecl = O_int - E_use[None, :, :]

    # -----------------------
    # Rotate ecliptic → EME
    # -----------------------
    # NOTE: eps argument in signature is unused here previously; keep your hard-coded mean obliquity.
    eps_rad = np.deg2rad(23.439281)
    c, s = np.cos(eps_rad), np.sin(eps_rad)

    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )

    r = geo_ecl[:, :, :3]
    r_eme = np.einsum("ij,mnj->mni", R, r)

    if obj_dim == 3:
        out_int = r_eme
    else:
        v = geo_ecl[:, :, 3:]
        v_eme = np.einsum("ij,mnj->mni", R, v)
        out_int = np.concatenate([r_eme, v_eme], axis=2)  # (M,N,6)

    # -----------------------
    # Restore original obj layout
    # -----------------------
    def _restore_obj_from_hint(out_int_MNdim, restore):
        order = restore["order"]
        orig_ndim = restore["orig_ndim"]

        if orig_ndim == 1:
            return out_int_MNdim[0, 0, :]

        have_batch = "batch" in order
        have_time  = "time"  in order

        A = out_int_MNdim  # (M,N,dim)

        if not have_batch:
            A = A[0, :, :]           # (N,dim)
        if not have_time:
            if have_batch:
                A = A[:, 0, :]       # (M,dim)
            else:
                A = A[0, :]          # (dim,)

        present = []
        if have_batch:
            present.append("batch")
        if have_time:
            present.append("time")
        present.append("dim")

        if A.ndim == 1:
            return A

        if set(present) != set(order):
            raise RuntimeError(f"Internal restore mismatch: present={present}, order={order}")

        perm = [present.index(ax) for ax in order]
        return np.transpose(A, axes=perm)

    if out_style[0] == "hinted":
        return _restore_obj_from_hint(out_int, out_style[1])

    # fallback restore (your original)
    if out_style[0] == "single":
        return out_int[0, 0, :]
    if out_style[0] == "Mxdim":
        return out_int[:, 0, :]
    if out_style[0] == "dimxN":
        return out_int[0, :, :].T
    if out_style[0] == "Nxdim_time":
        return out_int[0, :, :]
    return out_int


def geo_secr_to_geo_eclip_generic(obj, earth, eps=1e-12, layout="auto",
                                 obj_hint=None, earth_hint=None):
    """
    Convert SECR (Earth-centered rotating) position(s) or state(s) to geocentric ECLIPJ2000.

    STRICT RULES:
    - obj dim = 3 → earth may be 3 or 6 (earth velocity ignored)
    - obj dim = 6 → earth MUST be 6 (otherwise ValueError)

    HINT SYSTEM (new, explicit structure):
      You can now explicitly define the axis order using tokens:
        - 'batch'   : M axis (multiple objects / spacecraft)
        - 'time'    : N axis (time series)
        - 'position': dim=3
        - 'state'   : dim=6
        - 3 or 6    : dim override (optional)

      Examples (all permutations supported):
        obj_hint=('batch','position')  -> (M,3)
        obj_hint=('position','batch')  -> (3,M)
        obj_hint=('batch','state')     -> (M,6)

        obj_hint=('time','state')      -> (N,6)
        obj_hint=('state','time')      -> (6,N)

        obj_hint=('time','state','batch') -> (N,6,M)
        obj_hint=('batch','time','state') -> (M,N,6)
        etc.

      Earth hints:
        Earth is treated as time-indexed only (no 'batch' axis supported):
          earth_hint=('time','state') -> (N,6)
          earth_hint=('state','time') -> (6,N)
          earth_hint=('time','position') -> (N,3)
          earth_hint=('position','time') -> (3,N)
        If earth is constant, pass (3,) or (6,) (hint optional).

    Returns: same layout as obj (including the hinted axis order if hint was used).
    """

    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    O = np.asarray(obj, dtype=float)
    E = np.asarray(earth, dtype=float)

    # -----------------------
    # Hint parsing: explicit axis order
    # -----------------------
    def _parse_struct_hint(h, name, allow_batch=True):
        """
        Returns (order, dim) where:
          order: list like ['batch','dim'] or ['time','dim','batch'] (axis order)
          dim: 3 or 6 or None
        """
        if h is None:
            return None, None

        # tokenize
        if isinstance(h, (tuple, list, set)):
            tokens = list(h)
        elif isinstance(h, str):
            s = h.strip()
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            tokens = [t.strip() for t in s.split(",") if t.strip()]
        else:
            raise ValueError(f"{name}_hint must be None, a tuple/list/set, or a string like '(time,state,batch)'")

        order = []
        dim = None

        def _add_axis(ax):
            if ax in order:
                raise ValueError(f"{name}_hint repeats axis '{ax}'. Got {h}.")
            order.append(ax)

        for t in tokens:
            # ints
            if isinstance(t, (int, np.integer)):
                iv = int(t)
                if iv in (3, 6):
                    dim = iv
                else:
                    raise ValueError(f"{name}_hint invalid dim {t} (use 3 or 6)")
                continue

            ts = str(t).strip().lower()

            if ts == "batch":
                if not allow_batch:
                    raise ValueError(f"{name}_hint may not include 'batch'. Earth cannot be batch-indexed here.")
                _add_axis("batch")
            elif ts == "time":
                _add_axis("time")
            elif ts in ("position", "pos"):
                if dim is not None and dim != 3:
                    raise ValueError(f"{name}_hint conflicts: both state(6) and position(3) implied.")
                dim = 3
                _add_axis("dim")
            elif ts in ("state", "st"):
                if dim is not None and dim != 6:
                    raise ValueError(f"{name}_hint conflicts: both position(3) and state(6) implied.")
                dim = 6
                _add_axis("dim")
            elif ts in ("3", "6"):
                dim = int(ts)
                # if user did NOT include position/state token, we still need a 'dim' axis
                if "dim" not in order:
                    _add_axis("dim")
            elif ts == "":
                continue
            else:
                raise ValueError(
                    f"{name}_hint token '{t}' unrecognized. Use 'batch','time','position','state',3,6."
                )

        # if they set dim via int but forgot to include dim axis explicitly (should not happen due to above)
        if dim is not None and "dim" not in order:
            _add_axis("dim")

        # allow hints like ('batch','time')? -> no, must specify position/state or 3/6
        if "dim" not in order:
            raise ValueError(f"{name}_hint must include 'position'/'state' (or 3/6). Got {h}.")

        # sanity
        if (dim not in (3, 6)):
            raise ValueError(f"{name}_hint must resolve dim to 3 or 6. Got {h}.")

        return order, dim

    obj_order_hint, obj_dim_hint = _parse_struct_hint(obj_hint, "obj", allow_batch=True)
    earth_order_hint, earth_dim_hint = _parse_struct_hint(earth_hint, "earth", allow_batch=False) if earth_hint is not None else (None, None)

    # -----------------------
    # Dim inference (fallback when no explicit hint)
    # -----------------------
    def infer_dim_fallback(A, name):
        if A.ndim == 1:
            if A.shape in [(3,), (6,)]:
                return A.shape[0]
            raise ValueError(f"{name} expected (3,) or (6,), got {A.shape}")

        if A.ndim == 2:
            r, c = A.shape
            r_is = r in (3, 6)
            c_is = c in (3, 6)
            if r_is and not c_is:
                return r
            if c_is and not r_is:
                return c
            if r_is and c_is:
                # prefer 6 if present
                return 6 if (r == 6 or c == 6) else 3

        if A.ndim == 3 and A.shape[-1] in (3, 6):
            return A.shape[-1]

        raise ValueError(f"Could not infer dim for {name} with shape {A.shape}")

    obj_dim = obj_dim_hint if obj_dim_hint is not None else infer_dim_fallback(O, "obj")
    earth_dim = earth_dim_hint if earth_dim_hint is not None else infer_dim_fallback(E, "earth")

    # -----------------------
    # STRICT RULE
    # -----------------------
    if obj_dim == 6 and earth_dim == 3:
        raise ValueError(
            "Invalid input: obj is 6D state but earth is 3D position. "
            "Earth velocity is required to compute omega and the inertial velocity correction."
        )

    # -----------------------
    # Normalize obj to internal (M,N,dim), and record how to restore
    # -----------------------
    def _normalize_obj_with_hint(A, order, dim):
        """
        Uses explicit axis order to convert A to (M,N,dim).
        Returns (A_int, restore_info)
        restore_info includes:
          - used_hint: True
          - order: original order list
          - orig_ndim: 1/2/3
        Missing axes ('batch' or 'time') are treated as singleton.
        """
        if A.ndim == 1:
            if A.shape != (dim,):
                raise ValueError(f"obj expected ({dim},), got {A.shape}")
            A_int = A[None, None, :]
            restore = {"used_hint": True, "order": ["dim"], "orig_ndim": 1}
            return A_int, restore

        # order can be length 2 or 3; A.ndim must match it
        if A.ndim != len(order):
            raise ValueError(f"obj_hint implies {len(order)}D but obj has ndim={A.ndim}, shape={A.shape}")

        # locate axes
        ax_dim = order.index("dim")
        ax_batch = order.index("batch") if "batch" in order else None
        ax_time  = order.index("time")  if "time" in order else None

        # verify dim axis length
        if A.shape[ax_dim] != dim:
            raise ValueError(f"obj dim axis length {A.shape[ax_dim]} does not match hinted dim={dim}.")

        # compute M, N (singleton if missing)
        M = A.shape[ax_batch] if ax_batch is not None else 1
        N = A.shape[ax_time]  if ax_time  is not None else 1

        # Permute to (batch,time,dim) with missing axes injected
        # Build an array with explicit axes in current A:
        # Start from A, transpose existing axes into the order [batch,time,dim] (without missing)
        target_axes_existing = []
        if ax_batch is not None:
            target_axes_existing.append(ax_batch)
        if ax_time is not None:
            target_axes_existing.append(ax_time)
        target_axes_existing.append(ax_dim)

        A_perm = np.transpose(A, axes=target_axes_existing)

        # Now A_perm is either:
        #   (M,N,dim) if both present
        #   (M,dim)   if only batch
        #   (N,dim)   if only time
        if ax_batch is not None and ax_time is not None:
            A_int = A_perm  # (M,N,dim)
        elif ax_batch is not None and ax_time is None:
            A_int = A_perm[:, None, :]  # (M,1,dim)
        elif ax_batch is None and ax_time is not None:
            A_int = A_perm[None, :, :]  # (1,N,dim)
        else:
            # only dim axis in a 2D/3D hint would be weird; but keep safe
            A_int = A_perm[None, None, :]

        restore = {"used_hint": True, "order": order, "orig_ndim": A.ndim}
        return A_int, restore

    def _choose_obj_mode_for_Kdim(K):
        # fallback behavior when no explicit hint and obj is (K,dim)
        if layout in ("batch", "time"):
            return layout
        # layout == auto: infer from earth shape heuristic
        earth_time_like = (
            E.ndim == 2 and (
                (E.shape[1] == earth_dim and E.shape[0] == K) or
                (E.shape[0] == earth_dim and E.shape[1] == K)
            )
        )
        return "time" if earth_time_like else "batch"

    if obj_order_hint is not None:
        O_int, obj_restore = _normalize_obj_with_hint(O, obj_order_hint, obj_dim)
        out_style = ("hinted", obj_restore)
    else:
        # ---- original-ish fallback normalization ----
        if O.ndim == 1:
            if O.shape != (obj_dim,):
                raise ValueError(f"obj expected ({obj_dim},), got {O.shape}")
            O_int = O[None, None, :]
            out_style = ("single",)

        elif O.ndim == 2:
            if O.shape == (obj_dim, 1):
                O_int = O[:, 0][None, None, :]
                out_style = ("single",)

            elif O.shape == (1, obj_dim):
                O_int = O[0, :][None, None, :]
                out_style = ("single",)

            elif O.shape[0] == obj_dim and O.shape[1] != obj_dim:  # (dim,N)
                O_int = O.T[None, :, :]
                out_style = ("dimxN",)

            elif O.shape[1] == obj_dim and O.shape[0] != obj_dim:  # (K,dim)
                K = O.shape[0]
                mode = _choose_obj_mode_for_Kdim(K)
                if mode == "time":
                    O_int = O[None, :, :]
                    out_style = ("Nxdim_time",)
                else:
                    O_int = O[:, None, :]
                    out_style = ("Mxdim",)

            elif O.shape[0] == obj_dim and O.shape[1] == obj_dim:
                raise ValueError(
                    f"Ambiguous obj shape {O.shape}. Reshape explicitly or use obj_hint."
                )
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

    # -----------------------
    # Normalize earth to (N, earth_dim) (time-indexed only)
    # -----------------------
    def _normalize_earth_to_Nd(Earr, N, dim, order_hint=None):
        # If explicit earth hint provided, obey it and transpose/reshape accordingly.
        if order_hint is not None:
            # earth can be (dim,), (N,dim), (dim,N); or hinted 2D/1D
            if Earr.ndim == 1:
                if Earr.shape != (dim,):
                    raise ValueError(f"earth expected ({dim},), got {Earr.shape}")
                return np.repeat(Earr[None, :], N, axis=0)

            if Earr.ndim != len(order_hint):
                raise ValueError(f"earth_hint implies {len(order_hint)}D but earth has ndim={Earr.ndim}, shape={Earr.shape}")

            if "batch" in order_hint:
                raise ValueError("earth_hint may not include 'batch' in this function.")

            ax_dim = order_hint.index("dim")
            ax_time = order_hint.index("time") if "time" in order_hint else None

            if Earr.shape[ax_dim] != dim:
                raise ValueError(f"earth dim axis length {Earr.shape[ax_dim]} does not match hinted dim={dim}.")

            if ax_time is None:
                # no time axis: treat as constant
                if Earr.ndim != 1:
                    # if user gave e.g. (dim,1) or (1,dim) they should include time; but allow (dim,1) etc by squeeze
                    squeezed = np.squeeze(Earr)
                    if squeezed.shape != (dim,):
                        raise ValueError(f"earth constant form must squeeze to ({dim},), got {squeezed.shape}")
                    return np.repeat(squeezed[None, :], N, axis=0)
                return np.repeat(Earr[None, :], N, axis=0)

            # transpose to (time,dim)
            E_td = np.transpose(Earr, axes=[ax_time, ax_dim])
            if E_td.shape[0] != N:
                raise ValueError(f"earth has N={E_td.shape[0]} but obj has N={N}")
            return E_td

        # ---- fallback behavior (original) ----
        if Earr.ndim == 1:
            if Earr.shape != (dim,):
                raise ValueError(f"earth expected ({dim},), got {Earr.shape}")
            return np.repeat(Earr[None, :], N, axis=0)

        if Earr.ndim == 2:
            if Earr.shape == (dim, 1):
                return np.repeat(Earr[:, 0][None, :], N, axis=0)
            if Earr.shape == (1, dim):
                return np.repeat(Earr[0, :][None, :], N, axis=0)

            if Earr.shape[1] == dim and Earr.shape[0] == N:  # (N,dim)
                return Earr
            if Earr.shape[0] == dim and Earr.shape[1] == N:  # (dim,N)
                return Earr.T

            if Earr.shape[0] == dim and Earr.shape[1] == dim:
                raise ValueError(
                    f"Ambiguous earth shape {Earr.shape} with dim={dim}. Reshape explicitly or use earth_hint."
                )

        raise ValueError(f"earth unsupported shape {Earr.shape} for dim={dim} and N={N}")

    E_N = _normalize_earth_to_Nd(E, N, earth_dim, earth_order_hint)

    # For obj_dim==3, allow earth_dim==6 but ignore vel; for obj_dim==6 earth_dim==6 is guaranteed
    h_r_E = E_N[:, :3]                 # (N,3)
    h_v_E = E_N[:, 3:] if obj_dim == 6 else None

    # ---- unpack SECR obj ----
    r_p = O_int[:, :, :3]              # (M,N,3) in SECR
    v_p = O_int[:, :, 3:] if obj_dim == 6 else None

    # angles from Earth-Sun direction (same convention as forward function)
    angles = np.arctan2(-h_r_E[:, 1], -h_r_E[:, 0])  # (N,)

    c = np.cos(angles)
    s = np.sin(angles)

    # Rotation SECR -> inertial ecliptic: Rz(+angles)
    R = np.zeros((N, 3, 3), dtype=float)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0

    # inertial geocentric position: r = R * r'
    geo_r = np.einsum("nij,mnj->mni", R, r_p)  # (M,N,3)

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
        omega_p = np.einsum("nij,nj->ni", R, omega)  # (N,3)

        # inertial relative velocity: v = R * (v' + omega' x r')
        v_rel_p = v_p + np.cross(omega_p[None, :, :], r_p)  # (M,N,3)
        geo_v = np.einsum("nij,mnj->mni", R, v_rel_p)        # (M,N,3)

        out_int = np.concatenate([geo_r, geo_v], axis=2)     # (M,N,6)

    # -----------------------
    # Restore original layout
    # -----------------------
    def _restore_obj_from_hint(out_int_MNdim, restore):
        order = restore["order"]
        orig_ndim = restore["orig_ndim"]

        # If original was 1D dim-only
        if orig_ndim == 1:
            return out_int_MNdim[0, 0, :]

        # Build an array in the original hinted axis order.
        # We have out_int in (M,N,dim). We need to:
        #  - drop singleton axes for missing batch/time in the hint
        #  - permute to the hint order
        have_batch = "batch" in order
        have_time  = "time" in order

        # start from (M,N,dim)
        A = out_int_MNdim

        # if hint did not include batch, drop M axis
        if not have_batch:
            A = A[0, :, :]  # (N,dim)
        # if hint did not include time, drop N axis
        if not have_time:
            if have_batch:
                A = A[:, 0, :]  # (M,dim)
            else:
                A = A[0, :]     # (dim,)

        # Now A has axes among:
        #  - batch, time, dim that were present
        present_axes = []
        if have_batch:
            present_axes.append("batch")
        if have_time:
            present_axes.append("time")
        present_axes.append("dim")

        # If A is 1D (dim,), nothing to permute
        if A.ndim == 1:
            return A

        # Permute from present_axes order to the user's requested 'order'
        # BUT order may include all 2 or 3 axes; present_axes is in canonical [batch,time,dim] subset order
        src = present_axes
        dst = order[:]  # user's order (2D or 3D)

        # sanity: they should match as sets
        if set(src) != set(dst):
            raise RuntimeError(f"Internal restore mismatch: src={src}, dst={dst}")

        perm = [src.index(ax) for ax in dst]
        return np.transpose(A, axes=perm)

    if out_style[0] == "hinted":
        return _restore_obj_from_hint(out_int, out_style[1])

    # fallback restore (original)
    if out_style[0] == "single":
        return out_int[0, 0, :]
    if out_style[0] == "Mxdim":
        return out_int[:, 0, :]
    if out_style[0] == "dimxN":
        return out_int[0, :, :].T
    if out_style[0] == "Nxdim_time":
        return out_int[0, :, :]
    return out_int


def geo_eclip_to_geo_secr_generic(obj, earth, eps=1e-12, layout="auto",
                                 obj_hint=None, earth_hint=None):
    """
    Convert geocentric ECLIPJ2000 position(s) or state(s) to SECR (Earth-centered rotating).

    STRICT RULES:
    - obj dim = 3 → earth may be 3 or 6 (earth velocity ignored)
    - obj dim = 6 → earth MUST be 6 (otherwise ValueError)

    HINT SYSTEM (new, explicit structure):
      You can now explicitly define the axis order using tokens:
        - 'batch'   : M axis (multiple objects / spacecraft)
        - 'time'    : N axis (time series)
        - 'position': dim=3
        - 'state'   : dim=6
        - 3 or 6    : dim override (optional)

      Examples (all permutations supported):
        obj_hint=('batch','position')  -> (M,3)
        obj_hint=('position','batch')  -> (3,M)
        obj_hint=('batch','state')     -> (M,6)

        obj_hint=('time','state')      -> (N,6)
        obj_hint=('state','time')      -> (6,N)

        obj_hint=('time','state','batch') -> (N,6,M)
        obj_hint=('batch','time','state') -> (M,N,6)
        etc.

      Earth hints:
        Earth is treated as time-indexed only (no 'batch' axis supported):
          earth_hint=('time','state') -> (N,6)
          earth_hint=('state','time') -> (6,N)
          earth_hint=('time','position') -> (N,3)
          earth_hint=('position','time') -> (3,N)
        If earth is constant, pass (3,) or (6,) (hint optional).

    Returns: same layout as obj (including the hinted axis order if hint was used).
    """

    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    O = np.asarray(obj, dtype=float)
    E = np.asarray(earth, dtype=float)

    # -----------------------
    # Hint parsing: explicit axis order
    # -----------------------
    def _parse_struct_hint(h, name, allow_batch=True):
        """
        Returns (order, dim) where:
          order: list like ['batch','dim'] or ['time','dim','batch'] (axis order)
          dim: 3 or 6 or None
        """
        if h is None:
            return None, None

        # tokenize
        if isinstance(h, (tuple, list, set)):
            tokens = list(h)
        elif isinstance(h, str):
            s = h.strip()
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            tokens = [t.strip() for t in s.split(",") if t.strip()]
        else:
            raise ValueError(f"{name}_hint must be None, a tuple/list/set, or a string like '(time,state,batch)'")

        order = []
        dim = None

        def _add_axis(ax):
            if ax in order:
                raise ValueError(f"{name}_hint repeats axis '{ax}'. Got {h}.")
            order.append(ax)

        for t in tokens:
            # ints
            if isinstance(t, (int, np.integer)):
                iv = int(t)
                if iv in (3, 6):
                    dim = iv
                else:
                    raise ValueError(f"{name}_hint invalid dim {t} (use 3 or 6)")
                continue

            ts = str(t).strip().lower()

            if ts == "batch":
                if not allow_batch:
                    raise ValueError(f"{name}_hint may not include 'batch'. Earth cannot be batch-indexed here.")
                _add_axis("batch")
            elif ts == "time":
                _add_axis("time")
            elif ts in ("position", "pos"):
                if dim is not None and dim != 3:
                    raise ValueError(f"{name}_hint conflicts: both state(6) and position(3) implied.")
                dim = 3
                _add_axis("dim")
            elif ts in ("state", "st"):
                if dim is not None and dim != 6:
                    raise ValueError(f"{name}_hint conflicts: both position(3) and state(6) implied.")
                dim = 6
                _add_axis("dim")
            elif ts in ("3", "6"):
                dim = int(ts)
                # if user did NOT include position/state token, we still need a 'dim' axis
                if "dim" not in order:
                    _add_axis("dim")
            elif ts == "":
                continue
            else:
                raise ValueError(
                    f"{name}_hint token '{t}' unrecognized. Use 'batch','time','position','state',3,6."
                )

        if dim is not None and "dim" not in order:
            _add_axis("dim")

        if "dim" not in order:
            raise ValueError(f"{name}_hint must include 'position'/'state' (or 3/6). Got {h}.")

        if (dim not in (3, 6)):
            raise ValueError(f"{name}_hint must resolve dim to 3 or 6. Got {h}.")

        return order, dim

    obj_order_hint, obj_dim_hint = _parse_struct_hint(obj_hint, "obj", allow_batch=True)
    earth_order_hint, earth_dim_hint = _parse_struct_hint(earth_hint, "earth", allow_batch=False) if earth_hint is not None else (None, None)

    # -----------------------
    # Dim inference (fallback when no explicit hint)
    # -----------------------
    def infer_dim_fallback(A, name):
        if A.ndim == 1:
            if A.shape in [(3,), (6,)]:
                return A.shape[0]
            raise ValueError(f"{name} expected (3,) or (6,), got {A.shape}")

        if A.ndim == 2:
            r, c = A.shape
            r_is = r in (3, 6)
            c_is = c in (3, 6)
            if r_is and not c_is:
                return r
            if c_is and not r_is:
                return c
            if r_is and c_is:
                return 6 if (r == 6 or c == 6) else 3

        if A.ndim == 3 and A.shape[-1] in (3, 6):
            return A.shape[-1]

        raise ValueError(f"Could not infer dim for {name} with shape {A.shape}")

    obj_dim = obj_dim_hint if obj_dim_hint is not None else infer_dim_fallback(O, "obj")
    earth_dim = earth_dim_hint if earth_dim_hint is not None else infer_dim_fallback(E, "earth")

    # -----------------------
    # STRICT RULE
    # -----------------------
    if obj_dim == 6 and earth_dim == 3:
        raise ValueError(
            "Invalid input: obj is 6D state but earth is 3D position. "
            "Earth velocity is required to compute omega and the rotating-frame velocity correction."
        )

    # -----------------------
    # Normalize obj to internal (M,N,dim), and record how to restore
    # -----------------------
    def _normalize_obj_with_hint(A, order, dim):
        """
        Uses explicit axis order to convert A to (M,N,dim).
        Missing axes ('batch' or 'time') are treated as singleton.
        """
        if A.ndim == 1:
            if A.shape != (dim,):
                raise ValueError(f"obj expected ({dim},), got {A.shape}")
            A_int = A[None, None, :]
            restore = {"used_hint": True, "order": ["dim"], "orig_ndim": 1}
            return A_int, restore

        if A.ndim != len(order):
            raise ValueError(f"obj_hint implies {len(order)}D but obj has ndim={A.ndim}, shape={A.shape}")

        ax_dim = order.index("dim")
        ax_batch = order.index("batch") if "batch" in order else None
        ax_time  = order.index("time")  if "time" in order else None

        if A.shape[ax_dim] != dim:
            raise ValueError(f"obj dim axis length {A.shape[ax_dim]} does not match hinted dim={dim}.")

        target_axes_existing = []
        if ax_batch is not None:
            target_axes_existing.append(ax_batch)
        if ax_time is not None:
            target_axes_existing.append(ax_time)
        target_axes_existing.append(ax_dim)

        A_perm = np.transpose(A, axes=target_axes_existing)

        if ax_batch is not None and ax_time is not None:
            A_int = A_perm
        elif ax_batch is not None and ax_time is None:
            A_int = A_perm[:, None, :]
        elif ax_batch is None and ax_time is not None:
            A_int = A_perm[None, :, :]
        else:
            A_int = A_perm[None, None, :]

        restore = {"used_hint": True, "order": order, "orig_ndim": A.ndim}
        return A_int, restore

    def _choose_obj_mode_for_Kdim(K):
        if layout in ("batch", "time"):
            return layout
        earth_time_like = (
            E.ndim == 2 and (
                (E.shape[1] == earth_dim and E.shape[0] == K) or
                (E.shape[0] == earth_dim and E.shape[1] == K)
            )
        )
        return "time" if earth_time_like else "batch"

    if obj_order_hint is not None:
        O_int, obj_restore = _normalize_obj_with_hint(O, obj_order_hint, obj_dim)
        out_style = ("hinted", obj_restore)
    else:
        if O.ndim == 1:
            if O.shape != (obj_dim,):
                raise ValueError(f"obj expected ({obj_dim},), got {O.shape}")
            O_int = O[None, None, :]
            out_style = ("single",)

        elif O.ndim == 2:
            if O.shape == (obj_dim, 1):
                O_int = O[:, 0][None, None, :]
                out_style = ("single",)

            elif O.shape == (1, obj_dim):
                O_int = O[0, :][None, None, :]
                out_style = ("single",)

            elif O.shape[0] == obj_dim and O.shape[1] != obj_dim:  # (dim,N)
                O_int = O.T[None, :, :]
                out_style = ("dimxN",)

            elif O.shape[1] == obj_dim and O.shape[0] != obj_dim:  # (K,dim)
                K = O.shape[0]
                mode = _choose_obj_mode_for_Kdim(K)
                if mode == "time":
                    O_int = O[None, :, :]
                    out_style = ("Nxdim_time",)
                else:
                    O_int = O[:, None, :]
                    out_style = ("Mxdim",)

            elif O.shape[0] == obj_dim and O.shape[1] == obj_dim:
                raise ValueError(f"Ambiguous obj shape {O.shape}. Reshape explicitly or use obj_hint.")
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

    # -----------------------
    # Normalize earth to (N, earth_dim) (time-indexed only)
    # -----------------------
    def _normalize_earth_to_Nd(Earr, N, dim, order_hint=None):
        if order_hint is not None:
            if Earr.ndim == 1:
                if Earr.shape != (dim,):
                    raise ValueError(f"earth expected ({dim},), got {Earr.shape}")
                return np.repeat(Earr[None, :], N, axis=0)

            if Earr.ndim != len(order_hint):
                raise ValueError(f"earth_hint implies {len(order_hint)}D but earth has ndim={Earr.ndim}, shape={Earr.shape}")

            if "batch" in order_hint:
                raise ValueError("earth_hint may not include 'batch' in this function.")

            ax_dim = order_hint.index("dim")
            ax_time = order_hint.index("time") if "time" in order_hint else None

            if Earr.shape[ax_dim] != dim:
                raise ValueError(f"earth dim axis length {Earr.shape[ax_dim]} does not match hinted dim={dim}.")

            if ax_time is None:
                squeezed = np.squeeze(Earr)
                if squeezed.shape != (dim,):
                    raise ValueError(f"earth constant form must squeeze to ({dim},), got {squeezed.shape}")
                return np.repeat(squeezed[None, :], N, axis=0)

            E_td = np.transpose(Earr, axes=[ax_time, ax_dim])
            if E_td.shape[0] != N:
                raise ValueError(f"earth has N={E_td.shape[0]} but obj has N={N}")
            return E_td

        if Earr.ndim == 1:
            if Earr.shape != (dim,):
                raise ValueError(f"earth expected ({dim},), got {Earr.shape}")
            return np.repeat(Earr[None, :], N, axis=0)

        if Earr.ndim == 2:
            if Earr.shape == (dim, 1):
                return np.repeat(Earr[:, 0][None, :], N, axis=0)
            if Earr.shape == (1, dim):
                return np.repeat(Earr[0, :][None, :], N, axis=0)

            if Earr.shape[1] == dim and Earr.shape[0] == N:
                return Earr
            if Earr.shape[0] == dim and Earr.shape[1] == N:
                return Earr.T

            if Earr.shape[0] == dim and Earr.shape[1] == dim:
                raise ValueError(
                    f"Ambiguous earth shape {Earr.shape} with dim={dim}. Reshape explicitly or use earth_hint."
                )

        raise ValueError(f"earth unsupported shape {Earr.shape} for dim={dim} and N={N}")

    E_N = _normalize_earth_to_Nd(E, N, earth_dim, earth_order_hint)

    # For obj_dim==3, allow earth_dim==6 but ignore vel; for obj_dim==6 earth_dim==6 is guaranteed
    h_r_E = E_N[:, :3]                 # (N,3)
    h_v_E = E_N[:, 3:] if obj_dim == 6 else None

    # ---- unpack ECLIP obj ----
    r = O_int[:, :, :3]                # (M,N,3) in inertial ecliptic
    v = O_int[:, :, 3:] if obj_dim == 6 else None

    # angles from Earth-Sun direction (same convention as inverse function)
    angles = np.arctan2(-h_r_E[:, 1], -h_r_E[:, 0])  # (N,)

    c = np.cos(angles)
    s = np.sin(angles)

    # Rotation inertial ecliptic -> SECR: Rz(-angles)
    Rt = np.zeros((N, 3, 3), dtype=float)
    Rt[:, 0, 0] = c
    Rt[:, 0, 1] = s
    Rt[:, 1, 0] = -s
    Rt[:, 1, 1] = c
    Rt[:, 2, 2] = 1.0

    # rotating position: r' = Rt * r
    secr_r = np.einsum("nij,mnj->mni", Rt, r)  # (M,N,3)

    if obj_dim == 3:
        out_int = secr_r
    else:
        # omega magnitude from Earth motion (inertial)
        rE_norm2 = np.sum(h_r_E * h_r_E, axis=1)
        rE_norm2 = np.maximum(rE_norm2, eps)
        omega_mag = np.linalg.norm(np.cross(h_r_E, h_v_E), axis=1) / rE_norm2  # (N,)

        omega = np.zeros((N, 3), dtype=float)
        omega[:, 2] = omega_mag

        # omega in SECR coordinates (rotate inertial omega into SECR using Rt)
        omega_p = np.einsum("nij,nj->ni", Rt, omega)  # (N,3)

        # rotating velocity: v' = Rt*v - omega' x r'
        secr_v = np.einsum("nij,mnj->mni", Rt, v) - np.cross(omega_p[None, :, :], secr_r)

        out_int = np.concatenate([secr_r, secr_v], axis=2)  # (M,N,6)

    # -----------------------
    # Restore original layout
    # -----------------------
    def _restore_obj_from_hint(out_int_MNdim, restore):
        order = restore["order"]
        orig_ndim = restore["orig_ndim"]

        if orig_ndim == 1:
            return out_int_MNdim[0, 0, :]

        have_batch = "batch" in order
        have_time  = "time" in order

        A = out_int_MNdim

        if not have_batch:
            A = A[0, :, :]  # (N,dim)
        if not have_time:
            if have_batch:
                A = A[:, 0, :]  # (M,dim)
            else:
                A = A[0, :]     # (dim,)

        present_axes = []
        if have_batch:
            present_axes.append("batch")
        if have_time:
            present_axes.append("time")
        present_axes.append("dim")

        if A.ndim == 1:
            return A

        src = present_axes
        dst = order[:]

        if set(src) != set(dst):
            raise RuntimeError(f"Internal restore mismatch: src={src}, dst={dst}")

        perm = [src.index(ax) for ax in dst]
        return np.transpose(A, axes=perm)

    if out_style[0] == "hinted":
        return _restore_obj_from_hint(out_int, out_style[1])

    if out_style[0] == "single":
        return out_int[0, 0, :]
    if out_style[0] == "Mxdim":
        return out_int[:, 0, :]
    if out_style[0] == "dimxN":
        return out_int[0, :, :].T
    if out_style[0] == "Nxdim_time":
        return out_int[0, :, :]
    return out_int



def geo_eme_to_geo_eclip_generic(x, eps=1e-12, layout="auto", hint=None):
    """
    Convert geocentric EME/J2000 position(s) or state(s) to geocentric ECLIPJ2000.

    If x has 3 components -> treat as position, return position.
    If x has 6 components -> treat as full state, return full state.

    Supported x shapes (fallback, when no explicit hint):
      Position: (3,), (M,3), (N,3), (3,N), (M,N,3)
      State:    (6,), (M,6), (N,6), (6,N), (M,N,6)

    layout resolves ambiguity when x is (K,3) or (K,6) (fallback mode):
      - "batch": interpret as (M,dim) objects at one time (N=1)
      - "time" : interpret as (N,dim) time series for one object (M=1)
      - "auto" : default to "batch" (safer)

    NEW (explicit structure hint):
      You can explicitly define the axis order using tokens:
        - 'batch'   : M axis (multiple objects)
        - 'time'    : N axis (time series)
        - 'position': dim=3
        - 'state'   : dim=6
        - 3 or 6    : dim override (optional)

      Any permutation is accepted, and missing axes are treated as singleton.

      Examples:
        hint=('batch','position')   -> expects x shape (M,3)
        hint=('position','batch')   -> expects x shape (3,M)

        hint=('time','state')       -> expects x shape (N,6)
        hint=('state','time')       -> expects x shape (6,N)

        hint=('time','state','batch')  -> expects x shape (N,6,M)
        hint=('batch','time','state')  -> expects x shape (M,N,6)
        hint=('state','batch','time')  -> expects x shape (6,M,N)
        etc.

      Notes:
        - If you provide a hint, it is used strictly (shape must match its rank and dim axis size).
        - If you do NOT provide a hint, behavior matches your previous function:
            infer dim, then use layout/auto heuristics for (K,dim) 2D inputs.

    Returns: same layout as input x; if a hint was used, it returns in that hinted axis order.
    """
    if layout not in ("auto", "batch", "time"):
        raise ValueError("layout must be one of {'auto','batch','time'}")

    X = np.asarray(x, dtype=float)

    # -----------------------
    # Hint parsing: explicit axis order
    # -----------------------
    def _parse_struct_hint(h, name="hint"):
        """
        Returns (order, dim) where:
          order: list like ['batch','dim'] or ['time','dim','batch'] (axis order)
          dim: 3 or 6
        """
        if h is None:
            return None, None

        if isinstance(h, (tuple, list, set)):
            tokens = list(h)
        elif isinstance(h, str):
            s = h.strip()
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            tokens = [t.strip() for t in s.split(",") if t.strip()]
        else:
            raise ValueError(f"{name} must be None, a tuple/list/set, or a string like '(time,state,batch)'")

        order = []
        dim = None

        def _add_axis(ax):
            if ax in order:
                raise ValueError(f"{name} repeats axis '{ax}'. Got {h}.")
            order.append(ax)

        for t in tokens:
            if isinstance(t, (int, np.integer)):
                iv = int(t)
                if iv in (3, 6):
                    dim = iv
                    if "dim" not in order:
                        _add_axis("dim")
                else:
                    raise ValueError(f"{name} invalid dim {t} (use 3 or 6)")
                continue

            ts = str(t).strip().lower()

            if ts == "batch":
                _add_axis("batch")
            elif ts == "time":
                _add_axis("time")
            elif ts in ("position", "pos"):
                if dim is not None and dim != 3:
                    raise ValueError(f"{name} conflicts: both state(6) and position(3) implied.")
                dim = 3
                _add_axis("dim")
            elif ts in ("state", "st"):
                if dim is not None and dim != 6:
                    raise ValueError(f"{name} conflicts: both position(3) and state(6) implied.")
                dim = 6
                _add_axis("dim")
            elif ts in ("3", "6"):
                dim = int(ts)
                if "dim" not in order:
                    _add_axis("dim")
            elif ts == "":
                continue
            else:
                raise ValueError(
                    f"{name} token '{t}' unrecognized. Use 'batch','time','position','state',3,6."
                )

        if "dim" not in order:
            raise ValueError(f"{name} must include 'position'/'state' (or 3/6). Got {h}.")
        if dim not in (3, 6):
            raise ValueError(f"{name} must resolve dim to 3 or 6. Got {h}.")

        return order, dim

    hint_order, hint_dim = _parse_struct_hint(hint, "hint")

    # -----------------------
    # Dim inference (fallback when no explicit hint)
    # -----------------------
    def infer_dim_fallback(A):
        if A.ndim == 1 and A.shape in [(3,), (6,)]:
            return A.shape[0]

        if A.ndim == 2:
            r, c = A.shape
            r_is = r in (3, 6)
            c_is = c in (3, 6)
            if r_is and not c_is:
                return r
            if c_is and not r_is:
                return c
            if r_is and c_is:
                return 6 if (r == 6 or c == 6) else 3

        if A.ndim == 3 and A.shape[-1] in (3, 6):
            return A.shape[-1]

        raise ValueError(f"Input must be position (3) or state (6); got shape {A.shape}")

    dim = hint_dim if hint_dim is not None else infer_dim_fallback(X)

    # -----------------------
    # Normalize X to internal (M,N,dim) and remember how to restore
    # -----------------------
    def _normalize_with_hint(A, order, dim):
        """
        Strictly interpret A using explicit axis order -> return (M,N,dim)
        Missing 'batch' or 'time' axes are treated as singleton.
        Also returns restore_info to put output back into the same hinted order.
        """
        if A.ndim == 1:
            if A.shape != (dim,):
                raise ValueError(f"x expected ({dim},), got {A.shape}")
            A_int = A[None, None, :]
            restore = {"used_hint": True, "order": ["dim"], "orig_ndim": 1}
            return A_int, restore

        if A.ndim != len(order):
            raise ValueError(f"hint implies {len(order)}D but x has ndim={A.ndim}, shape={A.shape}")

        ax_dim = order.index("dim")
        ax_batch = order.index("batch") if "batch" in order else None
        ax_time  = order.index("time")  if "time"  in order else None

        if A.shape[ax_dim] != dim:
            raise ValueError(f"x dim axis length {A.shape[ax_dim]} does not match hinted dim={dim}.")

        # transpose existing axes into canonical [batch,time,dim] (dropping missing)
        axes = []
        if ax_batch is not None:
            axes.append(ax_batch)
        if ax_time is not None:
            axes.append(ax_time)
        axes.append(ax_dim)

        A_perm = np.transpose(A, axes=axes)

        # inject missing axes to make (M,N,dim)
        if ax_batch is not None and ax_time is not None:
            A_int = A_perm                       # (M,N,dim)
        elif ax_batch is not None and ax_time is None:
            A_int = A_perm[:, None, :]           # (M,1,dim)
        elif ax_batch is None and ax_time is not None:
            A_int = A_perm[None, :, :]           # (1,N,dim)
        else:
            # only 'dim' in order (shouldn't happen for nd>1 because we'd require dim + something),
            # but keep a safe path
            A_int = A_perm[None, None, :]

        restore = {"used_hint": True, "order": order, "orig_ndim": A.ndim}
        return A_int, restore

    def _choose_mode_for_Kdim(K):
        # Priority: layout if not auto; else default batch
        if layout in ("batch", "time"):
            return layout
        return "batch"

    if hint_order is not None:
        X_int, restore_info = _normalize_with_hint(X, hint_order, dim)
        out_style = ("hinted", restore_info)
    else:
        # ---- fallback normalization (matches your previous behavior) ----
        if X.ndim == 1:
            if X.shape != (dim,):
                raise ValueError(f"Expected ({dim},), got {X.shape}")
            X_int = X[None, None, :]
            out_style = ("single",)

        elif X.ndim == 2:
            if X.shape == (dim, 1):
                X_int = X[:, 0][None, None, :]
                out_style = ("single",)

            elif X.shape == (1, dim):
                X_int = X[0, :][None, None, :]
                out_style = ("single",)

            elif X.shape[0] == dim and X.shape[1] != dim:      # (dim,N)
                X_int = X.T[None, :, :]                         # (1,N,dim)
                out_style = ("dimxN",)

            elif X.shape[1] == dim and X.shape[0] != dim:      # (K,dim)
                K = X.shape[0]
                mode = _choose_mode_for_Kdim(K)
                if mode == "time":
                    X_int = X[None, :, :]                       # (1,N,dim)
                    out_style = ("Nxdim_time",)
                else:
                    X_int = X[:, None, :]                       # (M,1,dim)
                    out_style = ("Mxdim",)

            elif X.shape[0] == dim and X.shape[1] == dim:
                raise ValueError(
                    f"Ambiguous input shape {X.shape}. Reshape explicitly or pass hint."
                )
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

    # ecliptic->EME:
    #   R = [[1,0,0],[0,c,-s],[0,s,c]]
    # so EME->ecliptic = R^T:
    R_T = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=float,
    )

    r = X_int[:, :, :3]
    r_ecl = np.einsum("ij,mnj->mni", R_T, r)

    if dim == 3:
        out_int = r_ecl
    else:
        v = X_int[:, :, 3:]
        v_ecl = np.einsum("ij,mnj->mni", R_T, v)
        out_int = np.concatenate([r_ecl, v_ecl], axis=2)  # (M,N,6)

    # -----------------------
    # Restore output layout
    # -----------------------
    def _restore_from_hint(out_int_MNdim, restore):
        order = restore["order"]
        orig_ndim = restore["orig_ndim"]

        if orig_ndim == 1:
            return out_int_MNdim[0, 0, :]

        have_batch = "batch" in order
        have_time  = "time"  in order

        A = out_int_MNdim  # (M,N,dim)

        if not have_batch:
            A = A[0, :, :]         # (N,dim)
        if not have_time:
            if have_batch:
                A = A[:, 0, :]     # (M,dim)
            else:
                A = A[0, :]        # (dim,)

        # present axes are in canonical subset order:
        present = []
        if have_batch:
            present.append("batch")
        if have_time:
            present.append("time")
        present.append("dim")

        if A.ndim == 1:
            return A

        if set(present) != set(order):
            raise RuntimeError(f"Internal restore mismatch: present={present}, order={order}")

        perm = [present.index(ax) for ax in order]
        return np.transpose(A, axes=perm)

    if out_style[0] == "hinted":
        return _restore_from_hint(out_int, out_style[1])

    # fallback restore
    if out_style[0] == "single":
        return out_int[0, 0, :]
    if out_style[0] == "Mxdim":
        return out_int[:, 0, :]
    if out_style[0] == "dimxN":
        return out_int[0, :, :].T
    if out_style[0] == "Nxdim_time":
        return out_int[0, :, :]
    return out_int



def piecewise_anchor_and_propagate_spacecraft_trajs(
    *,
    formation,
    minimoon,
    timer,                     # needs curr_integration_index
    t_targets_jdtdb,            # (T,) JD TDB epochs you want states at
    n_body_propagator,

    # units
    au_km=149_597_870.700,      # km per AU (only used because table values are AU and AU/day)

    # SPICE config for Earth at LPF epoch
    earth_id=399,
    sun_id=10,
    frame_eclip="ECLIPJ2000",

    strict_bounds=True,

    # NEW: return metadata about anchor usage
    return_anchor_info=False,
):
    """
    Piecewise anchored propagation using quasi-halo reference rows.

    Index/anchor logic (IMPORTANT, fixes off-by-one):
      - Start index is k0 = timer.curr_integration_index
      - Anchor epochs come from the minimoon hourly grid: jd_grid[k]
      - Targets are assigned to anchors RELATIVE to k0:
            dt_hour = jd_grid[k0+1] - jd_grid[k0]
            steps_after = floor((t - jd_grid[k0]) / dt_hour)
            k = k0 + steps_after
        so the first anchor is always k0 for targets near the start.

    Returns:
      - if return_anchor_info == False:
            out: (T, M, 6) GEO-EME states in km, km/s
      - if return_anchor_info == True:
            (out, anchor_info) where anchor_info is a dict containing:
              * anchor_k_of_t: (T,) anchor index used for each target epoch
              * anchor_epoch_of_t: (T,) anchor JD for each target epoch
              * anchors_used_k: (A,) sorted unique anchor indices actually used
              * anchors_used_epoch: (A,) JDs of anchors_used_k
              * groups: dict[int, list[int]] mapping anchor k -> list of target indices
    """

    # -------------------------
    # Inputs
    # -------------------------
    t_targets = np.asarray(t_targets_jdtdb, dtype=float).ravel()
    T = int(t_targets.shape[0])
    M = int(len(formation.spacecraft))

    if T == 0:
        out = np.zeros((0, M, 6), dtype=float)
        if return_anchor_info:
            anchor_info = {
                "anchor_k_of_t": np.zeros((0,), dtype=int),
                "anchor_epoch_of_t": np.zeros((0,), dtype=float),
                "anchors_used_k": np.zeros((0,), dtype=int),
                "anchors_used_epoch": np.zeros((0,), dtype=float),
                "groups": {},
            }
            return out, anchor_info
        return out

    jd_grid = np.asarray(minimoon.orbit["Julian Date"], dtype=float).ravel()
    if jd_grid.ndim != 1 or jd_grid.size < 2:
        raise ValueError("minimoon.orbit['Julian Date'] must be 1D with >=2 entries")

    k0 = int(timer.curr_integration_index)
    if strict_bounds and not (0 <= k0 < jd_grid.size - 1):
        raise ValueError(
            f"timer.curr_integration_index={k0} out of bounds for jd_grid size {jd_grid.size}"
        )

    # grid-derived hour step (robust; avoids assuming exactly 1/24)
    dt_hour = float(jd_grid[k0 + 1] - jd_grid[k0])
    if dt_hour <= 0:
        raise ValueError("jd_grid must be strictly increasing (dt_hour <= 0 detected)")

    t0_grid = float(jd_grid[k0])

    # -------------------------
    # Earth helio-eclip from TABLE (AU, AU/day)
    # -------------------------
    earth_cols = [
        "Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)",
        "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)",
    ]
    for c in earth_cols:
        if c not in minimoon.orbit.columns:
            raise KeyError(f"minimoon.orbit missing required Earth ephemeris column '{c}'")

    earth_table_au_aud = np.asarray(minimoon.orbit[earth_cols], dtype=float)  # (N,6)

    AU_KM = float(au_km)
    AU_PER_DAY_TO_KMPS = AU_KM / 86400.0

    def _state_au_aud_to_kms(x6):
        x6 = np.asarray(x6, dtype=float).reshape(6,)
        y = x6.copy()
        y[:3] *= AU_KM
        y[3:] *= AU_PER_DAY_TO_KMPS
        return y

    def _earth_ast_from_table_at_index_kms(k):
        return _state_au_aud_to_kms(earth_table_au_aud[int(k)])

    # -------------------------
    # Time conversions (LPF Earth only)
    # -------------------------
    def _timestamp_to_et(ts):
        # ts is expected to be a pandas.Timestamp or datetime-like
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        s = ts.strftime("%Y-%m-%dT%H:%M:%S")
        return float(spice.utc2et(s))

    def _earth_lpf_from_spice_kms(et):
        st, _lt = spice.spkgeo(int(earth_id), float(et), frame_eclip, int(sun_id))
        return np.asarray(st, dtype=float).reshape(6,)

    # -------------------------
    # SC row -> GEO-EME state (km, km/s)
    # -------------------------
    def _sc_row_geo_eme_state_kms(row):
        # columns labeled km/km/s but actually AU and AU/day
        x_au = np.array(
            [float(row["GEO_EME_X_(km)"]), float(row["GEO_EME_Y_(km)"]), float(row["GEO_EME_Z_(km)"])],
            dtype=float,
        )
        v_au_per_day = np.array(
            [float(row["GEO_EME_Vx_(km/s)"]), float(row["GEO_EME_Vy_(km/s)"]), float(row["GEO_EME_Vz_(km/s)"])],
            dtype=float,
        )
        return np.concatenate([x_au * AU_KM, v_au_per_day * AU_PER_DAY_TO_KMPS], axis=0)

    # -------------------------
    # Re-epoch mapping: LPF -> SECR(using SPICE Earth) -> AST(using TABLE Earth)
    # -------------------------
    def _reepoch_sc_state_to_anchor_epoch(*, sc_row, earth_ast_kms):
        """
        GEO-EME(lpf) -> GEO-ECLIP(lpf)
        GEO-ECLIP(lpf) -> GEO-SECR(lpf)   using Earth_lpf from SPICE
        GEO-SECR(lpf)  -> GEO-ECLIP(ast)  using Earth_ast from TABLE
        GEO-ECLIP(ast) -> GEO-EME(ast)
        """
        et_lpf = _timestamp_to_et(sc_row["Time"])
        earth_lpf_kms = _earth_lpf_from_spice_kms(et_lpf)
        earth_ast_kms = np.asarray(earth_ast_kms, dtype=float).reshape(6,)

        x_eme_lpf = _sc_row_geo_eme_state_kms(sc_row)

        # GEO-EME -> GEO-ECLIP
        try:
            x_ecl_lpf = geo_eme_to_geo_eclip_generic(x_eme_lpf, hint=("state",))
        except TypeError:
            x_ecl_lpf = geo_eme_to_geo_eclip_generic(x_eme_lpf)

        # GEO-ECLIP -> GEO-SECR at LPF (Earth from SPICE)
        x_secr = geo_eclip_to_geo_secr_generic(
            x_ecl_lpf, earth_lpf_kms,
            obj_hint=("state",),
            earth_hint=("state",),
        )

        # GEO-SECR -> GEO-ECLIP at AST (Earth from TABLE)
        x_ecl_ast = geo_secr_to_geo_eclip_generic(
            x_secr, earth_ast_kms,
            obj_hint=("state",),
            earth_hint=("state",),
        )

        # GEO-ECLIP -> GEO-EME
        try:
            _x_eme_ast = geo_eclip_to_geo_eme_generic(x_ecl_ast, hint=("state",))
        except TypeError:
            _x_eme_ast = geo_eclip_to_geo_eme_generic(x_ecl_ast)

        # NOTE: matches your original behavior: return ECL-at-ast (not EME)
        return np.asarray(x_ecl_ast, dtype=float).reshape(6,)

    # -------------------------
    # Assign each target to an anchor index k RELATIVE to k0 (fixes off-by-one)
    # -------------------------
    eps = 1e-12
    steps_after = np.floor((t_targets - t0_grid) / dt_hour + eps).astype(int)
    anchor_k_of_t = k0 + steps_after

    if strict_bounds:
        if np.any(anchor_k_of_t < 0) or np.any(anchor_k_of_t >= jd_grid.size):
            bad = np.where((anchor_k_of_t < 0) | (anchor_k_of_t >= jd_grid.size))[0][:10]
            raise ValueError(
                "Some requested epochs fall outside minimoon.orbit grid when anchored at curr_integration_index. "
                f"Example bad target indices: {bad}, times: {t_targets[bad]}, anchor_k_of_t: {anchor_k_of_t[bad]}"
            )
        if np.any(steps_after < 0):
            bad = np.where(steps_after < 0)[0][:10]
            raise ValueError(
                "Some requested epochs are earlier than the anchor grid time jd_grid[k0]. "
                f"Example indices: {bad}, times: {t_targets[bad]}, dt_sec: {(t_targets[bad]-t0_grid)*86400.0}"
            )
    else:
        anchor_k_of_t = np.clip(anchor_k_of_t, 0, jd_grid.size - 1)

    anchor_epoch_of_t = jd_grid[anchor_k_of_t]

    # group target indices by anchor index k
    groups = {}
    for ti, k in enumerate(anchor_k_of_t.tolist()):
        groups.setdefault(int(k), []).append(int(ti))

    # -------------------------
    # Propagate per-group and stitch
    # -------------------------
    out = np.full((T, M, 6), np.nan, dtype=float)
    all_x0m = []

    for k, idx_list in groups.items():
        idx = np.asarray(idx_list, dtype=int)

        # anchor epoch from table
        t_anchor = float(jd_grid[k])

        # Earth at anchor from TABLE row k (converted to km/km/s)
        earth_ast_kms = _earth_ast_from_table_at_index_kms(k)

        # targets in this group, sorted for propagation
        t_sub = t_targets[idx]
        order = np.argsort(t_sub)
        t_sub_sorted = t_sub[order]
        idx_sorted = idx[order]

        # build initial states at anchor (M,6) from row k
        x0_M6 = np.zeros((M, 6), dtype=float)
        for m, sc in enumerate(formation.spacecraft):
            row = sc.matched_trajectory_full.iloc[k]
            x0_M6[m, :] = _reepoch_sc_state_to_anchor_epoch(
                sc_row=row,
                earth_ast_kms=earth_ast_kms,
            )

        all_x0m.append(x0_M6)

        traj = np.asarray(
            n_body_propagator.propagate_multiple_objects(x0_M6, t_anchor, t_sub_sorted),
            dtype=float,
        )

        # ---- normalize traj to (len(t_sub_sorted), M, 6) ----
        Tsub = int(len(t_sub_sorted))

        if traj.ndim == 3:
            if traj.shape == (Tsub, M, 6):
                traj_T_M_6 = traj
            elif traj.shape == (M, Tsub, 6):
                traj_T_M_6 = np.transpose(traj, (1, 0, 2))
            else:
                raise ValueError(
                    f"propagate_multiple_objects returned unexpected 3D shape {traj.shape}; "
                    f"expected ({Tsub},{M},6) or ({M},{Tsub},6)"
                )

        elif traj.ndim == 2:
            # common when Tsub == 1
            if Tsub != 1:
                raise ValueError(
                    f"propagate_multiple_objects returned 2D shape {traj.shape} but Tsub={Tsub} (expected 3D)."
                )

            if traj.shape == (M, 6):
                traj_T_M_6 = traj.reshape(1, M, 6)
            elif traj.shape == (6, M):
                traj_T_M_6 = traj.T.reshape(1, M, 6)
            elif traj.shape == (1, 6) and M == 1:
                traj_T_M_6 = traj.reshape(1, 1, 6)
            elif traj.shape == (6, 1) and M == 1:
                traj_T_M_6 = traj.T.reshape(1, 1, 6)
            else:
                raise ValueError(
                    f"propagate_multiple_objects returned unexpected 2D shape {traj.shape} for Tsub=1; "
                    f"expected ({M},6) (or (6,{M}) if transposed)."
                )

        elif traj.ndim == 1:
            # possible when M==1 and Tsub==1
            if Tsub == 1 and traj.shape == (6,) and M == 1:
                traj_T_M_6 = traj.reshape(1, 1, 6)
            else:
                raise ValueError(
                    f"propagate_multiple_objects returned unexpected 1D shape {traj.shape}; "
                    f"expected (6,) only when M==1 and Tsub==1."
                )

        else:
            raise ValueError(
                f"propagate_multiple_objects returned traj with ndim={traj.ndim}, shape={traj.shape}, unsupported."
            )

        out[idx_sorted, :, :] = traj_T_M_6

    # -------------------------
    # Optional debug plot of anchors
    # -------------------------
    plot_graphs = False
    if plot_graphs:
        all_x0m = np.asarray(all_x0m, dtype=float)
        try:
            fig = plt.figure(figsize=(9.5, 7.5))
            ax3d = fig.add_subplot(111, projection="3d")
            for i in range(M):
                ax3d.plot(all_x0m[:, i, 0], all_x0m[:, i, 1], all_x0m[:, i, 2])
            plt.show()
        except Exception:
            # don't fail if running headless
            pass

    if return_anchor_info:
        anchors_used_k = np.array(sorted(groups.keys()), dtype=int)
        anchors_used_epoch = np.array([float(jd_grid[k]) for k in anchors_used_k], dtype=float)
        anchor_info = {
            "anchor_k_of_t": np.asarray(anchor_k_of_t, dtype=int),
            "anchor_epoch_of_t": np.asarray(anchor_epoch_of_t, dtype=float),
            "anchors_used_k": anchors_used_k,
            "anchors_used_epoch": anchors_used_epoch,
            "groups": groups,
        }
        return out, anchor_info

    return out


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
        geo_eclip_state = geo_eme_to_geo_eclip_generic(geo_eme_state)
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
        formation.match_spacecraft_trajectory_full(int(detection['total_length']), config)

        # 2) Detecting spacecraft ID (1-based, from MultiIndex third level)
        sc_id_detect = int(detection.name[2])
        detecting_ids.append(sc_id_detect)
        detecting_spacecraft = formation.spacecraft[sc_id_detect - 1]

        # sample index of first detection for this row
        idx0 = int(detection['min_nonnegative'])

        geo_eme_state_detect = detecting_spacecraft.matched_trajectory_full.loc[idx0, eme_cols].to_numpy(dtype=float)
        geo_eme_state_detect[:3] *= (config['AU_TO_M'] / 1000.0)
        geo_eme_state_detect[3:] *= (config['AU_TO_M'] / 1000.0 / config['SECONDS_PER_DAY'])
        sc_time = detecting_spacecraft.matched_trajectory_full.loc[idx0, 'Time']

        # Your code put GEO_ECLIP in the DF; keep doing that for compatibility
        geo_eclip_state_detect = geo_eme_to_geo_eclip_generic(geo_eme_state_detect)
        det_geo_eclip_states.append(geo_eclip_state_detect)

        sc_epochs.append(sc_time.strftime("%Y-%m-%d %H:%M:%S"))

        # 6) Now expand the *same* matching logic to EVERY spacecraft
        sc_states_geo_eclip = []  # will be (num_sc, 6), ordered by spacecraft ID (1..N)
        sc_boresights = []
        fig = plt.figure()
        for jdx, sc in enumerate(formation.spacecraft, start=1):
            geo_eme_state_detect_j = sc.matched_trajectory_full.loc[idx0, eme_cols].to_numpy(
                dtype=float)

            full = sc.matched_trajectory_full.loc[:, eme_cols].to_numpy(
                dtype=float)
            full[:, :3] *= (config['AU_TO_M'] / 1000.0)
            full[:, 3:] *= (config['AU_TO_M'] / 1000.0 / config['SECONDS_PER_DAY'])
            plt.plot(full[:, 0], full[:, 1])

            geo_eme_state_detect_j[:3] *= (config['AU_TO_M'] / 1000.0)
            geo_eme_state_detect_j[3:] *= (config['AU_TO_M'] / 1000.0 / config['SECONDS_PER_DAY'])

            # Your code put GEO_ECLIP in the DF; keep doing that for compatibility
            geo_eclip_state_detect_j = geo_eme_to_geo_eclip_generic(geo_eme_state_detect_j)

            sc_states_geo_eclip.append(geo_eclip_state_detect_j)
            sc_boresights.append(sc.boresight)

        all_sc_geo_eclip = np.vstack(sc_states_geo_eclip)  # (num_sc, 6)
        all_sc_geo_eclip_list.append(all_sc_geo_eclip)
        plt.show()

        ####################
        # the reconstruction of the s/c formation seems off
        ####################

        sc_helio_se_kms = np.vstack(sc_states_geo_eclip).copy()
        fig = plt.figure()
        for i, sc in enumerate(sc_helio_se_kms):
            plt.scatter(sc[0], sc[1], label=f'{i}')
        plt.scatter(geo_eclip_state_detect[0], geo_eclip_state_detect[1], color='black')
        plt.legend()
        plt.axis('equal')
        plt.show()

        all_sc_boresights = np.vstack(sc_boresights)  # (num_sc, 3)
        all_sc_boresights_list.append(all_sc_boresights)

    # 7) Write the detecting s/c info back into the DataFrame (no chained assignment)
    out_df.loc[:, 'sc_epoch'] = sc_epochs

    det_geo_eclip = np.vstack(det_geo_eclip_states)
    out_df.loc[:, ['GEO_ECLIP_X_(km)', 'GEO_ECLIP_Y_(km)', 'GEO_ECLIP_Z_(km)',
                   'GEO_ECLIP_Vx_(km/s)', 'GEO_ECLIP_Vy_(km/s)', 'GEO_ECLIP_Vz_(km/s)']] = det_geo_eclip

    # Return: augmented DF, per-row all-s/c GEO_EME arrays, and detecting IDs
    return out_df, all_sc_geo_eclip_list, detecting_ids, all_sc_boresights_list


def get_scs_initial_states_new(detected_pop, config):
    """
    For each detection row in `detected_pop`:

      - Rebuild formation from saved SC1 initial position
      - Match trajectories to `total_length`
      - At the first detection instant idx0 = min_nonnegative:
          * For EVERY spacecraft j (ordered by ID 1..num_sc):
              - store epoch (Time) in column:  SC{j}_epoch
              - store GEO_ECLIP state (km, km/s) in column: SC{j}_GEO_ECLIP_state  (np.ndarray shape (6,))
              - store boresight in column:     SC{j}_boresight      (np.ndarray shape (3,))
          * Also store detecting spacecraft:
              - detecting_id
              - sc_epoch (same as SC{detecting_id}_epoch)
              - GEO_ECLIP_* columns (same as SC{detecting_id}_GEO_ECLIP_state split)

    Returns:
      out_df (augmented), with one row per detection instance (same as input indexing).
    """

    out_df = detected_pop.copy(deep=True)

    eme_cols = [
        "GEO_EME_X_(km)", "GEO_EME_Y_(km)", "GEO_EME_Z_(km)",
        "GEO_EME_Vx_(km/s)", "GEO_EME_Vy_(km/s)", "GEO_EME_Vz_(km/s)"
    ]

    # Unit conversions (assuming matched_trajectory_full stores AU and AU/day, even if column names say km/km/s)
    AU_km = config["AU_TO_M"] / 1000.0
    SEC_PER_DAY = float(config["SECONDS_PER_DAY"])

    def au_auday_state_to_km_kms(state6):
        """Convert [AU, AU/day] -> [km, km/s]. Accepts shape (6,) or (...,6)."""
        x = np.asarray(state6, dtype=float).copy()
        x[..., :3] *= AU_km
        x[..., 3:] *= (AU_km / SEC_PER_DAY)
        return x

    # Per-row outputs we always write
    detecting_ids = []
    det_epochs = []
    det_geo_eclip_states = []

    # Per-spacecraft columns (initialized after we know num_sc)
    sc_epoch_cols = {}      # SC{j}_epoch -> list[str]
    sc_state_cols = {}      # SC{j}_GEO_ECLIP_state -> list[np.ndarray(6,)]
    sc_boresight_cols = {}  # SC{j}_boresight -> list[np.ndarray(3,)]
    num_sc = None

    for _, detection in out_df.iterrows():
        # --- recreate formation from saved SC1 initial pos ---
        formation = Formation(config)
        sc1_ini_index = formation.get_index_from_pos(detection["spacecraft_1_ini_pos"])
        formation.recall_formation(sc1_ini_index, config)

        total_length = int(detection["total_length"])
        formation.match_spacecraft_trajectory(total_length, config)
        formation.match_spacecraft_trajectory_full(total_length, config)

        # infer num_sc once (assumed constant across rows)
        if num_sc is None:
            num_sc = len(formation.spacecraft)
            for j in range(1, num_sc + 1):
                sc_epoch_cols[f"SC{j}_epoch"] = []
                sc_state_cols[f"SC{j}_GEO_ECLIP_state"] = []
                sc_boresight_cols[f"SC{j}_boresight"] = []

        # detecting spacecraft id (1-based, from MultiIndex third level)
        sc_id_detect = int(detection.name[2])
        detecting_ids.append(sc_id_detect)

        # first detection sample index
        idx0 = int(detection["index_used"])

        # --- compute per-spacecraft epochs + states + boresights at idx0 ---
        for j, sc in enumerate(formation.spacecraft, start=1):
            traj_full = sc.matched_trajectory_full

            # epoch at idx0 (store as formatted string for df friendliness)
            t = traj_full.iloc[idx0]["Time"]
            sc_epoch_cols[f"SC{j}_epoch"].append(t.strftime("%Y-%m-%d %H:%M:%S"))

            # boresight (store as length-3 ndarray)
            sc_boresight_cols[f"SC{j}_boresight"].append(np.asarray(sc.boresight, dtype=float).reshape(3,))

            # GEO_EME state at idx0 (AU, AU/day) -> convert -> GEO_ECLIP (km, km/s)
            geo_eme_au_auday = traj_full.iloc[idx0][eme_cols].to_numpy(dtype=float)  # (6,)
            geo_eme_km_kms = au_auday_state_to_km_kms(geo_eme_au_auday)
            geo_eclip_km_kms = geo_eme_to_geo_eclip_generic(geo_eme_km_kms)                 # (6,)

            sc_state_cols[f"SC{j}_GEO_ECLIP_state"].append(geo_eclip_km_kms)

        # detecting spacecraft epoch/state for compatibility columns
        det_epochs.append(sc_epoch_cols[f"SC{sc_id_detect}_epoch"][-1])
        det_state = sc_state_cols[f"SC{sc_id_detect}_GEO_ECLIP_state"][-1]
        det_geo_eclip_states.append(det_state)

    # --- write per-sc columns into out_df ---
    out_df.loc[:, "detecting_id"] = detecting_ids

    for col, values in sc_epoch_cols.items():
        out_df.loc[:, col] = values

    for col, values in sc_state_cols.items():
        out_df.loc[:, col] = values

    for col, values in sc_boresight_cols.items():
        out_df.loc[:, col] = values

    return out_df



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


def fov_deg2_to_half_angle_rad(FOV_deg2):
    """
    Convert sky area FOV (deg^2) to cone half-angle (radians)
    using spherical cap geometry.
    """
    return np.arccos(
        1.0 - (FOV_deg2 / (180.0 / np.pi) ** 2) / (2.0 * np.pi)
    )


def plot_attcoord_costs_from_series(result_kcoverage_series, *, title="Attitude coordination costs"):
    """
    Plot J_t and objective/cost across all tested epochs.

    Expects result_kcoverage_series: list[dict] where each dict has:
      - dt (float)
      - feasible (bool)
      - J (float)  # dual coverage score
      - cost (float)  # objective = -J + penalty (per your optimizer)
    Infeasible rows may have J/cost = NaN and feasible=False.
    """
    if result_kcoverage_series is None or len(result_kcoverage_series) == 0:
        print("No k-coverage series data to plot.")
        return None, None

    dts = np.array([row.get("dt", np.nan) for row in result_kcoverage_series], dtype=float)
    feasible = np.array([bool(row.get("feasible", False)) for row in result_kcoverage_series], dtype=bool)

    J_vals = np.array([row.get("J", np.nan) for row in result_kcoverage_series], dtype=float)
    cost_vals = np.array([row.get("cost", np.nan) for row in result_kcoverage_series], dtype=float)

    # Best epoch = min cost among feasible
    best_idx = None
    if np.any(feasible):
        idxs = np.where(feasible)[0]
        best_idx = idxs[np.nanargmin(cost_vals[idxs])]

    fig, ax = plt.subplots(figsize=(8, 4))

    line1, = ax.plot(dts, J_vals, marker="o", linestyle="-", label=r"$J_t$")
    line2, = ax.plot(dts, -cost_vals, marker="x", linestyle="--", label="-(objective)")

    # Mark infeasible epochs
    if np.any(~feasible):
        ax.scatter(dts[~feasible], np.zeros(np.sum(~feasible)), marker="v", label="infeasible")

    # Mark best
    if best_idx is not None:
        ax.axvline(dts[best_idx], linestyle=":", linewidth=2, label=f"best dt={dts[best_idx]:.2f}s")

    ax.set_xlabel(r"Epoch time $\Delta t_s$ [s]")
    ax.set_ylabel(r"$J_t$ and -(objective)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    # Reasonable y-limits (ignore NaNs)
    y_all = np.concatenate([J_vals[np.isfinite(J_vals)], (-cost_vals)[np.isfinite(cost_vals)]])
    if y_all.size > 0:
        ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
        pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()

    return fig, ax


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


def plot_matched_trajectory_full_range(
    *,
    formation,
    # selection (choose one)
    idx_start=None,
    idx_stop=None,              # inclusive if inclusive_stop=True else python-slice style
    time_start=None,            # datetime-like or pandas Timestamp
    time_stop=None,             # datetime-like or pandas Timestamp

    inclusive_stop=True,

    # what to plot
    frame="GEO_EME",            # "GEO_EME" or "GEO_ECLIP" (must exist as columns)
    plot_3d=False,              # if True: one 3D axis; else: 3 subplots x(t),y(t),z(t)
    plot_xy=True,               # if not plot_3d: add an XY trajectory subplot (extra)
    use_time_axis=True,         # x-axis is Time if available; else index
    units="auto",               # "auto" | "au" | "km" (your tables are AU, AU/day even if labeled km)
    au_km=149_597_870.700,

    # styling
    show_markers=True,
    mark_every=10,
    linewidth=1.2,
    alpha=0.9,
    title=None,
    legend=True,
):
    """
    Plot a desired range of each spacecraft's matched_trajectory_full from a formation object.

    Assumes each spacecraft has:
      sc.matched_trajectory_full : pandas DataFrame with columns like:
        Time
        GEO_EME_X_(km), GEO_EME_Y_(km), GEO_EME_Z_(km)
        GEO_ECLIP_X_(km), GEO_ECLIP_Y_(km), GEO_ECLIP_Z_(km)
      (Despite labels, you told me these are AU; this function can convert to km.)

    Selection:
      - By index: idx_start/idx_stop
      - OR by time: time_start/time_stop (uses the 'Time' column)
    """

    # -------------------------
    # helpers
    # -------------------------
    def _get_cols(prefix):
        x = f"{prefix}_X_(km)"
        y = f"{prefix}_Y_(km)"
        z = f"{prefix}_Z_(km)"
        return x, y, z

    def _coerce_ts(x):
        if x is None:
            return None
        return pd.to_datetime(x)

    def _slice_df(df):
        # Prefer time slicing if time_start/stop provided
        if (time_start is not None) or (time_stop is not None):
            if "Time" not in df.columns:
                raise KeyError("time_start/time_stop provided but matched_trajectory_full has no 'Time' column.")
            ts0 = _coerce_ts(time_start)
            ts1 = _coerce_ts(time_stop)

            tt = pd.to_datetime(df["Time"])
            mask = np.ones(len(df), dtype=bool)
            if ts0 is not None:
                mask &= (tt >= ts0)
            if ts1 is not None:
                mask &= (tt <= ts1) if inclusive_stop else (tt < ts1)
            return df.loc[mask].copy()

        # else index slicing
        if idx_start is None and idx_stop is None:
            return df.copy()

        i0 = 0 if idx_start is None else int(idx_start)
        if idx_stop is None:
            return df.iloc[i0:].copy()
        i1 = int(idx_stop)
        if inclusive_stop:
            return df.iloc[i0 : i1 + 1].copy()
        return df.iloc[i0 : i1].copy()

    def _infer_units_scale(df_xyz):
        # If user says "auto", try to guess if values look like AU (~1e-2 here) vs km (~1e6+)
        # Your examples are ~0.008, so AU.
        # We'll base on median norm.
        v = np.asarray(df_xyz, float)
        med = float(np.nanmedian(np.linalg.norm(v, axis=1))) if v.size else 0.0
        if med == 0.0 or not np.isfinite(med):
            return 1.0, "raw"
        # Heuristic: if med < 1e3 -> likely AU; if med > 1e5 -> likely km
        if med < 1e3:
            return float(au_km), "km"
        return 1.0, "km"

    # -------------------------
    # validate spacecraft + columns
    # -------------------------
    if not hasattr(formation, "spacecraft"):
        raise AttributeError("formation must have attribute 'spacecraft' (list of spacecraft objects).")

    xcol, ycol, zcol = _get_cols(frame)
    AU_KM = float(au_km)

    # -------------------------
    # figure layout
    # -------------------------
    if plot_3d:
        fig = plt.figure(figsize=(9.5, 7.5))
        ax3d = fig.add_subplot(111, projection="3d")
        axes = (ax3d,)
    else:
        # x/y/z time series
        nrows = 4 if plot_xy else 3
        fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(11, 9), sharex=True)
        axs = np.atleast_1d(axs)
        axes = tuple(axs)

    # -------------------------
    # plotting loop
    # -------------------------
    for m, sc in enumerate(formation.spacecraft):
        if not hasattr(sc, "matched_trajectory_full"):
            raise AttributeError(f"spacecraft[{m}] has no matched_trajectory_full.")

        df = sc.matched_trajectory_full
        if not hasattr(df, "columns"):
            raise TypeError(f"spacecraft[{m}].matched_trajectory_full is not a DataFrame-like object.")

        for c in (xcol, ycol, zcol):
            if c not in df.columns:
                raise KeyError(
                    f"spacecraft[{m}].matched_trajectory_full missing column '{c}'. "
                    f"Available columns include: {list(df.columns)[:20]} ..."
                )

        dfr = _slice_df(df)
        if len(dfr) == 0:
            continue

        xyz = dfr[[xcol, ycol, zcol]].to_numpy(dtype=float)

        # units handling
        if units == "au":
            scale = 1.0
            yunit = "AU"
        elif units == "km":
            scale = AU_KM
            yunit = "km"
        else:
            # auto: infer; but for your case this will choose km conversion
            scale, yunit = _infer_units_scale(xyz)

        xyzp = xyz * float(scale)

        # x-axis
        if use_time_axis and ("Time" in dfr.columns):
            t = pd.to_datetime(dfr["Time"])
            tx = t
            xlabel = "Time"
        else:
            tx = dfr.index.to_numpy()
            xlabel = "Index"

        label = getattr(sc, "name", None)
        if label is None:
            label = f"SC{m+1}"

        if plot_3d:
            ax = axes[0]
            ax.plot(xyzp[:, 0], xyzp[:, 1], xyzp[:, 2], lw=linewidth, alpha=alpha, label=label)
            if show_markers:
                ax.scatter(
                    xyzp[::max(1, int(mark_every)), 0],
                    xyzp[::max(1, int(mark_every)), 1],
                    xyzp[::max(1, int(mark_every)), 2],
                    s=10,
                    alpha=min(1.0, alpha),
                )
        else:
            axx, axy, axz = axes[0], axes[1], axes[2]
            axx.plot(tx, xyzp[:, 0], lw=linewidth, alpha=alpha, label=label)
            axy.plot(tx, xyzp[:, 1], lw=linewidth, alpha=alpha, label=label)
            axz.plot(tx, xyzp[:, 2], lw=linewidth, alpha=alpha, label=label)

            if show_markers:
                step = max(1, int(mark_every))
                axx.scatter(tx[::step], xyzp[::step, 0], s=10, alpha=min(1.0, alpha))
                axy.scatter(tx[::step], xyzp[::step, 1], s=10, alpha=min(1.0, alpha))
                axz.scatter(tx[::step], xyzp[::step, 2], s=10, alpha=min(1.0, alpha))

            if plot_xy:
                axxy = axes[3]
                axxy.plot(xyzp[:, 0], xyzp[:, 1], lw=linewidth, alpha=alpha, label=label)
                if show_markers:
                    axxy.scatter(xyzp[::step, 0], xyzp[::step, 1], s=10, alpha=min(1.0, alpha))

    # -------------------------
    # labels / titles
    # -------------------------
    if plot_3d:
        ax = axes[0]
        ax.set_xlabel(f"{frame} X [{yunit}]")
        ax.set_ylabel(f"{frame} Y [{yunit}]")
        ax.set_zlabel(f"{frame} Z [{yunit}]")
        ax.grid(alpha=0.25)
        if title is None:
            title = f"{frame} matched_trajectory_full (3D)"
        ax.set_title(title)
        if legend:
            ax.legend(loc="best")
    else:
        axes[0].set_ylabel(f"X [{yunit}]")
        axes[1].set_ylabel(f"Y [{yunit}]")
        axes[2].set_ylabel(f"Z [{yunit}]")
        axes[-1].set_xlabel(xlabel)

        if plot_xy:
            axes[3].set_xlabel(f"X [{yunit}]")
            axes[3].set_ylabel(f"Y [{yunit}]")
            axes[3].set_title(f"{frame} XY trajectory")
            axes[3].grid(alpha=0.25)

        axes[0].grid(alpha=0.25)
        axes[1].grid(alpha=0.25)
        axes[2].grid(alpha=0.25)

        if title is None:
            sel = ""
            if (time_start is not None) or (time_stop is not None):
                sel = f" | {time_start} → {time_stop}"
            elif (idx_start is not None) or (idx_stop is not None):
                sel = f" | idx {idx_start} → {idx_stop}"
            title = f"{frame} matched_trajectory_full{sel}"
        axes[0].set_title(title)

        if legend:
            for ax in axes:
                ax.legend(loc="best")

    fig.tight_layout()
    return fig, axes