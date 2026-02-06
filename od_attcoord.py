import numpy as np
import matplotlib.pyplot as plt
import time

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
import itertools
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass



# -----------------------------
# Helpers: sampling + geometry
# -----------------------------

def sample_uniform_ball(n_samples, radius=1.0, dim=3, rng=None):
    """Uniform samples inside a dim-D ball of given radius."""
    if rng is None:
        rng = np.random.default_rng()
    x = rng.normal(size=(n_samples, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    u = rng.random(n_samples)
    r = radius * (u ** (1.0 / dim))
    return x * r[:, None]


def orthonormal_basis_from_u(u):
    """Two orthonormal vectors spanning plane normal to u."""
    u = u / np.linalg.norm(u)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, u)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = a - np.dot(a, u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def u_from_cap(u_curr, theta, phi):
    """Eq. (uit): candidate pointing vector on spherical cap around u_curr."""
    u_curr = u_curr / np.linalg.norm(u_curr)
    e1, e2 = orthonormal_basis_from_u(u_curr)
    u_phi = np.cos(phi) * e1 + np.sin(phi) * e2
    u_new = np.cos(theta) * u_curr + np.sin(theta) * u_phi
    return u_new / np.linalg.norm(u_new)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softplus(z, beta=1.0):
    """Numerically stable softplus: (1/beta) * log(1 + exp(beta*z))."""
    z_beta = beta * z
    return (1.0 / beta) * np.log1p(np.exp(z_beta))


# -----------------------------------------------------------
# Continuous FOV membership & dual coverage in whitened space
# -----------------------------------------------------------

def c_tilde_i(y_samples, Lp, p_hat, p_i, u_i, cos_theta_h, kappa_sigma):
    """
    c~_i(y) = sigma( kappa_sigma( ((p - p_i)/||p-p_i||)·u_i - cos(theta_h)) )
    with p = Lp y + p_hat
    """
    p_samples = (Lp @ y_samples.T).T + p_hat[None, :]
    v = p_samples - p_i[None, :]
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_unit = v / np.maximum(v_norm, 1e-12)

    cos_ang = np.einsum('ij,j->i', v_unit, u_i)
    z = kappa_sigma * (cos_ang - cos_theta_h)
    return sigmoid(z)


def k2_tilde(y_samples, Lp, p_hat, p_agents, u_agents, cos_theta_h, kappa_sigma):
    """Eq. (k2ty) continuous dual coverage."""
    M = len(p_agents)
    N = y_samples.shape[0]
    C = np.zeros((M, N))
    for i in range(M):
        C[i] = c_tilde_i(y_samples, Lp, p_hat, p_agents[i], u_agents[i],
                         cos_theta_h, kappa_sigma)

    if M == 2:
        # Algebraic identity: k2 = C0 * C1 for M=2
        return C[0] * C[1] + 0.01 * (C[0] + C[1])

    one_minus_C = 1.0 - C

    if M ==3:
        k2 = C[0] * C[1] * one_minus_C[2] + C[0] * C[2] * one_minus_C[1] + C[1] * C[2] * one_minus_C[0]
        k1 = 0.01 * (C[0] * one_minus_C[1] * one_minus_C[2] + C[2] * one_minus_C[0] * one_minus_C[1] + C[1] * one_minus_C[0] * one_minus_C[2])
        return k1 + k2

    if M == 4:
        # k2: exactly two detect
        k2 = (
                C[0] * C[1] * one_minus_C[2] * one_minus_C[3] +
                C[0] * C[2] * one_minus_C[1] * one_minus_C[3] +
                C[0] * C[3] * one_minus_C[1] * one_minus_C[2] +
                C[1] * C[2] * one_minus_C[0] * one_minus_C[3] +
                C[1] * C[3] * one_minus_C[0] * one_minus_C[2] +
                C[2] * C[3] * one_minus_C[0] * one_minus_C[1]
        )

        # k1: exactly one detects
        k1 = 0.01 * (
                C[0] * one_minus_C[1] * one_minus_C[2] * one_minus_C[3] +
                C[1] * one_minus_C[0] * one_minus_C[2] * one_minus_C[3] +
                C[2] * one_minus_C[0] * one_minus_C[1] * one_minus_C[3] +
                C[3] * one_minus_C[0] * one_minus_C[1] * one_minus_C[2]
        )

        return k1 + k2

    if M == 5:
        # k2: exactly two detect
        k2 = (
            # pairs involving 0
                C[0] * C[1] * one_minus_C[2] * one_minus_C[3] * one_minus_C[4] +
                C[0] * C[2] * one_minus_C[1] * one_minus_C[3] * one_minus_C[4] +
                C[0] * C[3] * one_minus_C[1] * one_minus_C[2] * one_minus_C[4] +
                C[0] * C[4] * one_minus_C[1] * one_minus_C[2] * one_minus_C[3] +

                # pairs involving 1
                C[1] * C[2] * one_minus_C[0] * one_minus_C[3] * one_minus_C[4] +
                C[1] * C[3] * one_minus_C[0] * one_minus_C[2] * one_minus_C[4] +
                C[1] * C[4] * one_minus_C[0] * one_minus_C[2] * one_minus_C[3] +

                # pairs involving 2
                C[2] * C[3] * one_minus_C[0] * one_minus_C[1] * one_minus_C[4] +
                C[2] * C[4] * one_minus_C[0] * one_minus_C[1] * one_minus_C[3] +

                # pairs involving 3
                C[3] * C[4] * one_minus_C[0] * one_minus_C[1] * one_minus_C[2]
        )

        # k1: exactly one detects
        k1 = 0.01 * (
                C[0] * one_minus_C[1] * one_minus_C[2] * one_minus_C[3] * one_minus_C[4] +
                C[1] * one_minus_C[0] * one_minus_C[2] * one_minus_C[3] * one_minus_C[4] +
                C[2] * one_minus_C[0] * one_minus_C[1] * one_minus_C[3] * one_minus_C[4] +
                C[3] * one_minus_C[0] * one_minus_C[1] * one_minus_C[2] * one_minus_C[4] +
                C[4] * one_minus_C[0] * one_minus_C[1] * one_minus_C[2] * one_minus_C[3]
        )

        return k1 + k2

    if M == 6:
        # k2: exactly two detect
        k2 = (
                C[0] * C[1] * one_minus_C[2] * one_minus_C[3] * one_minus_C[4] * one_minus_C[5] +
                C[0] * C[2] * one_minus_C[1] * one_minus_C[3] * one_minus_C[4] * one_minus_C[5] +
                C[0] * C[3] * one_minus_C[1] * one_minus_C[2] * one_minus_C[4] * one_minus_C[5] +
                C[0] * C[4] * one_minus_C[1] * one_minus_C[2] * one_minus_C[3] * one_minus_C[5] +
                C[0] * C[5] * one_minus_C[1] * one_minus_C[2] * one_minus_C[3] * one_minus_C[4] +

                C[1] * C[2] * one_minus_C[0] * one_minus_C[3] * one_minus_C[4] * one_minus_C[5] +
                C[1] * C[3] * one_minus_C[0] * one_minus_C[2] * one_minus_C[4] * one_minus_C[5] +
                C[1] * C[4] * one_minus_C[0] * one_minus_C[2] * one_minus_C[3] * one_minus_C[5] +
                C[1] * C[5] * one_minus_C[0] * one_minus_C[2] * one_minus_C[3] * one_minus_C[4] +

                C[2] * C[3] * one_minus_C[0] * one_minus_C[1] * one_minus_C[4] * one_minus_C[5] +
                C[2] * C[4] * one_minus_C[0] * one_minus_C[1] * one_minus_C[3] * one_minus_C[5] +
                C[2] * C[5] * one_minus_C[0] * one_minus_C[1] * one_minus_C[3] * one_minus_C[4] +

                C[3] * C[4] * one_minus_C[0] * one_minus_C[1] * one_minus_C[2] * one_minus_C[5] +
                C[3] * C[5] * one_minus_C[0] * one_minus_C[1] * one_minus_C[2] * one_minus_C[4] +
                C[4] * C[5] * one_minus_C[0] * one_minus_C[1] * one_minus_C[2] * one_minus_C[3]
        )

        # k1: exactly one detects
        k1 = 0.01 * (
                C[0] * one_minus_C[1] * one_minus_C[2] * one_minus_C[3] * one_minus_C[4] * one_minus_C[5] +
                C[1] * one_minus_C[0] * one_minus_C[2] * one_minus_C[3] * one_minus_C[4] * one_minus_C[5] +
                C[2] * one_minus_C[0] * one_minus_C[1] * one_minus_C[3] * one_minus_C[4] * one_minus_C[5] +
                C[3] * one_minus_C[0] * one_minus_C[1] * one_minus_C[2] * one_minus_C[4] * one_minus_C[5] +
                C[4] * one_minus_C[0] * one_minus_C[1] * one_minus_C[2] * one_minus_C[3] * one_minus_C[5] +
                C[5] * one_minus_C[0] * one_minus_C[1] * one_minus_C[2] * one_minus_C[3] * one_minus_C[4]
        )

        return k1 + k2

    prod_all = np.prod(one_minus_C, axis=0)

    k2 = np.zeros(N)
    for i in range(M):
        for j in range(i+1, M):  # i < j (no i=j terms)
            denom = one_minus_C[i] * one_minus_C[j]
            prod_excl = prod_all / np.maximum(denom, 1e-12)
            k2 += C[i] * C[j] * prod_excl

    return k2


def J_t_dual_coverage(
    p_hat, P_p, p_agents, u_agents,
    theta_h, d_M=3.0, kappa_sigma=100.0,
    n_mc=20000, seed=None, y_samples_cached=None
):
    """
    Eq. (Jty1) Monte-Carlo estimate.
    If y_samples_cached is provided, reuse the same MC points for smoother gradients.
    """
    rng = np.random.default_rng(seed)
    Lp = np.linalg.cholesky(P_p)

    if y_samples_cached is None:
        y = sample_uniform_ball(n_mc, radius=d_M, dim=3, rng=rng)
    else:
        y = y_samples_cached
        n_mc = y.shape[0]

    y2 = np.sum(y**2, axis=1)
    w = np.exp(-0.5 * y2)

    cos_theta_h = np.cos(theta_h)
    k2 = k2_tilde(y, Lp, p_hat, p_agents, u_agents, cos_theta_h, kappa_sigma)

    V_ball = (4.0/3.0) * np.pi * d_M**3
    integral_est = V_ball * np.mean(w * k2)

    J = integral_est / ((2*np.pi)**1.5)
    return J


# -----------------------------------------
# EMS exclusion penalty
# -----------------------------------------

def ems_exclusion_penalty(
    p_agents,          # (M,3) spacecraft positions
    u_agents,          # (M,3) pointing unit vectors
    p_em,              # (3,) EMS center position
    R_em,              # scalar EMS effective radius
    theta_h,           # FOV half-angle
    alpha_s,           # safety margin
    lambda_em,         # penalty weight
    beta_zeta=50.0     # softplus sharpness
):
    """
    Smooth EMS exclusion penalty term:

        Θ_EM,t = -λ_EM ∑ ζ(z_i),

    but here we *return* +λ_EM ∑ ζ(z_i) so it can be added to a cost function
    which we minimize:

        objective = -J_t + λ_EM ∑ ζ(z_i),

    with

        z_i = u_i^T u_i^{EM} - cos(θ_h + α^{EM}_i + α_s),
        α^{EM}_i = arcsin(R_EM / ||p_EM - p_i||),
        u_i^{EM} = (p_EM - p_i) / ||p_EM - p_i||.
    """
    M = len(p_agents)
    total = 0.0

    for i in range(M):
        r = p_em - p_agents[i]
        dist = np.linalg.norm(r)
        if dist < 1e-12:
            # Degenerate: s/c at EMS center; skip or heavily penalize if desired
            continue

        u_em = r / dist

        # α^{EM}_i = arcsin(R_EM / ||r||), clipped
        ratio = np.clip(R_em / dist, -1.0, 1.0)
        alpha_em = np.arcsin(ratio)

        angle_req = theta_h + alpha_em + alpha_s
        cos_req = np.cos(angle_req)

        dot_val = float(np.dot(u_agents[i], u_em))
        z_i = dot_val - cos_req

        total += softplus(z_i, beta=beta_zeta)

    return lambda_em * total


# -----------------------------------------
# Joint L-BFGS-B optimization over all agents
# -----------------------------------------

def unpack_angles(x, M):
    """x = [theta1,phi1,...,thetaM,phiM] -> arrays."""
    thetas = x[0::2]
    phis   = x[1::2]
    return thetas, phis


def angles_to_pointings(x, p_agents, u_curr_agents, theta_lower, theta_upper, M):
    """Convert joint angles to u_agents (M,3) with per-agent bounds."""
    thetas, phis = unpack_angles(x, M)
    u_agents = np.zeros_like(u_curr_agents, dtype=float)
    for i in range(M):
        # enforce per-agent θ-bounds softly (optimizer also has hard bounds)
        th = np.clip(thetas[i], theta_lower[i], theta_upper[i])
        ph = phis[i] % (2*np.pi)
        u_agents[i] = u_from_cap(u_curr_agents[i], th, ph)
    return u_agents


def make_cached_y(P_p, d_M, n_mc, seed):
    """Pre-draw y samples once to stabilize finite-diff gradients."""
    rng = np.random.default_rng(seed)
    y = sample_uniform_ball(n_mc, radius=d_M, dim=3, rng=rng)
    return y



def objective_joint(x, p_hat, P_p, p_agents, u_curr_agents,
                    theta_lower, theta_upper,
                    theta_h, d_M, kappa_sigma,
                    y_cached,
                    p_em=None, R_em=0.0,
                    alpha_s=0.0, lambda_em=0.0,
                    beta_zeta=50.0):
    """
    Joint objective for the optimizer:

        objective = -J_t_dual_coverage + EMS_penalty

    where EMS_penalty = λ_EM ∑ softplus(z_i) if enabled.
    """
    M = len(p_agents)
    u_agents = angles_to_pointings(x, p_agents, u_curr_agents, theta_lower, theta_upper, M)

    # Base objective: maximize J_t, so minimize -J_t
    J = J_t_dual_coverage(
        p_hat, P_p, p_agents, u_agents,
        theta_h, d_M=d_M, kappa_sigma=kappa_sigma,
        n_mc=y_cached.shape[0], y_samples_cached=y_cached
    )
    obj = -J

    # EMS exclusion penalty (only if configured)
    if lambda_em != 0.0 and p_em is not None and R_em > 0.0:
        penalty_em = ems_exclusion_penalty(
            p_agents, u_agents,
            p_em=p_em, R_em=R_em,
            theta_h=theta_h,
            alpha_s=alpha_s,
            lambda_em=lambda_em,
            beta_zeta=beta_zeta
        )
        obj += penalty_em

    return obj


def init_theta_phi_boundary_projection(
    p_hat,
    P_p,
    p_agents,
    u_curr_agents,
    theta_lower,
    theta_upper,
    theta_h,
    d_M,
    kappa_sigma,
    y_cached,
    p_em,
    R_em,
    alpha_s,
    seed,
    eps=1e-10,
):
    """
    3D warm-start initializer with EMS keep-out via boundary projection.

    Steps:
      1) Compute mean-pointing boresights u_star[i] (respecting slew limits).
      2) If all u_star are EMS-safe (with theta_h, R_em, alpha_s), return
         a jittered version of the mean-based angles (original behaviour).
      3) Otherwise, for each spacecraft:
           - If u_star[i] is safe  -> keep as single candidate.
           - If u_star[i] is unsafe:
               * Project u_star[i] onto the EMS exclusion boundary circle
                 (angle(u, v_em) = theta_h + alpha_em + alpha_s),
                 getting up to two candidate directions u_b1, u_b2,
                 subject to slew limits and keep-out.
      4) Enumerate all combinations of per-spacecraft candidates, evaluate
         J_t_dual_coverage, and select the one with minimal cost.
      5) Convert the resulting boresights back to (theta_i, phi_i).
      6) Apply a small theta jitter where it remains EMS-safe and
         within slew bounds.

    Returns
    -------
    x0 : np.ndarray, shape (2*M,)
        Stacked [theta_0, phi_0, theta_1, phi_1, ..., theta_{M-1}, phi_{M-1}].
    """

    rng = np.random.default_rng(seed=seed)
    jitter_values = np.deg2rad([0.0, 1.0, -1.0])  # small theta jitter, in radians
    p_hat = np.asarray(p_hat, dtype=float)
    p_agents = np.asarray(p_agents, dtype=float)
    u_curr_agents = np.asarray(u_curr_agents, dtype=float)
    theta_lower = np.asarray(theta_lower, dtype=float)
    theta_upper = np.asarray(theta_upper, dtype=float)

    M = p_agents.shape[0]
    x0 = np.zeros(2 * M, dtype=float)
    n_mc = y_cached.shape[0]

    # --------------------------------------------------
    # Helper: single-ray keep-out check for spacecraft i
    # --------------------------------------------------
    def keepout_safe_single(p_i, u_i, theta_h, p_em, R_em, alpha_s, eps=1e-12):
        r_vec = p_em - p_i
        r_norm = np.linalg.norm(r_vec)
        if r_norm < R_em + eps:
            # Spacecraft effectively inside EMS sphere -> treat as violation
            return False

        v_em = r_vec / r_norm
        cos_gamma = np.clip(np.dot(u_i, v_em), -1.0, 1.0)
        gamma = np.arccos(cos_gamma)

        ratio = np.clip(R_em / r_norm, -1.0, 1.0)
        alpha_em = np.arcsin(ratio)

        gamma_bound = theta_h + alpha_em + alpha_s
        return gamma >= gamma_bound

    # --------------------------------------------------
    # Helper: config-wise keep-out
    # --------------------------------------------------
    def config_keepout_safe(p_agents, u_array, theta_h, p_em, R_em, alpha_s):
        for i in range(p_agents.shape[0]):
            if not keepout_safe_single(p_agents[i], u_array[i], theta_h, p_em, R_em, alpha_s):
                return False
        return True

    # --------------------------------------------------
    # Helper: local parameterization u(theta, phi)
    # --------------------------------------------------
    def u_from_theta_phi(theta, phi, u_curr, e1, e2):
        """
        u = cos(theta)*u_curr + sin(theta)*(cos(phi)*e1 + sin(phi)*e2)
        All vectors assumed unit, theta in [0, pi].
        """
        return (
            np.cos(theta) * u_curr
            + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
        )

    # --------------------------------------------------
    # Precompute local bases and mean-pointing directions u_star[i]
    # --------------------------------------------------
    u_curr_norm = np.zeros_like(u_curr_agents)
    e1_list = np.zeros_like(u_curr_agents)
    e2_list = np.zeros_like(u_curr_agents)
    theta_star = np.zeros(M)
    phi_star = np.zeros(M)
    u_star = np.zeros_like(u_curr_agents)

    for i in range(M):
        p_i = p_agents[i]
        u_curr = u_curr_agents[i]
        n_u = np.linalg.norm(u_curr)
        if n_u < eps:
            # degenerate, pick arbitrary unit vector
            u_curr = np.array([0.0, 0.0, 1.0])
            n_u = 1.0
        u_curr /= n_u
        u_curr_norm[i] = u_curr

        # Desired mean direction
        d_vec = p_hat - p_i
        dist = np.linalg.norm(d_vec)
        if dist < eps:
            v_des = u_curr
        else:
            v_des = d_vec / dist

        # Local basis around u_curr
        e1, e2 = orthonormal_basis_from_u(u_curr)
        e1_list[i] = e1
        e2_list[i] = e2

        # Decompose v_des in {u_curr, e1, e2}
        a = np.dot(v_des, u_curr)
        b1 = np.dot(v_des, e1)
        b2 = np.dot(v_des, e2)
        s = np.sqrt(b1**2 + b2**2)

        # Spherical-like angles
        theta_i = np.arctan2(s, a)      # [0, pi]
        phi_i = np.arctan2(b2, b1)      # (-pi, pi]

        # Clamp theta to slew interval
        theta_i = np.clip(theta_i, theta_lower[i], theta_upper[i])

        theta_star[i] = theta_i
        phi_star[i] = phi_i

        u_star[i] = u_from_theta_phi(theta_i, phi_i, u_curr, e1, e2)

    # --------------------------------------------------
    # Step 1: If mean-based config is EMS-safe, use original style + jitter
    # --------------------------------------------------
    if config_keepout_safe(p_agents, u_star, theta_h, p_em, R_em, alpha_s):
        for i in range(M):
            jitter = rng.choice(jitter_values)
            theta_i = theta_star[i] + jitter
            # keep phi as-is
            # ensure slew bounds
            theta_i = np.clip(theta_i, theta_lower[i], theta_upper[i])

            # rebuild u and re-check keep-out; if violated, drop jitter
            u_i = u_from_theta_phi(theta_i, phi_star[i], u_curr_norm[i], e1_list[i], e2_list[i])
            if not keepout_safe_single(p_agents[i], u_i, theta_h, p_em, R_em, alpha_s):
                theta_i = theta_star[i]  # revert

            x0[2 * i] = theta_i
            x0[2 * i + 1] = phi_star[i]
        return x0

    # --------------------------------------------------
    # Step 2: Mean-based config not safe -> boundary projection per spacecraft
    # --------------------------------------------------
    candidate_u_list = []

    for i in range(M):
        p_i = p_agents[i]
        u_curr = u_curr_norm[i]
        u_i_star = u_star[i]

        # Direction to EMS center
        r_vec = p_em - p_i
        r_norm = np.linalg.norm(r_vec)
        if r_norm < R_em + eps:
            # pathological: spacecraft basically inside EMS
            # fallback: just keep u_star
            candidate_u_list.append([u_i_star])
            continue

        v_em = r_vec / r_norm
        ratio = np.clip(R_em / r_norm, -1.0, 1.0)
        alpha_em = np.arcsin(ratio)
        gamma_bound = theta_h + alpha_em + alpha_s

        # If gamma_bound is nonsensical (>= pi), just keep u_star
        if gamma_bound >= np.pi - 1e-6:
            candidate_u_list.append([u_i_star])
            continue

        # Check whether u_star is actually safe; if so, just keep it
        if keepout_safe_single(p_i, u_i_star, theta_h, p_em, R_em, alpha_s):
            candidate_u_list.append([u_i_star])
            continue


        # Otherwise, project u_star onto boundary circle: angle(u, v_em) = gamma_bound
        # Decompose u_star into parallel and perpendicular wrt v_em
        c = np.dot(u_i_star, v_em)
        v_parallel = c * v_em
        v_perp = u_i_star - v_parallel
        n_perp = np.linalg.norm(v_perp)

        if n_perp < eps:
            # u_star is nearly colinear with v_em; choose an arbitrary perp direction
            e1_em, e2_em = orthonormal_basis_from_u(v_em)
            v_perp_hat = e1_em
        else:
            v_perp_hat = v_perp / n_perp

        cos_gb = np.cos(gamma_bound)
        sin_gb = np.sin(gamma_bound)

        # Two symmetric boundary directions on the circle
        u_b1 = cos_gb * v_em + sin_gb * v_perp_hat
        u_b2 = cos_gb * v_em - sin_gb * v_perp_hat

        # Normalize for safety
        u_b1 /= max(np.linalg.norm(u_b1), eps)
        u_b2 /= max(np.linalg.norm(u_b2), eps)

        candidate_dirs = []

        # Check slew feasibility + keep-out for each boundary candidate
        for u_b in (u_b1, u_b2):
            cos_theta = np.clip(np.dot(u_b, u_curr), -1.0, 1.0)
            theta_b = np.arccos(cos_theta)

            if theta_b < theta_lower[i] - 1e-6 or theta_b > theta_upper[i] + 1e-6:
                continue  # outside slew envelope

            # if not keepout_safe_single(p_i, u_b, theta_h, p_em, R_em, alpha_s):
            #     continue  # should be rare, but guard anyway

            candidate_dirs.append(u_b)

        if len(candidate_dirs) == 0:
            # Fallback if both boundary projections fail: use u_star anyway
            candidate_u_list.append([u_i_star])
        else:
            # Use the boundary candidates (1 or 2) for this spacecraft
            candidate_u_list.append(candidate_dirs)

    # --------------------------------------------------
    # Step 3: Enumerate all combinations of candidate_u_list and pick best J_t
    # --------------------------------------------------
    best_J = -np.inf
    best_u = None

    index_ranges = [range(len(cands)) for cands in candidate_u_list]

    for choice in itertools.product(*index_ranges):
        u_trial = np.zeros_like(u_curr_agents)
        for i, idx in enumerate(choice):
            u_trial[i] = candidate_u_list[i][idx]

        J_val = J_t_dual_coverage(
            p_hat,
            P_p,
            p_agents,
            u_trial,
            theta_h,
            d_M=d_M,
            kappa_sigma=kappa_sigma,
            n_mc=n_mc,
            y_samples_cached=y_cached,
        )

        if J_val > best_J:
            best_J = J_val
            best_u = u_trial.copy()

    if best_u is None:
        # Total fallback (should be extremely rare): revert to mean-based u_star
        best_u = u_star.copy()

    # --------------------------------------------------
    # Step 4: Convert best_u back to (theta_i, phi_i), then apply safe jitter
    # --------------------------------------------------
    for i in range(M):
        u_curr = u_curr_norm[i]
        e1 = e1_list[i]
        e2 = e2_list[i]
        u_i = best_u[i]

        a = np.dot(u_i, u_curr)
        b1 = np.dot(u_i, e1)
        b2 = np.dot(u_i, e2)
        s = np.sqrt(b1**2 + b2**2)

        theta_i = np.arctan2(s, a)
        phi_i = np.arctan2(b2, b1)

        # Ensure theta within bounds (small numeric repair if needed)
        theta_i = np.clip(theta_i, theta_lower[i], theta_upper[i])

        # Optional: apply jitter in theta while preserving keep-out
        jitter = rng.choice(jitter_values)
        theta_j = np.clip(theta_i + jitter, theta_lower[i], theta_upper[i])
        u_j = u_from_theta_phi(theta_j, phi_i, u_curr, e1, e2)

        if keepout_safe_single(p_agents[i], u_j, theta_h, p_em, R_em, alpha_s):
            theta_i = theta_j  # accept jitter

        x0[2 * i] = theta_i
        x0[2 * i + 1] = phi_i

    return x0


def estimate_theta_bounds_from_ellipsoid(p_hat, P_p, p_agents, u_curr_agents,
                                         d_M, n_shell=400, seed=12345):
    """
    For each spacecraft i:
      - sample points on the Mahalanobis shell (||y|| = d_M) of the ellipsoid
      - compute angular distance θ_i(p) between u_curr_i and direction to each point
      - return θ_min_i, θ_max_i^{(ell)} over those samples
    """
    M = len(p_agents)
    rng = np.random.default_rng(seed)

    # Sample unit directions on the 3D sphere
    dirs = rng.normal(size=(n_shell, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # Map to ellipsoid shell: p = p_hat + Lp * (d_M * dir)
    Lp = np.linalg.cholesky(P_p)
    y_shell = d_M * dirs
    p_shell = (Lp @ y_shell.T).T + p_hat[None, :]   # (n_shell, 3)

    theta_min = np.zeros(M)
    theta_max = np.zeros(M)

    for i in range(M):
        p_i = p_agents[i]
        u_i_curr = u_curr_agents[i] / np.linalg.norm(u_curr_agents[i])

        r = p_shell - p_i[None, :]             # vectors from s/c to ellipsoid points
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)
        r_unit = r / np.maximum(r_norm, 1e-12)

        cos_theta = np.einsum('ij,j->i', r_unit, u_i_curr)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_vals = np.arccos(cos_theta)     # [0, π]

        theta_min[i] = np.min(theta_vals)
        theta_max[i] = np.max(theta_vals)

        if np.rad2deg(theta_min[i]) < 0.5:
            theta_min[i] = 0

    return theta_min, theta_max


def theta_s_of_dt(delta_t_s, alpha_max, omega_max):
    """
    Compute the maximum allowable slew angle θ_{s,t} for a given slew time Δt_s.

    Implements the piecewise definition:
        Δt_crit = 2 ω_max / α_max

        θ_{s,t} = α_max Δt_s^2 / 4                         if Δt_s <  Δt_crit
                  (Δt_s - ω_max / α_max) ω_max            if Δt_s >  Δt_crit

    At Δt_s = Δt_crit both expressions are equal, so we use the first branch
    for Δt_s <= Δt_crit and the second for Δt_s > Δt_crit.

    Parameters
    ----------
    delta_t_s : float or array_like
        Slew time Δt_s (seconds).
    alpha_max : float
        Maximum angular acceleration α_max (rad/s^2).
    omega_max : float
        Maximum angular rate ω_max (rad/s).

    Returns
    -------
    theta_s_t : float or ndarray
        Maximum allowable slew angle θ_{s,t} (radians), matching the shape of delta_t_s.
    """
    delta_t_s = np.asarray(delta_t_s, dtype=float)
    delta_t_crit = 2.0 * omega_max / alpha_max

    theta_s_t = np.where(
        delta_t_s <= delta_t_crit,
        0.25 * alpha_max * delta_t_s**2,
        (delta_t_s - omega_max / alpha_max) * omega_max
    )

    # Return scalar if input was scalar
    if np.isscalar(delta_t_s):
        return float(theta_s_t)
    return theta_s_t


def optimize_pointing_lbfgs_joint(
    p_hat, P_p, p_agents, u_curr_agents,
    theta_h, theta_s_list,
    d_M=3.0, kappa_sigma=120.0,
    n_mc=25000, seed=0,
    n_restarts=3,
    p_em=None, R_em=0.0,
    alpha_s=0.0, lambda_em=0.0,
    beta_zeta=50.0
):
    """
    Joint L-BFGS-B over all agents’ (theta_i,phi_i), with θ-bounds derived from
    the uncertainty ellipsoid and slew constraints, plus optional EMS penalty.

    Returns:
      u_best      : (M,3) best pointing vectors
      angles_best : list[(theta_i, phi_i)]
      J_best      : best J_t value
      history     : list of dicts with state logs
      best_cost   : best objective value (cost = -J_t + penalty)
    """
    rng = np.random.default_rng(seed)
    M = len(p_agents)

    # --- θ-bounds from ellipsoid + slew ---
    theta_min_ell, theta_max_ell = estimate_theta_bounds_from_ellipsoid(
        p_hat, P_p, p_agents, u_curr_agents, d_M,
        n_shell=400, seed=seed+999
    )

    theta_upper = np.minimum(theta_max_ell, theta_s_list)
    theta_lower = theta_min_ell.copy()
    theta_lower = np.maximum(theta_lower, 0.0)

    infeasible_mask = theta_upper < theta_lower
    if np.any(infeasible_mask):
        print(f"Infeasible: slew limit smaller than required to reach ellipsoid {np.rad2deg(theta_upper)}, {np.rad2deg(theta_lower)}.")
        # NOTE: now returning 5 values
        return None, None, 0.0, [], np.inf

    # MC cache
    y_cached = make_cached_y(P_p, d_M, n_mc, seed=seed+123)

    # Bounds in (θ, φ)
    bounds = []
    for i in range(M):
        bounds.append((theta_lower[i], theta_upper[i]))  # theta_i
        bounds.append((0.0, 2*np.pi))                    # phi_i

    # ---- History logger ----
    history = []

    def log_state(x, restart_idx):
        """
        Record:
          - u from (θ, φ)
          - slew between u_curr and u
          - J_t at this point (no EMS penalty)
        """
        u = angles_to_pointings(x, p_agents, u_curr_agents, theta_lower, theta_upper, M)
        dots = np.einsum("ij,ij->i", u_curr_agents, u)
        dots = np.clip(dots, -1.0, 1.0)
        slews = np.arccos(dots)

        J_val = J_t_dual_coverage(
            p_hat, P_p, p_agents, u,
            theta_h, d_M=d_M, kappa_sigma=kappa_sigma,
            n_mc=y_cached.shape[0], y_samples_cached=y_cached
        )

        history.append({
            "x": x.copy(),
            "u": u.copy(),
            "slew": slews.copy(),
            "J": J_val,
            "restart": restart_idx
        })

    best_x = None
    best_f = np.inf   # objective = cost = -J_t + penalty
    best_cost = np.inf

    # x0_mean = init_theta_phi_to_mean(
    #     p_hat, p_agents, u_curr_agents,
    #     theta_lower, theta_upper, seed
    # )

    x0_mean = init_theta_phi_boundary_projection(
        p_hat, P_p, p_agents, u_curr_agents,
        theta_lower, theta_upper, theta_h,
        d_M, kappa_sigma, y_cached, p_em,
        R_em, alpha_s, seed
    )


    for r in range(n_restarts):
        if r == 0:
            x0 = x0_mean.copy()
        else:

            noise = rng.normal(scale=0.05, size=2 * M)
            x0 = x0_mean + noise
            for i in range(M):
                x0[2 * i] = np.clip(x0[2 * i], theta_lower[i], theta_upper[i])
                x0[2 * i + 1] = x0[2 * i + 1] % (2 * np.pi)

        log_state(x0, r)

        f = lambda z: objective_joint(
            z, p_hat, P_p, p_agents, u_curr_agents,
            theta_lower, theta_upper,
            theta_h, d_M, kappa_sigma, y_cached,
            p_em=p_em, R_em=R_em,
            alpha_s=alpha_s, lambda_em=lambda_em,
            beta_zeta=beta_zeta
        )

        def cb(xk, restart_idx=r):
            log_state(xk, restart_idx)

        res = minimize(
            f, x0, method="L-BFGS-B",
            bounds=bounds,
            callback=cb,
            options=dict(maxiter=60, ftol=1e-10, disp=False)
        )
        x_star = res.x
        f_star = res.fun

        if f_star < best_f:
            best_f = f_star
            best_x = x_star.copy()
            best_cost = f_star   # store best objective

    # Unpack best and compute J_t
    thetas_best, phis_best = unpack_angles(best_x, M)
    u_best = angles_to_pointings(best_x, p_agents, u_curr_agents, theta_lower, theta_upper, M)

    J_best = J_t_dual_coverage(
        p_hat, P_p, p_agents, u_best,
        theta_h, d_M=d_M, kappa_sigma=kappa_sigma,
        n_mc=y_cached.shape[0], y_samples_cached=y_cached
    )


    angles_best = [(thetas_best[i], phis_best[i]) for i in range(M)]
    return u_best, angles_best, J_best, history, best_cost


# -----------------------
# Runtime Attitude Coordinator API (for one trial per epoch)
# -----------------------
@dataclass
class AttCoordResult:
    u_cmd: np.ndarray                 # (M,3) unit boresight vectors
    chosen_dt: float                  # seconds
    cost: float
    J: float
    coverage: int                     # -1 if not computed
    theta_req_avg_deg: float
    extra: Dict[str, Any]

class AttitudeCoordinator:
    """
    Lightweight wrapper around the joint pointing optimizer so you can call one
    attitude-coordination trial per training epoch.

    This version:
      - Supports time-varying inputs over dt_grid (N timesteps):
          p_agents: (M,3) or (N,M,3)
          p_hat:    (3,)  or (N,3)
          P_p:      (3,3) or (N,3,3)
      - Computes TWO results:
          (1) Optimizer result: best feasible across dt_grid (min cost)
          (2) Mean-pointing result: earliest dt where all agents can point to mean
              using all_agents_can_point_to_mean(...) which includes
              slew + EMS keep-out constraints.
      - If a method is infeasible across the entire horizon:
          coverage = -1, and other scalars are NaN, u_cmd = None

    CHANGE: outputs AVERAGE slew (deg), not sum:
      theta_req_avg_deg = mean_i angle_between(u_curr[i], u_cmd[i]) in degrees
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        opt = cfg.get("optimizer_att_coord", {})
        self.kappa_sigma = float(opt.get("kappa_sigma", 1.0))
        self.n_mc = int(opt.get("n_mc", 200))
        self.n_restarts = int(opt.get("n_restarts", 1))

        # Optional EMS / keepout config defaults
        ems = cfg.get("ems", {})
        self.p_em_default = np.array(ems.get("p_em", [0.0, 0.0, 0.0]), dtype=float)
        self.R_em_default = float(ems.get("R_em", 0.0))
        self.alpha_s_default = np.deg2rad(float(ems.get("alpha_s_deg", 0.0)))
        self.lambda_em_default = float(ems.get("lambda_em", 0.0))
        self.beta_zeta_default = float(ems.get("beta_zeta", 50.0))


    @staticmethod
    def _broadcast_time_series(x: np.ndarray, N: int, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Accept either a static array with shape=target_shape, or a time series with
        shape=(N,)+target_shape. Return a time series array with shape=(N,)+target_shape.
        """
        x = np.asarray(x, dtype=float)

        if x.shape == target_shape:
            return np.broadcast_to(x, (N,) + target_shape).copy()

        if x.shape == (N,) + target_shape:
            return x

        raise ValueError(f"Expected {target_shape} or {(N,) + target_shape}, got {x.shape}")


    @staticmethod
    def _nan_result() -> AttCoordResult:
        return AttCoordResult(
            u_cmd=None,
            chosen_dt=float("nan"),
            cost=float("nan"),
            J=float("nan"),
            coverage=-1,                     # required: -1 for infeasible
            theta_req_avg_deg=float("nan"),  # avg slew in degrees
            extra={"history": None},
        )

    def step(
            self,
            p_agents: np.ndarray,
            u_curr_agents: np.ndarray,
            p_hat: np.ndarray,
            P_p: np.ndarray,
            dt_grid: np.ndarray,
            theta_h: float,
            alpha_max: float,
            omega_max: float,
            *,
            d_M: Optional[float] = None,
            trial_seed: int = 0,
            # EMS params
            p_em: Optional[np.ndarray] = None,
            R_em: Optional[float] = None,
            alpha_s: Optional[float] = None,
            # optimizer penalty params
            lambda_em: Optional[float] = None,
            beta_zeta: Optional[float] = None,
            # coverage counting
            coverage_point: Optional[np.ndarray] = None,  # (3,) or (N,3)
    ) -> Tuple[AttCoordResult, AttCoordResult]:
        """
        Returns:
            (res_opt, res_mean)

        res_opt:
          - best feasible optimizer solution across dt_grid (min cost)

        res_mean:
          - earliest dt where all agents can point to mean using:
                all_agents_can_point_to_mean(...)
            which includes slew + keepout constraints.
          - If infeasible across horizon -> coverage=-1, everything else NaN, u_cmd=None
        """

        # --- normalize / validate ---
        dt_grid = np.asarray(dt_grid, dtype=float).ravel()
        N = int(dt_grid.size)
        if N == 0:
            return self._nan_result(), self._nan_result()

        u_curr_agents = np.asarray(u_curr_agents, dtype=float)
        if u_curr_agents.ndim != 2 or u_curr_agents.shape[1] != 3:
            raise ValueError(f"u_curr_agents must be (M,3), got {u_curr_agents.shape}")
        M = int(u_curr_agents.shape[0])
        if M <= 0:
            return self._nan_result(), self._nan_result()

        if d_M is None:
            d_M = float(self.cfg.get("covariance", {}).get("d_mahal", 1.0))

        # EMS defaults
        if p_em is None:
            p_em = self.p_em_default
        else:
            p_em = np.asarray(p_em, dtype=float).reshape(3, )

        if R_em is None:
            R_em = self.R_em_default
        if alpha_s is None:
            alpha_s = self.alpha_s_default
        if lambda_em is None:
            lambda_em = self.lambda_em_default
        if beta_zeta is None:
            beta_zeta = self.beta_zeta_default

        # --- time-varying inputs ---
        p_agents_ts = self._broadcast_time_series(p_agents, N, (M, 3))
        p_hat_ts = self._broadcast_time_series(p_hat, N, (3,))
        P_p_ts = self._broadcast_time_series(P_p, N, (3, 3))

        # coverage_point may be static or time-varying
        covpt_ts = None
        if coverage_point is not None:
            cp = np.asarray(coverage_point, dtype=float)
            if cp.shape == (3,):
                covpt_ts = np.broadcast_to(cp.reshape(1, 3), (N, 3)).copy()
            elif cp.shape == (N, 3):
                covpt_ts = cp
            else:
                raise ValueError(f"coverage_point must be (3,) or (N,3), got {cp.shape}")

        # ============================================================
        # (A) OPTIMIZER result: best feasible over dt_grid (min cost)
        # ============================================================
        best_opt = {"cost": np.inf}

        for k, dt in enumerate(dt_grid):
            dt = float(dt)
            p_agents_k = p_agents_ts[k]
            p_hat_k = p_hat_ts[k].reshape(3, )
            P_p_k = P_p_ts[k]

            theta_s_t = float(theta_s_of_dt(dt, alpha_max, omega_max))
            theta_s_t = float(np.clip(theta_s_t, 0.0, np.deg2rad(179.0)))
            theta_s_list_t = np.full(M, theta_s_t)

            u_star, ang_star, J_star, history, cost_star = optimize_pointing_lbfgs_joint(
                p_hat_k, P_p_k, p_agents_k, u_curr_agents,
                float(theta_h), theta_s_list_t,
                d_M=float(d_M),
                kappa_sigma=float(self.kappa_sigma),
                n_mc=int(self.n_mc),
                seed=int(trial_seed),
                n_restarts=int(self.n_restarts),
                p_em=p_em, R_em=float(R_em),
                alpha_s=float(alpha_s),
                lambda_em=float(lambda_em),
                beta_zeta=float(beta_zeta),
            )

            if u_star is None:
                continue

            c = float(cost_star)
            if c < best_opt["cost"]:
                # Average slew relative to current
                slew_sum = 0.0
                for i in range(M):
                    slew_sum += angle_between(u_curr_agents[i], u_star[i])
                slew_avg_deg = float(np.rad2deg(slew_sum / max(M, 1)))

                cov_cnt = -1
                if covpt_ts is not None:
                    cov_cnt = coverage_count_point(covpt_ts[k], p_agents_k, u_star, float(theta_h))

                best_opt = dict(
                    cost=c,
                    dt=dt,
                    u=u_star,
                    J=float(J_star),
                    theta_req_avg_deg=slew_avg_deg,
                    coverage=int(cov_cnt),
                    history=history,
                )

        if not np.isfinite(best_opt.get("cost", np.inf)):
            res_opt = self._nan_result()
        else:
            res_opt = AttCoordResult(
                u_cmd=best_opt["u"],
                chosen_dt=float(best_opt["dt"]),
                cost=float(best_opt["cost"]),
                J=float(best_opt["J"]),
                coverage=int(best_opt["coverage"]),
                theta_req_avg_deg=float(best_opt["theta_req_avg_deg"]),
                extra={"history": best_opt["history"]},
            )

        # ============================================================
        # (B) MEAN method: earliest epoch all can point to mean (3D)
        #     Uses updated all_agents_can_point_to_mean(...)
        # ============================================================
        best_mean = None

        for k, dt in enumerate(dt_grid):
            dt = float(dt)
            p_agents_k = p_agents_ts[k]
            p_hat_k = p_hat_ts[k].reshape(3, )

            all_ok, per_ok, u_mean, theta_req, theta_s_t = all_agents_can_point_to_mean(
                dt,
                p_hat_k,  # <-- 3D mean at this epoch
                p_agents_k, u_curr_agents,
                float(theta_h),
                float(alpha_max), float(omega_max),
                p_em, float(R_em), float(alpha_s),
            )

            if all_ok:
                # theta_req is per-agent required slew (rad). Average it.
                slew_avg_deg = float(np.rad2deg(np.nanmean(theta_req)))

                cov_cnt = -1
                if covpt_ts is not None:
                    cov_cnt = coverage_count_point(covpt_ts[k], p_agents_k, u_mean, float(theta_h))

                best_mean = dict(
                    dt=dt,
                    u=u_mean,
                    theta_req_avg_deg=slew_avg_deg,
                    coverage=int(cov_cnt),
                    per_agent_ok=per_ok,
                    theta_required=theta_req,
                    theta_s_allowed=float(theta_s_t),
                )
                break  # earliest feasible

        if best_mean is None:
            res_mean = self._nan_result()
        else:
            res_mean = AttCoordResult(
                u_cmd=best_mean["u"],
                chosen_dt=float(best_mean["dt"]),
                cost=float("nan"),
                J=float("nan"),
                coverage=int(best_mean["coverage"]),
                theta_req_avg_deg=float(best_mean["theta_req_avg_deg"]),
                extra={
                    "history": None,
                    "per_agent_ok": best_mean["per_agent_ok"],
                    "theta_required": best_mean["theta_required"],
                    "theta_s_allowed": best_mean["theta_s_allowed"],
                },
            )

        return res_opt, res_mean


def unit(v: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.arccos(c))



def coverage_count_point(p_point: np.ndarray,
                         p_agents: np.ndarray,
                         u_boresights: np.ndarray,
                         theta_h: float,
                         eps: float = 1e-12) -> int:
    """
    Count how many agents' FOV cones (half-angle theta_h) contain a given 3D point.

    p_point:      (3,)
    p_agents:     (M,3)
    u_boresights: (M,3) unit vectors (assumed; will be normalized defensively)
    """
    p_point = np.asarray(p_point, dtype=float).reshape(3,)
    p_agents = np.asarray(p_agents, dtype=float)
    u_boresights = np.asarray(u_boresights, dtype=float)

    M = int(p_agents.shape[0])
    count = 0
    for i in range(M):
        r = p_point - p_agents[i]
        d = float(np.linalg.norm(r))
        if d < eps:
            # If point is at spacecraft location, treat as covered (degenerate)
            count += 1
            continue
        u_r = r / d

        u_b = u_boresights[i]
        nb = float(np.linalg.norm(u_b))
        if nb < eps:
            continue
        u_b = u_b / nb

        if angle_between(u_b, u_r) <= float(theta_h):
            count += 1
    return int(count)


def keepout_safe_single(p_sc, u_boresight, theta_h, p_em, R_em, alpha_s, eps=1e-12):
    """
    Returns True if the entire FOV cone (half-angle theta_h) about boresight u_boresight
    stays at least alpha_s away from the EMS sphere as seen from the spacecraft.

    Condition used (simple & common):
      gamma >= alpha_em + theta_h + alpha_s
    where
      gamma    = angle between boresight and direction-to-EMS-center
      alpha_em = apparent half-angle of EMS sphere from spacecraft = asin(R_em / d)
    """
    r_em = p_em - p_sc
    d = np.linalg.norm(r_em)
    if d < eps:
        return False  # spacecraft at EMS center -> invalid

    u_em = r_em / d
    gamma = angle_between(u_boresight, u_em)

    # apparent angular radius of the EMS sphere
    if d <= R_em:
        return False  # inside/at sphere
    alpha_em = float(np.arcsin(np.clip(R_em / d, 0.0, 1.0)))

    return gamma >= (alpha_em + theta_h + alpha_s)


def all_agents_can_point_to_mean(
    dt: float,
    p_hat_t: np.ndarray,
    p_agents: np.ndarray,
    u_curr_agents: np.ndarray,
    theta_h: float,
    alpha_max: float,
    omega_max: float,
    p_em: np.ndarray,
    R_em: float,
    alpha_s: float,
    eps: float = 1e-12,
):
    """
    Time-series friendly, 3D version.

    Checks whether *all* agents can (1) slew to point at the 3D mean target
    at this epoch and (2) satisfy the EMS keep-out constraint.

    Inputs:
      - dt: scalar seconds for this epoch (used only to compute slew limit theta_s_t)
      - p_hat_t: (3,) 3D target mean at this epoch
      - p_agents: (M,3) spacecraft positions at this epoch
      - u_curr_agents: (M,3) current boresight unit vectors (at "now")
      - theta_h: FOV half-angle (rad)
      - alpha_max, omega_max: slew envelope parameters for theta_s_of_dt
      - p_em: (3,) EMS center
      - R_em: EMS radius
      - alpha_s: EMS half-angle keepout (rad)
      - eps: numeric epsilon for unit()

    Returns:
      (all_ok, per_agent_ok, u_mean_list, theta_req_list, theta_s_t)

      - all_ok: bool, True iff all agents are OK
      - per_agent_ok: (M,) bool array
      - u_mean_list: (M,3) desired pointing directions to the mean
      - theta_req_list: (M,) required slew angles (rad) from current to desired
      - theta_s_t: scalar allowed slew (rad) for this dt
    """

    # Slew limit for this epoch
    theta_s_t = float(theta_s_of_dt(float(dt), float(alpha_max), float(omega_max)))
    theta_s_t = float(np.clip(theta_s_t, 0.0, np.deg2rad(179.0)))

    p_hat_t = np.asarray(p_hat_t, dtype=float).reshape(3,)
    p_agents = np.asarray(p_agents, dtype=float)
    u_curr_agents = np.asarray(u_curr_agents, dtype=float)
    p_em = np.asarray(p_em, dtype=float).reshape(3,)

    if p_agents.ndim != 2 or p_agents.shape[1] != 3:
        raise ValueError(f"p_agents must be (M,3), got {p_agents.shape}")
    if u_curr_agents.shape != p_agents.shape:
        raise ValueError(f"u_curr_agents must match p_agents shape (M,3), got {u_curr_agents.shape}")

    M = int(p_agents.shape[0])
    per_ok = np.zeros(M, dtype=bool)
    u_mean = np.zeros((M, 3), dtype=float)
    theta_req = np.full(M, np.nan, dtype=float)

    for i in range(M):
        # Desired pointing to mean (3D)
        u_des = unit(p_hat_t - p_agents[i], eps=eps)
        if u_des is None:
            per_ok[i] = False
            continue
        u_mean[i] = u_des

        # Required slew from current
        th = angle_between(u_curr_agents[i], u_des)
        theta_req[i] = th

        # Slew constraint
        if th > theta_s_t + 1e-12:
            per_ok[i] = False
            continue

        # Keep-out constraint (EMS exclusion)
        if not keepout_safe_single(p_agents[i], u_des, float(theta_h), p_em, float(R_em), float(alpha_s)):
            per_ok[i] = False
            continue

        per_ok[i] = True

    return bool(np.all(per_ok)), per_ok, u_mean, theta_req, theta_s_t

