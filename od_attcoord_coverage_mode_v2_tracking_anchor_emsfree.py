import numpy as np
import matplotlib.pyplot as plt
import time

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
import itertools
from scipy.special import expit
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import utilities as util


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
    return expit(z)


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


def k2_tilde(y_samples, Lp, p_hat, p_agents, u_agents, cos_theta_h, kappa_sigma, lambda_k1,
             coverage_mode="exact2_plus_k1"):
    """
    Continuous soft coverage score.

    coverage_mode="exact2_plus_k1": original behaviour: exactly two coverage,
        plus lambda_k1 times exactly one coverage.
    coverage_mode="atleast2": reward at least two agents covering each sample.
        This does not penalize three-or-more simultaneous coverage.
    """
    M = len(p_agents)
    N = y_samples.shape[0]
    C = np.zeros((M, N))
    for i in range(M):
        C[i] = c_tilde_i(y_samples, Lp, p_hat, p_agents[i], u_agents[i],
                         cos_theta_h, kappa_sigma)

    one_minus_C = 1.0 - C

    if coverage_mode in ("atleast2", "at_least_2", "at_least_two"):
        # Soft probability-like score for >=2 covered:
        #   P(K >= 2) = 1 - P(K = 0) - P(K = 1).
        # This is the desired mode when two or more spacecraft can each place
        # the full uncertainty volume inside their FOV by pointing at the mean.
        k0 = np.prod(one_minus_C, axis=0)
        k1 = np.zeros(N)
        for i in range(M):
            term = C[i].copy()
            for j in range(M):
                if j != i:
                    term *= one_minus_C[j]
            k1 += term
        return 1.0 - k0 - k1

    if coverage_mode not in ("exact2_plus_k1", "exact2", "original"):
        raise ValueError(f"Unknown coverage_mode: {coverage_mode}")

    if M == 2:
        # Original behaviour for M=2. Note this is not a probability-like
        # bounded score because the lambda_k1 fallback intentionally rewards
        # one-agent coverage.
        return C[0] * C[1] + lambda_k1 * (C[0] + C[1])


    if M ==3:
        k2 = C[0] * C[1] * one_minus_C[2] + C[0] * C[2] * one_minus_C[1] + C[1] * C[2] * one_minus_C[0]
        k1 = lambda_k1 * (C[0] * one_minus_C[1] * one_minus_C[2] + C[2] * one_minus_C[0] * one_minus_C[1] + C[1] * one_minus_C[0] * one_minus_C[2])
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
        k1 = lambda_k1 * (
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
        k1 = lambda_k1 * (
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
        k1 = lambda_k1 * (
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
    n_mc=20000, seed=None, y_samples_cached=None, lambda_k1=0.5,
    coverage_mode="exact2_plus_k1"
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
    k2 = k2_tilde(
        y, Lp, p_hat, p_agents, u_agents, cos_theta_h, kappa_sigma, lambda_k1,
        coverage_mode=coverage_mode
    )

    V_ball = (4.0/3.0) * np.pi * d_M**3
    integral_est = V_ball * np.mean(w * k2)

    J = integral_est / ((2*np.pi)**1.5)
    return J


def mean_pointing_full_uncertainty_cover_flags(
    p_hat,
    P_p,
    p_agents,
    u_curr_agents,
    theta_h,
    theta_s_list,
    d_M,
    y_samples,
    *,
    p_em=None,
    R_em=0.0,
    alpha_s=0.0,
    eps=1e-12,
):
    """
    Per-agent pre-check for objective selection.

    For each spacecraft, point the boresight at the target mean and check whether:
      1) the mean-pointing slew is feasible under theta_s_list[i],
      2) the mean-pointing direction satisfies EMS keepout, if enabled, and
      3) every sampled point in the Mahalanobis uncertainty volume lies inside
         that spacecraft's FOV cone.

    The volume check is Monte-Carlo, using the same y_samples as the objective.
    """
    p_hat = np.asarray(p_hat, dtype=float).reshape(3,)
    P_p = np.asarray(P_p, dtype=float).reshape(3, 3)
    p_agents = np.asarray(p_agents, dtype=float)
    u_curr_agents = np.asarray(u_curr_agents, dtype=float)
    theta_s_list = np.asarray(theta_s_list, dtype=float).ravel()
    y_samples = np.asarray(y_samples, dtype=float)

    M = int(p_agents.shape[0])
    flags = np.zeros(M, dtype=bool)
    u_mean = np.zeros((M, 3), dtype=float)
    theta_req = np.full(M, np.nan, dtype=float)

    Lp = np.linalg.cholesky(P_p)
    p_samples = (Lp @ y_samples.T).T + p_hat[None, :]
    cos_theta_h = float(np.cos(theta_h))

    use_keepout = (p_em is not None) and (float(R_em) > 0.0)
    if use_keepout:
        p_em = np.asarray(p_em, dtype=float).reshape(3,)

    for i in range(M):
        r_mean = p_hat - p_agents[i]
        nr = float(np.linalg.norm(r_mean))
        if nr < eps:
            continue
        u_des = r_mean / nr
        u_mean[i] = u_des

        u_curr = u_curr_agents[i]
        nu = float(np.linalg.norm(u_curr))
        if nu < eps:
            continue
        u_curr = u_curr / nu

        theta_req[i] = float(np.arccos(np.clip(np.dot(u_curr, u_des), -1.0, 1.0)))
        if theta_req[i] > float(theta_s_list[i]) + 1e-12:
            continue

        if use_keepout and not keepout_safe_single(
            p_agents[i], u_des, float(theta_h), p_em, float(R_em), float(alpha_s)
        ):
            continue

        v = p_samples - p_agents[i][None, :]
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        v_unit = v / np.maximum(v_norm, eps)
        cos_ang = np.einsum('ij,j->i', v_unit, u_des)
        flags[i] = bool(np.all(cos_ang >= cos_theta_h - 1e-12))

    return flags, u_mean, theta_req


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



def build_free_indices(M: int, idx_fix: Optional[int]) -> np.ndarray:
    """Return array of agent indices that remain free (all except idx_fix)."""
    if idx_fix is None:
        return np.arange(M, dtype=int)
    if not (0 <= int(idx_fix) < int(M)):
        raise ValueError(f"idx_fix out of range: {idx_fix} for M={M}")
    return np.array([i for i in range(M) if i != int(idx_fix)], dtype=int)


def unpack_angles_free(x_free: np.ndarray, M_free: int) -> Tuple[np.ndarray, np.ndarray]:
    """x_free = [theta,phi,...] for free agents -> (thetas, phis)."""
    x_free = np.asarray(x_free, dtype=float).ravel()
    if x_free.size != 2 * int(M_free):
        raise ValueError(f"x_free length {x_free.size} != 2*M_free ({2*int(M_free)})")
    thetas = x_free[0::2]
    phis = x_free[1::2]
    return thetas, phis


def angles_to_pointings_with_fixed(
    x_free: np.ndarray,
    p_agents: np.ndarray,
    u_curr_agents: np.ndarray,
    theta_lower: np.ndarray,
    theta_upper: np.ndarray,
    *,
    free_idx: np.ndarray,
    idx_fix: int,
    u_fix: np.ndarray,
) -> np.ndarray:
    """Build full u_agents (M,3) from reduced x_free with one fixed agent."""
    p_agents = np.asarray(p_agents, dtype=float)
    u_curr_agents = np.asarray(u_curr_agents, dtype=float)
    M = int(p_agents.shape[0])
    if u_curr_agents.shape != (M, 3):
        raise ValueError(f"u_curr_agents must be (M,3); got {u_curr_agents.shape}")

    free_idx = np.asarray(free_idx, dtype=int).ravel()
    if free_idx.size != M - 1:
        raise ValueError("free_idx must have size M-1 when using a fixed agent")

    idx_fix = int(idx_fix)
    if not (0 <= idx_fix < M):
        raise ValueError(f"idx_fix out of range: {idx_fix} for M={M}")

    u_agents = np.zeros((M, 3), dtype=float)

    # fixed boresight
    u0 = np.asarray(u_fix, dtype=float).reshape(3,)
    u0 = u0 / max(np.linalg.norm(u0), 1e-12)
    u_agents[idx_fix] = u0

    # free boresights
    thetas, phis = unpack_angles_free(x_free, int(free_idx.size))
    for k, i in enumerate(free_idx):
        i = int(i)
        th = float(np.clip(thetas[k], theta_lower[i], theta_upper[i]))
        ph = float(phis[k] % (2 * np.pi))
        u_agents[i] = u_from_cap(u_curr_agents[i], th, ph)

    return u_agents


def objective_joint_free(
    x_free: np.ndarray,
    p_hat: np.ndarray,
    P_p: np.ndarray,
    p_agents: np.ndarray,
    u_curr_agents: np.ndarray,
    theta_lower: np.ndarray,
    theta_upper: np.ndarray,
    theta_h: float,
    d_M: float,
    kappa_sigma: float,
    lambda_k1: float,
    y_cached: np.ndarray,
    *,
    free_idx: np.ndarray,
    idx_fix: int,
    u_fix: np.ndarray,
    p_em=None,
    R_em: float = 0.0,
    alpha_s: float = 0.0,
    lambda_em: float = 0.0,
    beta_zeta: float = 50.0,
    coverage_mode: str = "exact2_plus_k1",
) -> float:
    """Same as objective_joint, but with one fixed agent and reduced x."""
    u_agents = angles_to_pointings_with_fixed(
        x_free, p_agents, u_curr_agents, theta_lower, theta_upper,
        free_idx=free_idx, idx_fix=idx_fix, u_fix=u_fix
    )

    J = J_t_dual_coverage(
        p_hat, P_p, p_agents, u_agents,
        theta_h, d_M=d_M, kappa_sigma=kappa_sigma, lambda_k1=lambda_k1,
        n_mc=y_cached.shape[0], y_samples_cached=y_cached,
        coverage_mode=coverage_mode
    )
    obj = -J

    if lambda_em != 0.0 and p_em is not None and R_em > 0.0:
        obj += ems_exclusion_penalty(
            p_agents, u_agents,
            p_em=p_em, R_em=R_em,
            theta_h=theta_h,
            alpha_s=alpha_s,
            lambda_em=lambda_em,
            beta_zeta=beta_zeta
        )
    return float(obj)


def make_cached_y(P_p, d_M, n_mc, seed):
    """Pre-draw y samples once to stabilize finite-diff gradients."""
    rng = np.random.default_rng(seed)
    y = sample_uniform_ball(n_mc, radius=d_M, dim=3, rng=rng)
    return y



def objective_joint(x, p_hat, P_p, p_agents, u_curr_agents,
                    theta_lower, theta_upper,
                    theta_h, d_M, kappa_sigma, lambda_k1,
                    y_cached,
                    p_em=None, R_em=0.0,
                    alpha_s=0.0, lambda_em=0.0,
                    beta_zeta=50.0,
                    coverage_mode="exact2_plus_k1"):
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
        theta_h, d_M=d_M, kappa_sigma=kappa_sigma, lambda_k1=lambda_k1,
        n_mc=y_cached.shape[0], y_samples_cached=y_cached,
        coverage_mode=coverage_mode
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


def init_theta_phi_boundary_projection_old(
    p_hat,
    P_p,
    p_agents,
    u_curr_agents,
    theta_lower,
    theta_upper,
    theta_h,
    d_M,
    kappa_sigma,
    lambda_k1,
    y_cached,
    p_em,
    R_em,
    alpha_s,
    seed,
    jitter,
    eps=1e-10,
    n_boundary_candidates=2,
    coverage_mode="exact2_plus_k1",
):
    """
    3D warm-start initializer with EMS keep-out via boundary projection.

    NEW behavior:
      - When a spacecraft violates EMS keep-out and we generate boundary candidates,
        we FIRST keep only those candidates whose *central ray* intersects the
        target uncertainty ellipsoid (Mahalanobis d_M level set).
      - If NONE of the candidates intersect, we fall back to using ALL candidates.

    Notes:
      - "Intersects uncertainty ellipsoid" here means the ray
            x(t) = p_i + t*u,  t >= 0
        has at least one solution to
            (x - p_hat)^T P_p^{-1} (x - p_hat) = d_M^2
        (i.e., discriminant >= 0 and some root t >= 0).
      - This is a fast geometric filter (heuristic). It does not account for the
        full cone half-angle theta_h; it checks the boresight ray only.
    """

    rng = np.random.default_rng(seed=seed)
    jitter_values = np.deg2rad(jitter)

    p_hat = np.asarray(p_hat, dtype=float).reshape(3,)
    P_p = np.asarray(P_p, dtype=float).reshape(3, 3)
    p_agents = np.asarray(p_agents, dtype=float)
    u_curr_agents = np.asarray(u_curr_agents, dtype=float)
    theta_lower = np.asarray(theta_lower, dtype=float)
    theta_upper = np.asarray(theta_upper, dtype=float)

    M = p_agents.shape[0]
    x0 = np.zeros(2 * M, dtype=float)
    n_mc = y_cached.shape[0]

    # Precompute inverse covariance (stable)
    try:
        P_inv = np.linalg.inv(P_p)
    except np.linalg.LinAlgError:
        P_inv = np.linalg.pinv(P_p)

    # --------------------------------------------------
    # Helper: ray-ellipsoid intersection (boresight ray)
    # --------------------------------------------------
    def ray_intersects_ellipsoid(p0, u, mu, P_inv, d, eps=1e-12):
        """
        Returns True if exists t >= 0 such that:
          (p0 + t u - mu)^T P_inv (p0 + t u - mu) = d^2
        """
        p0 = np.asarray(p0, dtype=float).reshape(3,)
        u = np.asarray(u, dtype=float).reshape(3,)
        mu = np.asarray(mu, dtype=float).reshape(3,)

        nu = np.linalg.norm(u)
        if nu < eps:
            return False
        u = u / nu

        w = p0 - mu

        a = float(u.T @ P_inv @ u)
        b = float(2.0 * (u.T @ P_inv @ w))
        c = float(w.T @ P_inv @ w - (float(d) ** 2))

        if abs(a) < eps:
            # Degenerate (shouldn't happen if P_inv is PD and u nonzero), treat as linear
            if abs(b) < eps:
                return False
            t = -c / b
            return t >= 0.0

        disc = b*b - 4.0*a*c
        if disc < 0.0:
            return False

        sdisc = float(np.sqrt(max(disc, 0.0)))
        t1 = (-b - sdisc) / (2.0*a)
        t2 = (-b + sdisc) / (2.0*a)

        return (t1 >= 0.0) or (t2 >= 0.0)

    # --------------------------------------------------
    # Helper: single-ray keep-out check for spacecraft i
    # --------------------------------------------------
    def keepout_safe_single(p_i, u_i, theta_h, p_em, R_em, alpha_s, eps=1e-12):
        if p_em is None or float(R_em) <= 0.0:
            return True
        r_vec = p_em - p_i
        r_norm = np.linalg.norm(r_vec)
        if r_norm < R_em + eps:
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
        for ii in range(p_agents.shape[0]):
            if not keepout_safe_single(p_agents[ii], u_array[ii], theta_h, p_em, R_em, alpha_s):
                return False
        return True

    # --------------------------------------------------
    # Helper: local parameterization u(theta, phi)
    # --------------------------------------------------
    def u_from_theta_phi(theta, phi, u_curr, e1, e2):
        phi = float(phi) % (2 * np.pi)
        return (
            np.cos(theta) * u_curr
            + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
        )

    # --------------------------------------------------
    # Precompute local bases and mean-pointing directions u_star[i]
    # --------------------------------------------------
    u_curr_norm = np.zeros_like(u_curr_agents, dtype=float)
    e1_list = np.zeros_like(u_curr_agents, dtype=float)
    e2_list = np.zeros_like(u_curr_agents, dtype=float)
    theta_star = np.zeros(M, dtype=float)
    phi_star = np.zeros(M, dtype=float)
    u_star = np.zeros_like(u_curr_agents, dtype=float)

    for i in range(M):
        p_i = p_agents[i]
        u_curr = u_curr_agents[i]
        n_u = np.linalg.norm(u_curr)
        if n_u < eps:
            u_curr = np.array([0.0, 0.0, 1.0], dtype=float)
            n_u = 1.0
        u_curr = u_curr / n_u
        u_curr_norm[i] = u_curr

        d_vec = p_hat - p_i
        dist = np.linalg.norm(d_vec)
        v_des = u_curr if dist < eps else (d_vec / dist)

        e1, e2 = orthonormal_basis_from_u(u_curr)
        e1_list[i] = e1
        e2_list[i] = e2

        a = float(np.dot(v_des, u_curr))
        b1 = float(np.dot(v_des, e1))
        b2 = float(np.dot(v_des, e2))
        s = float(np.sqrt(b1*b1 + b2*b2))

        theta_i = float(np.arctan2(s, a))
        phi_i = float(np.arctan2(b2, b1)) % (2 * np.pi)

        theta_i = float(np.clip(theta_i, theta_lower[i], theta_upper[i]))

        theta_star[i] = theta_i
        phi_star[i] = phi_i
        u_star[i] = u_from_theta_phi(theta_i, phi_i, u_curr, e1, e2)

    # --------------------------------------------------
    # Step 1: If mean-based config is EMS-safe, use original style + jitter
    # --------------------------------------------------
    if config_keepout_safe(p_agents, u_star, theta_h, p_em, R_em, alpha_s):
        for i in range(M):
            jitter = float(rng.choice(jitter_values))
            theta_i = float(np.clip(theta_star[i] + jitter, theta_lower[i], theta_upper[i]))

            u_i = u_from_theta_phi(theta_i, phi_star[i], u_curr_norm[i], e1_list[i], e2_list[i])
            if not keepout_safe_single(p_agents[i], u_i, theta_h, p_em, R_em, alpha_s):
                theta_i = float(theta_star[i])

            x0[2 * i] = theta_i
            x0[2 * i + 1] = float(phi_star[i] % (2 * np.pi))
        return x0

    # --------------------------------------------------
    # Step 2: Mean-based config not safe -> candidates per spacecraft
    # --------------------------------------------------
    candidate_u_list = []
    N = int(max(1, n_boundary_candidates))

    for i in range(M):
        p_i = p_agents[i]
        u_curr = u_curr_norm[i]
        u_i_star = u_star[i]

        # EMS geometry
        r_vec = p_em - p_i
        r_norm = np.linalg.norm(r_vec)
        if r_norm < R_em + eps:
            candidate_u_list.append([u_i_star])
            continue

        v_em = r_vec / r_norm
        ratio = np.clip(R_em / r_norm, -1.0, 1.0)
        alpha_em = np.arcsin(ratio)
        gamma_bound = float(theta_h + alpha_em + alpha_s)

        if gamma_bound >= np.pi - 1e-6:
            candidate_u_list.append([u_i_star])
            continue

        # If u_star is safe, keep it (single candidate)
        if keepout_safe_single(p_i, u_i_star, theta_h, p_em, R_em, alpha_s):
            candidate_u_list.append([u_i_star])
            continue

        # Unsafe: boundary ring candidates (evenly spaced)
        cos_gb = float(np.cos(gamma_bound))
        sin_gb = float(np.sin(gamma_bound))

        e1_em, e2_em = orthonormal_basis_from_u(v_em)

        # Phase-align to u_star's perpendicular component
        c = float(np.dot(u_i_star, v_em))
        v_perp = u_i_star - c * v_em
        n_perp = float(np.linalg.norm(v_perp))
        if n_perp < eps:
            psi0 = 0.0
        else:
            v_perp_hat = v_perp / n_perp
            a1 = float(np.dot(v_perp_hat, e1_em))
            a2 = float(np.dot(v_perp_hat, e2_em))
            psi0 = float(np.arctan2(a2, a1))

        # 1) generate all candidates (slew-feasible)
        all_dirs = []
        for k in range(N):
            psi = psi0 + 2.0 * np.pi * (k / N)
            u_b = cos_gb * v_em + sin_gb * (np.cos(psi) * e1_em + np.sin(psi) * e2_em)
            u_b = u_b / max(np.linalg.norm(u_b), eps)

            # slew feasibility
            cos_theta = np.clip(np.dot(u_b, u_curr), -1.0, 1.0)
            theta_b = float(np.arccos(cos_theta))
            if theta_b < theta_lower[i] - 1e-6 or theta_b > theta_upper[i] + 1e-6:
                continue

            all_dirs.append(u_b)

        if len(all_dirs) == 0:
            candidate_u_list.append([u_i_star])
            continue

        # 2) prefer candidates whose boresight ray intersects the uncertainty ellipsoid
        hit_dirs = []
        for u_b in all_dirs:
            if ray_intersects_ellipsoid(p_i, u_b, p_hat, P_inv, d_M, eps=1e-12):
                hit_dirs.append(u_b)

        # If we have any "hits", use only those; otherwise use all
        candidate_u_list.append(hit_dirs if (len(hit_dirs) > 0) else all_dirs)

    # --------------------------------------------------
    # Step 3: Enumerate combinations and pick best J_t
    # --------------------------------------------------
    best_J = -np.inf
    best_u = None

    index_ranges = [range(len(cands)) for cands in candidate_u_list]

    for choice in itertools.product(*index_ranges):
        u_trial = np.zeros_like(u_curr_agents, dtype=float)
        for ii, idx in enumerate(choice):
            u_trial[ii] = candidate_u_list[ii][idx]

        J_val = J_t_dual_coverage(
            p_hat,
            P_p,
            p_agents,
            u_trial,
            theta_h,
            d_M=d_M,
            kappa_sigma=kappa_sigma,
            lambda_k1=lambda_k1,
            n_mc=n_mc,
            y_samples_cached=y_cached,
            coverage_mode=coverage_mode,
        )

        if J_val > best_J:
            best_J = float(J_val)
            best_u = u_trial.copy()

    if best_u is None:
        best_u = u_star.copy()

    # --------------------------------------------------
    # Step 4: Convert best_u back to (theta_i, phi_i), then apply safe jitter
    # --------------------------------------------------
    for i in range(M):
        u_curr = u_curr_norm[i]
        e1 = e1_list[i]
        e2 = e2_list[i]
        u_i = best_u[i]

        a = float(np.dot(u_i, u_curr))
        b1 = float(np.dot(u_i, e1))
        b2 = float(np.dot(u_i, e2))
        s = float(np.sqrt(b1*b1 + b2*b2))

        theta_i = float(np.arctan2(s, a))
        phi_i = float(np.arctan2(b2, b1)) % (2 * np.pi)

        theta_i = float(np.clip(theta_i, theta_lower[i], theta_upper[i]))

        jitter = float(rng.choice(jitter_values))
        theta_j = float(np.clip(theta_i + jitter, theta_lower[i], theta_upper[i]))
        u_j = u_from_theta_phi(theta_j, phi_i, u_curr, e1, e2)

        if keepout_safe_single(p_agents[i], u_j, theta_h, p_em, R_em, alpha_s):
            theta_i = theta_j

        x0[2 * i] = theta_i
        x0[2 * i + 1] = phi_i

    return x0


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
    lambda_k1,
    y_cached,
    p_em,
    R_em,
    alpha_s,
    seed,
    jitter,
    eps=1e-10,
    n_boundary_candidates=2,
    coverage_mode="exact2_plus_k1",
    *,
    # Backward-compatible single-detector inputs
    detecting_idx=None,
    detecting_u=None,

    # Multi-detector inputs
    detecting_indices=None,
    detecting_us=None,

    detecting_mode="mean",
    n_uncertainty_candidates=64,
    n_los_candidates=64,
    return_log=False,
):
    """
    Sequential warm-start initializer.

    Logic:
      0. Assign detecting spacecraft first, if given.
      1. For each unassigned spacecraft, try pointing at mean.
      2. For remaining spacecraft, try uncertainty candidates near mean first.
      3. For remaining spacecraft, try points along a detecting spacecraft LOS.
      4. For remaining spacecraft, generate EMS-boundary candidates and choose
         the best J_t combination.

    Supports:
      - Single detector:
            detecting_idx=3, detecting_u=u3

      - Multiple detectors:
            detecting_indices=[3, 5], detecting_us=[u3, u5]

    Important logging / jitter behavior:
      - Each spacecraft is jittered at most once.
      - Jitter is applied only when a spacecraft is accepted/finalized.
      - The returned candidate_log has exactly one accepted=True row per spacecraft.
      - Accepted rows store the final u/theta/phi that correspond to x0.
    """

    rng = np.random.default_rng(seed=seed)
    jitter_values = np.atleast_1d(np.deg2rad(jitter)).astype(float)

    p_hat = np.asarray(p_hat, dtype=float).reshape(3,)
    P_p = np.asarray(P_p, dtype=float).reshape(3, 3)
    p_agents = np.asarray(p_agents, dtype=float)
    u_curr_agents = np.asarray(u_curr_agents, dtype=float)
    theta_lower = np.asarray(theta_lower, dtype=float)
    theta_upper = np.asarray(theta_upper, dtype=float)

    M = p_agents.shape[0]
    x0 = np.zeros(2 * M, dtype=float)
    n_mc = y_cached.shape[0]
    candidate_log = []

    try:
        P_inv = np.linalg.inv(P_p)
    except np.linalg.LinAlgError:
        P_inv = np.linalg.pinv(P_p)

    jitter_applied = np.zeros(M, dtype=bool)
    accepted_log_idx_by_sc = [None for _ in range(M)]

    def _unit(v):
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        if n < eps:
            return None
        return v / n

    def normalize_detector_inputs():
        """
        Normalize detector inputs into:
            detector_indices : list[int]
            detector_us      : list[np.ndarray or None]

        Supports both old single-detector API:
            detecting_idx=3, detecting_u=u3

        and new multi-detector API:
            detecting_indices=[3, 5], detecting_us=[u3, u5]
        """

        if detecting_indices is not None:
            detector_indices = list(np.atleast_1d(detecting_indices).astype(int))

            if detecting_us is None:
                detector_us = [None for _ in detector_indices]
            else:
                detector_us = list(detecting_us)

                if len(detector_us) != len(detector_indices):
                    raise ValueError(
                        "detecting_us must have the same length as detecting_indices. "
                        f"Got len(detecting_us)={len(detector_us)}, "
                        f"len(detecting_indices)={len(detector_indices)}."
                    )

            return detector_indices, detector_us

        if detecting_idx is not None:
            return [int(detecting_idx)], [detecting_u]

        return [], []

    def keepout_safe_single_local(p_i, u_i):
        if p_em is None or float(R_em) <= 0.0:
            return True

        r_vec = p_em - p_i
        r_norm = np.linalg.norm(r_vec)

        if r_norm < R_em + eps:
            return False

        v_em = r_vec / r_norm
        gamma = np.arccos(np.clip(np.dot(u_i, v_em), -1.0, 1.0))
        alpha_em = np.arcsin(np.clip(R_em / r_norm, -1.0, 1.0))

        return gamma >= (theta_h + alpha_em + alpha_s)

    def u_to_theta_phi(u_des, u_curr, e1, e2):
        u_des = _unit(u_des)
        if u_des is None:
            return None, None

        a = float(np.dot(u_des, u_curr))
        b1 = float(np.dot(u_des, e1))
        b2 = float(np.dot(u_des, e2))
        s = float(np.sqrt(b1 * b1 + b2 * b2))

        theta = float(np.arctan2(s, a))
        phi = float(np.arctan2(b2, b1)) % (2 * np.pi)

        return theta, phi

    def theta_phi_to_u(theta, phi, u_curr, e1, e2):
        phi = float(phi) % (2 * np.pi)

        u = (
            np.cos(theta) * u_curr
            + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
        )

        return u / max(np.linalg.norm(u), eps)

    def ray_intersects_ellipsoid(p0, u, mu, P_inv, d):
        u = _unit(u)
        if u is None:
            return False

        w = p0 - mu

        a = float(u.T @ P_inv @ u)
        b = float(2.0 * (u.T @ P_inv @ w))
        c = float(w.T @ P_inv @ w - d**2)

        if abs(a) < eps:
            if abs(b) < eps:
                return False

            t = -c / b
            return t >= 0.0

        disc = b * b - 4.0 * a * c

        if disc < 0.0:
            return False

        sdisc = float(np.sqrt(max(disc, 0.0)))
        t1 = (-b - sdisc) / (2.0 * a)
        t2 = (-b + sdisc) / (2.0 * a)

        return (t1 >= 0.0) or (t2 >= 0.0)

    def apply_jitter_once_if_valid(
        i,
        theta_i,
        phi_i,
        u_i,
        *,
        require_ellipsoid_hit=False,
    ):
        """
        Apply jitter to spacecraft i at most once.

        If the jittered direction is invalid, keep the original direction and do
        not mark jitter as applied.
        """

        if jitter_applied[i]:
            return u_i, theta_i, phi_i

        if jitter_values.size == 0:
            return u_i, theta_i, phi_i

        jitter_i = float(rng.choice(jitter_values))

        theta_j = float(np.clip(theta_i + jitter_i, theta_lower[i], theta_upper[i]))
        phi_j = float(phi_i) % (2 * np.pi)

        u_j = theta_phi_to_u(
            theta_j,
            phi_j,
            u_curr_norm[i],
            e1_list[i],
            e2_list[i],
        )

        if not keepout_safe_single_local(p_agents[i], u_j):
            return u_i, theta_i, phi_i

        if require_ellipsoid_hit:
            if not ray_intersects_ellipsoid(p_agents[i], u_j, p_hat, P_inv, d_M):
                return u_i, theta_i, phi_i

        jitter_applied[i] = True
        return u_j, theta_j, phi_j

    def mark_accepted_log_row(i, log_idx, u_i, theta_i, phi_i, stage, reason="accepted"):
        """
        Mark exactly one log row as accepted for spacecraft i.
        If this spacecraft had a previously accepted row, unaccept it.
        """

        old_idx = accepted_log_idx_by_sc[i]

        if old_idx is not None and 0 <= old_idx < len(candidate_log):
            candidate_log[old_idx]["accepted"] = False
            candidate_log[old_idx]["reason"] = "superseded_by_later_acceptance"

        candidate_log[log_idx]["stage"] = stage
        candidate_log[log_idx]["sc_idx"] = int(i)
        candidate_log[log_idx]["u"] = np.asarray(u_i, dtype=float).reshape(3,).copy()
        candidate_log[log_idx]["theta"] = float(theta_i)
        candidate_log[log_idx]["phi"] = float(phi_i) % (2 * np.pi)
        candidate_log[log_idx]["accepted"] = True
        candidate_log[log_idx]["reason"] = reason

        accepted_log_idx_by_sc[i] = log_idx

    def append_final_accepted_row(i, u_i, theta_i, phi_i, stage, reason):
        """
        Append a final accepted row for spacecraft i.
        Used mainly for Step 4 final-combination choices.
        """

        old_idx = accepted_log_idx_by_sc[i]

        if old_idx is not None and 0 <= old_idx < len(candidate_log):
            candidate_log[old_idx]["accepted"] = False
            candidate_log[old_idx]["reason"] = "superseded_by_final_solution"

        candidate_log.append({
            "stage": stage,
            "sc_idx": int(i),
            "p_target": p_agents[i] + np.asarray(u_i, dtype=float).reshape(3,),
            "u": np.asarray(u_i, dtype=float).reshape(3,).copy(),
            "theta": float(theta_i),
            "phi": float(phi_i) % (2 * np.pi),
            "accepted": True,
            "reason": reason,
        })

        accepted_log_idx_by_sc[i] = len(candidate_log) - 1

    def accept_candidate(i, u_i, theta_i, phi_i, stage, log_idx=None):
        """
        Accept a candidate into u_init/theta_init/phi_init and update the log.

        This does not itself apply jitter. Jitter should be applied before this
        function is called.
        """

        u_i = np.asarray(u_i, dtype=float).reshape(3,)

        u_init[i] = u_i.copy()
        theta_init[i] = float(theta_i)
        phi_init[i] = float(phi_i) % (2 * np.pi)
        assigned[i] = True

        if log_idx is not None and 0 <= log_idx < len(candidate_log):
            mark_accepted_log_row(i, log_idx, u_i, theta_i, phi_i, stage)

    def ensure_exactly_one_accepted_per_spacecraft():
        """
        Enforce exactly one accepted=True row per spacecraft.
        """

        keep = set(idx for idx in accepted_log_idx_by_sc if idx is not None)

        for kk, row in enumerate(candidate_log):
            if kk not in keep and row.get("accepted", False):
                row["accepted"] = False
                row["reason"] = "not_final_accepted_choice"

        for i, idx in enumerate(accepted_log_idx_by_sc):
            if idx is not None and 0 <= idx < len(candidate_log):
                candidate_log[idx]["accepted"] = True

        n_acc = sum(1 for row in candidate_log if row.get("accepted", False))

        if n_acc != M:
            raise RuntimeError(
                f"candidate_log should have exactly one accepted row per spacecraft. "
                f"Found {n_acc}, expected {M}."
            )

    def _finish():
        for ii in range(M):
            x0[2 * ii] = theta_init[ii]
            x0[2 * ii + 1] = phi_init[ii] % (2 * np.pi)

        if return_log:
            ensure_exactly_one_accepted_per_spacecraft()
            return x0, candidate_log

        return x0

    def try_point_to_point(i, p_target, stage, log_candidate=True):
        """
        Try pointing spacecraft i at p_target.

        Returns:
            ok, u_i, theta_i, phi_i, log_idx
        """

        log_idx = None

        p_i = p_agents[i]
        u_curr = u_curr_norm[i]
        e1 = e1_list[i]
        e2 = e2_list[i]

        u_des = _unit(p_target - p_i)

        if u_des is None:
            return False, None, None, None, log_idx

        theta, phi = u_to_theta_phi(u_des, u_curr, e1, e2)

        if theta is None:
            return False, None, None, None, log_idx

        slew_ok = (
            theta >= theta_lower[i] - 1e-12
            and theta <= theta_upper[i] + 1e-12
        )
        if not slew_ok:
            if log_candidate:
                candidate_log.append({
                    "stage": stage,
                    "sc_idx": int(i),
                    "p_target": np.asarray(p_target, dtype=float).copy(),
                    "u": u_des.copy(),
                    "theta": float(theta),
                    "phi": float(phi) % (2 * np.pi),
                    "accepted": False,
                    "reason": "slew_infeasible",
                })
                log_idx = len(candidate_log) - 1

            return False, None, None, None, log_idx

        theta = float(np.clip(theta, theta_lower[i], theta_upper[i]))
        phi = float(phi) % (2 * np.pi)

        u_i = theta_phi_to_u(theta, phi, u_curr, e1, e2)

        ems_ok = keepout_safe_single_local(p_i, u_i)
        if log_candidate:
            candidate_log.append({
                "stage": stage,
                "sc_idx": int(i),
                "p_target": np.asarray(p_target, dtype=float).copy(),
                "u": u_i.copy(),
                "theta": float(theta),
                "phi": float(phi) % (2 * np.pi),
                "accepted": False,
                "reason": "ems_infeasible" if not ems_ok else "candidate",
            })
            log_idx = len(candidate_log) - 1

        if not ems_ok:
            return False, None, None, None, log_idx

        return True, u_i, theta, phi, log_idx

    def detector_boresight_projected_candidates(
        p_hat,
        P_p,
        p_det,
        u_det,
        theta_h,
        d_M,
        n_candidates=64,
        n_shell=1200,
        seed=0,
        eps=1e-12,
    ):
        rng_local = np.random.default_rng(seed)

        u_det = np.asarray(u_det, dtype=float).reshape(3,)
        u_det = u_det / max(np.linalg.norm(u_det), eps)

        Lp = np.linalg.cholesky(P_p)

        dirs = rng_local.normal(size=(n_shell, 3))
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), eps)

        p_shell = p_hat[None, :] + (Lp @ (d_M * dirs).T).T

        r = p_shell - p_det[None, :]
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)
        r_unit = r / np.maximum(r_norm, eps)


        s_vals_shell = (p_shell - p_det[None, :]) @ u_det
        s_vals_shell = s_vals_shell[s_vals_shell >= 0.0]

        if s_vals_shell.size == 0:
            return np.empty((0, 3))

        s_max = float(np.max(s_vals_shell))

        s_raw = np.linspace(0.0, s_max, int(max(2, n_candidates)))
        p_candidates = p_det[None, :] + s_raw[:, None] * u_det[None, :]

        order = np.argsort(np.linalg.norm(p_candidates - p_hat[None, :], axis=1))
        return p_candidates[order]

    # --------------------------------------------------
    # Precompute local bases
    # --------------------------------------------------
    u_curr_norm = np.zeros_like(u_curr_agents, dtype=float)
    e1_list = np.zeros_like(u_curr_agents, dtype=float)
    e2_list = np.zeros_like(u_curr_agents, dtype=float)

    for i in range(M):
        u_curr = _unit(u_curr_agents[i])

        if u_curr is None:
            u_curr = np.array([0.0, 0.0, 1.0], dtype=float)

        u_curr_norm[i] = u_curr

        e1, e2 = orthonormal_basis_from_u(u_curr)
        e1_list[i] = e1
        e2_list[i] = e2

    u_init = np.zeros_like(u_curr_agents, dtype=float)
    theta_init = np.zeros(M, dtype=float)
    phi_init = np.zeros(M, dtype=float)
    assigned = np.zeros(M, dtype=bool)

    # --------------------------------------------------
    # Step 0: assign detecting spacecraft
    # --------------------------------------------------
    detector_indices, detector_us = normalize_detector_inputs()

    seen_detectors = set()

    for det_idx, det_u in zip(detector_indices, detector_us):
        det_idx = int(det_idx)

        if not (0 <= det_idx < M):
            raise ValueError(f"detecting_idx out of range: {det_idx} for M={M}")

        if det_idx in seen_detectors:
            raise ValueError(f"Duplicate detecting spacecraft index: {det_idx}")

        seen_detectors.add(det_idx)

        if det_u is not None:
            u_det_cmd = _unit(np.asarray(det_u, dtype=float).reshape(3,))
        elif detecting_mode == "current":
            u_det_cmd = u_curr_norm[det_idx]
        else:
            u_det_cmd = _unit(p_hat - p_agents[det_idx])

        if u_det_cmd is None:
            u_det_cmd = u_curr_norm[det_idx]

        theta_req, phi_req = u_to_theta_phi(
            u_det_cmd,
            u_curr_norm[det_idx],
            e1_list[det_idx],
            e2_list[det_idx],
        )

        if theta_req is None:
            theta_req = 0.0
            phi_req = 0.0

        theta_req = float(theta_req)
        phi_req = float(phi_req) % (2 * np.pi)

        detector_clipped = (
            theta_req < theta_lower[det_idx] - 1e-12
            or theta_req > theta_upper[det_idx] + 1e-12
        )

        theta_encoded = float(np.clip(theta_req, theta_lower[det_idx], theta_upper[det_idx]))

        u_det_encoded = theta_phi_to_u(
            theta_encoded,
            phi_req,
            u_curr_norm[det_idx],
            e1_list[det_idx],
            e2_list[det_idx],
        )

        u_init[det_idx] = u_det_encoded.copy()
        theta_init[det_idx] = theta_encoded
        phi_init[det_idx] = phi_req
        assigned[det_idx] = True

        reason = (
            "detecting_spacecraft_assigned_theta_clipped"
            if detector_clipped
            else "detecting_spacecraft_assigned"
        )

        candidate_log.append({
            "stage": "detector",
            "sc_idx": int(det_idx),
            "p_target": p_hat.copy(),
            "u": u_det_encoded.copy(),
            "u_requested": u_det_cmd.copy(),
            "theta": float(theta_encoded),
            "theta_requested": float(theta_req),
            "phi": float(phi_req),
            "accepted": True,
            "reason": reason,
        })

        accepted_log_idx_by_sc[det_idx] = len(candidate_log) - 1

    # --------------------------------------------------
    # Step 1: mean candidate
    # --------------------------------------------------
    for i in range(M):
        if assigned[i]:
            continue

        ok, u_i, theta_i, phi_i, log_idx = try_point_to_point(
            i,
            p_hat,
            stage="mean",
            log_candidate=True,
        )

        if ok:
            u_i, theta_i, phi_i = apply_jitter_once_if_valid(
                i,
                theta_i,
                phi_i,
                u_i,
                require_ellipsoid_hit=False,
            )

            accept_candidate(
                i,
                u_i,
                theta_i,
                phi_i,
                stage="mean",
                log_idx=log_idx,
            )

    if np.all(assigned):
        return _finish()

    # --------------------------------------------------
    # Step 2: uncertainty candidates near mean first
    # --------------------------------------------------
    try:
        Lp = np.linalg.cholesky(P_p)
    except np.linalg.LinAlgError:
        Lp = np.linalg.cholesky(P_p + 1e-12 * np.eye(3))

    rng_unc = np.random.default_rng(seed + 456)

    n_unc = int(max(1, n_uncertainty_candidates))
    y_unc = sample_uniform_ball(n_unc, radius=d_M, dim=3, rng=rng_unc)
    y_unc = np.vstack([np.zeros((1, 3)), y_unc])

    p_unc = (Lp @ y_unc.T).T + p_hat[None, :]
    dist_unc = np.linalg.norm(p_unc - p_hat[None, :], axis=1)
    p_unc = p_unc[np.argsort(dist_unc)]

    for i in range(M):
        if assigned[i]:
            continue

        best = None
        best_dist = np.inf
        best_log_idx = None

        for p_cand in p_unc:
            ok, u_i, theta_i, phi_i, log_idx = try_point_to_point(
                i,
                p_cand,
                stage="uncertainty",
                log_candidate=True,
            )

            if not ok:
                continue

            if not ray_intersects_ellipsoid(p_agents[i], u_i, p_hat, P_inv, d_M):
                if log_idx is not None:
                    candidate_log[log_idx]["reason"] = "ray_misses_ellipsoid"
                continue

            dmean = float(np.linalg.norm(p_cand - p_hat))

            if dmean < best_dist:
                best = (u_i, theta_i, phi_i)
                best_dist = dmean
                best_log_idx = log_idx

        if best is not None:
            u_i, theta_i, phi_i = best

            u_i, theta_i, phi_i = apply_jitter_once_if_valid(
                i,
                theta_i,
                phi_i,
                u_i,
                require_ellipsoid_hit=True,
            )

            accept_candidate(
                i,
                u_i,
                theta_i,
                phi_i,
                stage="uncertainty",
                log_idx=best_log_idx,
            )

    if np.all(assigned):
        return _finish()

    # --------------------------------------------------
    # Step 3: detecting spacecraft LOS candidates
    # --------------------------------------------------
    if len(detector_indices) > 0:
        # Use first detector as LOS reference.
        # If you later want to use all detector LOS rays, this can be generalized.
        los_ref_idx = int(detector_indices[0])
        p_det = p_agents[los_ref_idx]
        los_ref_u = detector_us[0]

        if los_ref_u is not None:
            u_det = _unit(np.asarray(los_ref_u, dtype=float).reshape(3,))
        else:
            u_det = _unit(p_hat - p_det)

        if u_det is not None:
            if np.dot(u_det, p_hat - p_det) < 0.0:
                u_det = -u_det

            p_los_candidates = detector_boresight_projected_candidates(
                p_hat,
                P_p,
                p_det,
                u_det,
                theta_h,
                d_M,
                n_candidates=n_los_candidates,
                n_shell=max(4 * n_los_candidates, 1200),
                seed=seed + 789,
            )

            for i in range(M):
                if assigned[i]:
                    continue

                best = None
                best_dist = np.inf
                best_log_idx = None

                for p_los in p_los_candidates:
                    ok, u_i, theta_i, phi_i, log_idx = try_point_to_point(
                        i,
                        p_los,
                        stage="los",
                        log_candidate=True,
                    )

                    if not ok:
                        continue

                    dmean = float(np.linalg.norm(p_los - p_hat))

                    if dmean < best_dist:
                        best = (u_i, theta_i, phi_i)
                        best_dist = dmean
                        best_log_idx = log_idx

                if best is not None:
                    u_i, theta_i, phi_i = best

                    u_i, theta_i, phi_i = apply_jitter_once_if_valid(
                        i,
                        theta_i,
                        phi_i,
                        u_i,
                        require_ellipsoid_hit=False,
                    )

                    accept_candidate(
                        i,
                        u_i,
                        theta_i,
                        phi_i,
                        stage="los",
                        log_idx=best_log_idx,
                    )

    if np.all(assigned):
        return _finish()

    # --------------------------------------------------
    # Step 4: EMS boundary fallback for remaining spacecraft
    # --------------------------------------------------
    candidate_u_list = []
    N = int(max(1, n_boundary_candidates))

    for i in range(M):
        if assigned[i]:
            candidate_u_list.append([u_init[i]])
            continue

        p_i = p_agents[i]
        u_curr = u_curr_norm[i]

        u_i_star = _unit(p_hat - p_i)

        if u_i_star is None:
            u_i_star = u_curr

        # If no EMS is active, keep mean/current fallback.
        if p_em is None or float(R_em) <= 0.0:
            candidate_u_list.append([u_i_star])
            candidate_log.append({
                "stage": "fallback",
                "sc_idx": int(i),
                "p_target": p_hat.copy(),
                "u": u_i_star.copy(),
                "accepted": False,
                "reason": "no_ems_active",
            })
            continue

        r_vec = p_em - p_i
        r_norm = np.linalg.norm(r_vec)

        if r_norm < R_em + eps:
            candidate_u_list.append([u_i_star])
            candidate_log.append({
                "stage": "fallback",
                "sc_idx": int(i),
                "p_target": p_hat.copy(),
                "u": u_i_star.copy(),
                "accepted": False,
                "reason": "inside_ems_fallback_to_mean",
            })
            continue

        v_em = r_vec / r_norm
        alpha_em = np.arcsin(np.clip(R_em / r_norm, -1.0, 1.0))
        gamma_bound = float(theta_h + alpha_em + alpha_s)

        if gamma_bound >= np.pi - 1e-6:
            candidate_u_list.append([u_i_star])
            candidate_log.append({
                "stage": "fallback",
                "sc_idx": int(i),
                "p_target": p_hat.copy(),
                "u": u_i_star.copy(),
                "accepted": False,
                "reason": "gamma_bound_too_large_fallback_to_mean",
            })
            continue

        if keepout_safe_single_local(p_i, u_i_star):
            candidate_u_list.append([u_i_star])
            candidate_log.append({
                "stage": "fallback",
                "sc_idx": int(i),
                "p_target": p_hat.copy(),
                "u": u_i_star.copy(),
                "accepted": False,
                "reason": "mean_safe_in_fallback",
            })
            continue

        cos_gb = float(np.cos(gamma_bound))
        sin_gb = float(np.sin(gamma_bound))

        e1_em, e2_em = orthonormal_basis_from_u(v_em)

        cproj = float(np.dot(u_i_star, v_em))
        v_perp = u_i_star - cproj * v_em
        n_perp = float(np.linalg.norm(v_perp))

        if n_perp < eps:
            psi0 = 0.0
        else:
            v_perp_hat = v_perp / n_perp
            psi0 = float(
                np.arctan2(
                    np.dot(v_perp_hat, e2_em),
                    np.dot(v_perp_hat, e1_em),
                )
            )

        all_dirs = []

        for k in range(N):
            psi = psi0 + 2.0 * np.pi * (k / N)

            u_b = (
                cos_gb * v_em
                + sin_gb * (np.cos(psi) * e1_em + np.sin(psi) * e2_em)
            )
            u_b = u_b / max(np.linalg.norm(u_b), eps)

            theta_b, phi_b = u_to_theta_phi(
                u_b,
                u_curr_norm[i],
                e1_list[i],
                e2_list[i],
            )

            if theta_b is None:
                candidate_log.append({
                    "stage": "ems",
                    "sc_idx": int(i),
                    "p_target": p_i + u_b,
                    "u": u_b.copy(),
                    "accepted": False,
                    "reason": "theta_phi_failed",
                })
                continue

            if theta_b < theta_lower[i] - 1e-6 or theta_b > theta_upper[i] + 1e-6:
                candidate_log.append({
                    "stage": "ems",
                    "sc_idx": int(i),
                    "p_target": p_i + u_b,
                    "u": u_b.copy(),
                    "theta": float(theta_b),
                    "phi": float(phi_b) % (2 * np.pi),
                    "accepted": False,
                    "reason": "slew_infeasible",
                })
                continue

            all_dirs.append(u_b)

            candidate_log.append({
                "stage": "ems",
                "sc_idx": int(i),
                "p_target": p_i + u_b,
                "u": u_b.copy(),
                "theta": float(theta_b),
                "phi": float(phi_b) % (2 * np.pi),
                "accepted": False,
                "reason": "candidate",
            })

        if len(all_dirs) == 0:
            candidate_u_list.append([u_i_star])
            candidate_log.append({
                "stage": "fallback",
                "sc_idx": int(i),
                "p_target": p_hat.copy(),
                "u": u_i_star.copy(),
                "accepted": False,
                "reason": "no_ems_dirs_fallback_to_mean",
            })
            continue

        hit_dirs = []

        for u_b in all_dirs:
            if ray_intersects_ellipsoid(p_i, u_b, p_hat, P_inv, d_M):
                hit_dirs.append(u_b)

        candidate_u_list.append(hit_dirs if len(hit_dirs) > 0 else all_dirs)

    # --------------------------------------------------
    # Evaluate remaining combinations with J_t
    # --------------------------------------------------
    best_J = -np.inf
    best_u = None

    index_ranges = [range(len(candidate_u_list[i])) for i in range(M)]

    for choice in itertools.product(*index_ranges):
        u_trial = np.zeros_like(u_curr_agents, dtype=float)

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
            lambda_k1=lambda_k1,
            n_mc=n_mc,
            y_samples_cached=y_cached,
            coverage_mode=coverage_mode,
        )

        if J_val > best_J:
            best_J = float(J_val)
            best_u = u_trial.copy()

    if best_u is None:
        best_u = np.zeros_like(u_curr_agents)

        for i in range(M):
            if assigned[i]:
                best_u[i] = u_init[i]
            else:
                ok, u_i, _, _, _ = try_point_to_point(
                    i,
                    p_hat,
                    stage="fallback",
                    log_candidate=True,
                )

                best_u[i] = u_i if ok else u_curr_norm[i]

    # --------------------------------------------------
    # Finalize unassigned spacecraft from best_u
    # --------------------------------------------------
    for i in range(M):
        if assigned[i]:
            continue

        theta_i, phi_i = u_to_theta_phi(
            best_u[i],
            u_curr_norm[i],
            e1_list[i],
            e2_list[i],
        )

        if theta_i is None:
            theta_i = 0.0
            phi_i = 0.0

        theta_i = float(np.clip(theta_i, theta_lower[i], theta_upper[i]))
        phi_i = float(phi_i) % (2 * np.pi)

        u_i = theta_phi_to_u(
            theta_i,
            phi_i,
            u_curr_norm[i],
            e1_list[i],
            e2_list[i],
        )

        u_i, theta_i, phi_i = apply_jitter_once_if_valid(
            i,
            theta_i,
            phi_i,
            u_i,
            require_ellipsoid_hit=False,
        )

        best_u[i] = u_i.copy()
        u_init[i] = u_i.copy()
        theta_init[i] = float(theta_i)
        phi_init[i] = float(phi_i) % (2 * np.pi)
        assigned[i] = True

        append_final_accepted_row(
            i,
            u_i,
            theta_i,
            phi_i,
            stage="final_combo",
            reason="accepted_from_best_u",
        )

    # --------------------------------------------------
    # Convert final initialized directions to x0
    # No jitter is applied here.
    # --------------------------------------------------
    for i in range(M):
        theta_i, phi_i = u_to_theta_phi(
            u_init[i],
            u_curr_norm[i],
            e1_list[i],
            e2_list[i],
        )

        if theta_i is None:
            theta_i = 0.0
            phi_i = 0.0

        theta_i = float(np.clip(theta_i, theta_lower[i], theta_upper[i]))
        phi_i = float(phi_i) % (2 * np.pi)

        x0[2 * i] = theta_i
        x0[2 * i + 1] = phi_i

        theta_init[i] = theta_i
        phi_init[i] = phi_i

        acc_idx = accepted_log_idx_by_sc[i]
        if acc_idx is not None and 0 <= acc_idx < len(candidate_log):
            candidate_log[acc_idx]["u"] = u_init[i].copy()
            candidate_log[acc_idx]["theta"] = float(theta_i)
            candidate_log[acc_idx]["phi"] = float(phi_i)
            candidate_log[acc_idx]["accepted"] = True

    if return_log:
        ensure_exactly_one_accepted_per_spacecraft()
        return x0, candidate_log

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
        theta_h, theta_s_list, jitter,
        d_M=3.0, kappa_sigma=120.0, lambda_k1=0.5,
        n_mc=25000, seed=0,
        n_restarts=3,
        p_em=None, R_em=0.0,
        alpha_s=0.0, lambda_em=0.0,
        beta_zeta=50.0,
        nshell=400,
        num_candidates=2,
        restart_noise_scale=0.05,
        maxiter=60,
        ftol=1e-10,
        display=False,
        detecting_idx=None,
        detecting_u=None,
        *,
        use_fixed_agent: bool = False,
        idx_fix: Optional[int] = None,
        u_fix: Optional[np.ndarray] = None,

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

    def estimate_ellipsoid_ems_visibility(
            p_hat,
            P_p,
            p_agents,
            d_M,
            p_em,
            R_em,
            theta_h,
            alpha_s,
            n_shell=1000,
            seed=0,
            eps=1e-12,
    ):
        """
        Returns visible_ems: bool array of shape (M,).

        visible_ems[i] is True if spacecraft i has at least one direction toward
        the sampled ellipsoid shell that satisfies EMS keepout.
        """

        p_hat = np.asarray(p_hat, dtype=float).reshape(3, )
        P_p = np.asarray(P_p, dtype=float).reshape(3, 3)
        p_agents = np.asarray(p_agents, dtype=float)

        M = p_agents.shape[0]

        if p_em is None or float(R_em) <= 0.0:
            return np.ones(M, dtype=bool)

        p_em = np.asarray(p_em, dtype=float).reshape(3, )

        rng = np.random.default_rng(seed)

        try:
            Lp = np.linalg.cholesky(P_p)
        except np.linalg.LinAlgError:
            Lp = np.linalg.cholesky(P_p + 1e-12 * np.eye(3))

        dirs = rng.normal(size=(n_shell, 3))
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), eps)

        p_shell = p_hat[None, :] + (Lp @ (d_M * dirs).T).T

        visible_ems = np.zeros(M, dtype=bool)

        for i in range(M):
            p_i = p_agents[i]

            r_em = p_em - p_i
            r_em_norm = np.linalg.norm(r_em)

            if r_em_norm < R_em + eps:
                visible_ems[i] = False
                continue

            v_em = r_em / r_em_norm
            alpha_em = np.arcsin(np.clip(R_em / r_em_norm, -1.0, 1.0))
            gamma_min = theta_h + alpha_em + alpha_s

            r = p_shell - p_i[None, :]
            r_norm = np.linalg.norm(r, axis=1)
            u = r / np.maximum(r_norm[:, None], eps)

            gamma = np.arccos(np.clip(u @ v_em, -1.0, 1.0))

            visible_ems[i] = np.any(gamma >= gamma_min)

        return visible_ems

    # --- θ-bounds from ellipsoid + slew ---
    theta_min_ell, theta_max_ell = estimate_theta_bounds_from_ellipsoid(
        p_hat,
        P_p,
        p_agents,
        u_curr_agents,
        d_M,
        n_shell=nshell,
        seed=seed + 999,
    )

    theta_pad = np.deg2rad(0.05)

    theta_lower_ell = np.maximum(theta_min_ell - theta_pad, 0.0)
    theta_upper_ell = theta_max_ell + theta_pad

    ellipsoid_reachable_by_slew = theta_lower_ell <= theta_s_list

    ellipsoid_visible_ems = estimate_ellipsoid_ems_visibility(
        p_hat,
        P_p,
        p_agents,
        d_M,
        p_em,
        R_em,
        theta_h,
        alpha_s,
        n_shell=nshell,
        seed=seed + 1999,
    )

    use_ell_bounds = ellipsoid_reachable_by_slew & ellipsoid_visible_ems

    theta_lower = np.where(
        use_ell_bounds,
        theta_lower_ell,
        0.0,
    )

    theta_upper = np.where(
        use_ell_bounds,
        np.minimum(theta_upper_ell, theta_s_list),
        theta_s_list,
    )

    infeasible_mask = theta_upper < theta_lower

    if np.any(infeasible_mask):
        return None, None, 0.0, [], np.inf

    # MC cache
    y_cached = make_cached_y(P_p, d_M, n_mc, seed=seed+123)

    # --- objective-mode pre-check ---
    # If two or more spacecraft can point at the mean and place the sampled
    # uncertainty volume inside their FOV, switch to an "at least two coverage"
    # objective. Otherwise retain the original exact-two + one-coverage fallback.
    mean_full_cover_flags, u_mean_check, theta_req_mean_check = mean_pointing_full_uncertainty_cover_flags(
        p_hat, P_p, p_agents, u_curr_agents,
        theta_h, theta_s_list, d_M, y_cached,
        p_em=p_em, R_em=R_em, alpha_s=alpha_s,
    )
    n_mean_full_cover = int(np.count_nonzero(mean_full_cover_flags))
    coverage_mode = "atleast2" if n_mean_full_cover >= 2 else "exact2_plus_k1"

    # --- choose optimization mode (original vs fixed-agent) ---
    use_fixed = bool(use_fixed_agent) and (idx_fix is not None)

    if not use_fixed:
        M_free = M
        free_idx = None
    else:
        if u_fix is None:
            raise ValueError("u_fix must be provided when use_fixed_agent=True and idx_fix is set")
        free_idx = build_free_indices(M, int(idx_fix))
        if free_idx.size != M - 1:
            raise ValueError("free_idx must have size M-1 for fixed-agent mode")
        M_free = int(free_idx.size)

    # Bounds in (θ, φ)
    bounds = []
    if not use_fixed:
        for i in range(M):
            bounds.append((theta_lower[i], theta_upper[i]))  # theta_i
            bounds.append((0.0, 2*np.pi - 1e-12))                    # phi_i
    else:
        for i in free_idx:
            bounds.append((theta_lower[int(i)], theta_upper[int(i)]))  # theta_i
            bounds.append((0.0, 2*np.pi))                              # phi_i

    # ---- History logger ----
    history = []

    def _x_free_to_x_full(x_free: np.ndarray) -> np.ndarray:
        """Convert x_free (2*(M-1),) to x_full (2*M,) with NaNs for fixed agent."""
        x_full = np.full(2 * M, np.nan, dtype=float)
        for k, i in enumerate(free_idx):
            i = int(i)
            x_full[2 * i] = float(x_free[2 * k])
            x_full[2 * i + 1] = float(x_free[2 * k + 1])
        return x_full

    def _u_from_decision(x_decision: np.ndarray) -> np.ndarray:
        if not use_fixed:
            return angles_to_pointings(x_decision, p_agents, u_curr_agents, theta_lower, theta_upper, M)
        return angles_to_pointings_with_fixed(
            x_decision, p_agents, u_curr_agents, theta_lower, theta_upper,
            free_idx=free_idx, idx_fix=int(idx_fix), u_fix=u_fix
        )

    def log_state(x_decision, restart_idx):
        """
        Record:
          - u from (θ, φ)
          - slew between u_curr and u
          - J_t at this point (no EMS penalty)
        """
        u = _u_from_decision(x_decision)

        dots = np.einsum("ij,ij->i", u_curr_agents, u)
        dots = np.clip(dots, -1.0, 1.0)
        slews = np.arccos(dots)

        J_val = J_t_dual_coverage(
            p_hat, P_p, p_agents, u,
            theta_h, d_M=d_M, kappa_sigma=kappa_sigma, lambda_k1=lambda_k1,
            n_mc=y_cached.shape[0], y_samples_cached=y_cached,
            coverage_mode=coverage_mode
        )

        history.append({
            "x": (x_decision.copy() if not use_fixed else _x_free_to_x_full(np.asarray(x_decision, dtype=float).ravel())),
            "x_free": (None if not use_fixed else np.asarray(x_decision, dtype=float).ravel().copy()),
            "u": u.copy(),
            "slew": slews.copy(),
            "J": J_val,
            "restart": restart_idx,
            "use_fixed_agent": bool(use_fixed),
            "idx_fix": (None if not use_fixed else int(idx_fix)),
            "coverage_mode": coverage_mode,
            "n_mean_full_cover": n_mean_full_cover,
            "mean_full_cover_flags": mean_full_cover_flags.copy(),
            "theta_req_mean_check": theta_req_mean_check.copy(),
        })

    best_x = None
    best_f = np.inf   # objective = cost = -J_t + penalty
    best_cost = np.inf

    # Initialize a sensible start (full), then slice if using fixed agent
    x0_full, candidate_log = init_theta_phi_boundary_projection(
        p_hat, P_p, p_agents, u_curr_agents,
        np.zeros_like(theta_s_list), theta_s_list, theta_h,
        d_M, kappa_sigma, lambda_k1, y_cached, p_em,
        R_em, alpha_s, seed, jitter, n_boundary_candidates=8,
        coverage_mode=coverage_mode, detecting_indices=detecting_idx, detecting_us=detecting_u, detecting_mode='current',
        n_uncertainty_candidates=num_candidates, n_los_candidates=num_candidates, return_log=True
    )

    debug_ini_flag = False
    if debug_ini_flag:
        from collections import Counter
        print("after uncertainty stage counts:", Counter(row["stage"] for row in candidate_log))
        print("after uncertainty reason counts:", Counter(row["reason"] for row in candidate_log))

        candidate_accepted_log = [
            row for row in candidate_log
            if row.get("accepted", False)
        ]

        def compare_candidate_log_to_x0(
                candidate_log,
                x0_full,
                p_agents,
                u_curr_agents,
                *,
                accepted_only=True,
                eps=1e-12,
        ):
            """
            Compare candidate_log['u'] against the direction reconstructed from x0_full.

            x0_full is assumed to contain:
                [theta_0, phi_0, theta_1, phi_1, ..., theta_M-1, phi_M-1]

            where theta/phi are local angles relative to u_curr_agents[i].
            """

            x0_full = np.asarray(x0_full, dtype=float)
            p_agents = np.asarray(p_agents, dtype=float)
            u_curr_agents = np.asarray(u_curr_agents, dtype=float)

            M = p_agents.shape[0]

            def unit(v):
                v = np.asarray(v, dtype=float).reshape(3, )
                n = float(np.linalg.norm(v))
                if n < eps:
                    return None
                return v / n

            def basis_from_u(u):
                u = unit(u)
                if u is None:
                    u = np.array([0.0, 0.0, 1.0], dtype=float)

                a = np.array([1.0, 0.0, 0.0], dtype=float)
                if abs(np.dot(a, u)) > 0.9:
                    a = np.array([0.0, 1.0, 0.0], dtype=float)

                e1 = a - np.dot(a, u) * u
                e1 = e1 / np.linalg.norm(e1)

                e2 = np.cross(u, e1)
                e2 = e2 / np.linalg.norm(e2)

                return e1, e2

            def theta_phi_to_u(theta, phi, u_curr):
                u_curr = unit(u_curr)
                if u_curr is None:
                    u_curr = np.array([0.0, 0.0, 1.0], dtype=float)

                e1, e2 = basis_from_u(u_curr)

                u = (
                        np.cos(theta) * u_curr
                        + np.sin(theta) * (
                                np.cos(phi) * e1
                                + np.sin(phi) * e2
                        )
                )

                return u / max(np.linalg.norm(u), eps)

            # reconstruct all directions from x0_full
            u_from_x0 = np.zeros((M, 3), dtype=float)

            for i in range(M):
                theta_i = float(x0_full[2 * i])
                phi_i = float(x0_full[2 * i + 1]) % (2.0 * np.pi)

                u_from_x0[i] = theta_phi_to_u(
                    theta_i,
                    phi_i,
                    u_curr_agents[i],
                )

            rows = []

            for row in candidate_log:
                if accepted_only and not row.get("accepted", False):
                    continue

                i = int(row["sc_idx"])

                u_log = unit(row["u"])
                if u_log is None:
                    continue

                u_x0 = u_from_x0[i]

                dot_val = float(np.clip(np.dot(u_log, u_x0), -1.0, 1.0))
                angle_rad = float(np.arccos(dot_val))
                angle_deg = float(np.rad2deg(angle_rad))

                rows.append({
                    "sc_idx": i,
                    "stage": row.get("stage", None),
                    "reason": row.get("reason", None),
                    "accepted": row.get("accepted", False),
                    "dot": dot_val,
                    "angle_rad": angle_rad,
                    "angle_deg": angle_deg,
                    "u_log": u_log,
                    "u_x0": u_x0,
                    "theta_x0": float(x0_full[2 * i]),
                    "phi_x0": float(x0_full[2 * i + 1]) % (2.0 * np.pi),
                })

            return rows, u_from_x0

        rows, u_from_x0 = compare_candidate_log_to_x0(
            candidate_accepted_log,
            x0_full,
            p_agents,
            u_curr_agents,
            accepted_only=True,
        )

        for r in rows:
            print(
                f"SC{r['sc_idx']} | "
                f"stage={r['stage']} | "
                # f"dot={r['dot']:.12f} | "
                # f"angle={r['angle_deg']:.6e} deg"
            )

        util.plot_init_candidate_geometry_light(
            p_hat,
            P_p,
            p_agents,
            theta_h,
            d_M,
            p_em=p_em,
            R_em=R_em,
            detecting_idx=idx_fix,
            detecting_u=u_fix,
            candidate_log=candidate_accepted_log,  # or candidate_log if you want all
            show=False,
        )

    if not use_fixed:
        x0_mean = x0_full.copy()
    else:
        x0_mean = np.zeros(2 * M_free, dtype=float)
        for k, i in enumerate(free_idx):
            i = int(i)
            x0_mean[2 * k] = x0_full[2 * i]
            x0_mean[2 * k + 1] = x0_full[2 * i + 1]

    for r in range(n_restarts):
        if r == 0:
            x0 = x0_mean.copy()
        else:
            noise = rng.normal(scale=restart_noise_scale, size=x0_mean.size)
            x0 = x0_mean + noise

            if not use_fixed:
                for i in range(M):
                    x0[2 * i] = np.clip(x0[2 * i], theta_lower[i], theta_upper[i])
                    x0[2 * i + 1] = x0[2 * i + 1] % (2 * np.pi)
            else:
                for k, i in enumerate(free_idx):
                    i = int(i)
                    x0[2 * k] = np.clip(x0[2 * k], theta_lower[i], theta_upper[i])
                    x0[2 * k + 1] = x0[2 * k + 1] % (2 * np.pi)

        log_state(x0, r)

        if not use_fixed:
            f = lambda z: objective_joint(
                z, p_hat, P_p, p_agents, u_curr_agents,
                theta_lower, theta_upper,
                theta_h, d_M, kappa_sigma, lambda_k1, y_cached,
                p_em=p_em, R_em=R_em,
                alpha_s=alpha_s, lambda_em=lambda_em,
                beta_zeta=beta_zeta,
                coverage_mode=coverage_mode
            )
        else:
            f = lambda z: objective_joint_free(
                z, p_hat, P_p, p_agents, u_curr_agents,
                theta_lower, theta_upper,
                theta_h, d_M, kappa_sigma, lambda_k1, y_cached,
                free_idx=free_idx, idx_fix=int(idx_fix), u_fix=u_fix,
                p_em=p_em, R_em=R_em,
                alpha_s=alpha_s, lambda_em=lambda_em,
                beta_zeta=beta_zeta,
                coverage_mode=coverage_mode
            )

        def cb(xk, restart_idx=r):
            log_state(xk, restart_idx)

        res = minimize(
            f, x0, method="L-BFGS-B",
            bounds=bounds,
            callback=cb,
            options=dict(maxiter=maxiter, ftol=ftol, disp=display)
        )
        x_star = res.x
        f_star = res.fun

        if f_star < best_f:
            best_f = f_star
            best_x = np.asarray(x_star, dtype=float).copy()
            best_cost = float(f_star)
    # Unpack best and compute J_t
    if best_x is None:
        return None, None, 0.0, history, np.inf

    if not use_fixed:
        thetas_best, phis_best = unpack_angles(best_x, M)
        u_best = angles_to_pointings(best_x, p_agents, u_curr_agents, theta_lower, theta_upper, M)
        angles_best = [(thetas_best[i], phis_best[i]) for i in range(M)]
    else:
        u_best = angles_to_pointings_with_fixed(
            best_x, p_agents, u_curr_agents, theta_lower, theta_upper,
            free_idx=free_idx, idx_fix=int(idx_fix), u_fix=u_fix
        )

        thetas_free, phis_free = unpack_angles_free(best_x, int(free_idx.size))
        angles_best = [(np.nan, np.nan) for _ in range(M)]
        for k, i in enumerate(free_idx):
            angles_best[int(i)] = (float(thetas_free[k]), float(phis_free[k]))
        angles_best[int(idx_fix)] = (np.nan, np.nan)

    J_best = J_t_dual_coverage(
        p_hat, P_p, p_agents, u_best,
        theta_h, d_M=d_M, kappa_sigma=kappa_sigma, lambda_k1=lambda_k1,
        n_mc=y_cached.shape[0], y_samples_cached=y_cached,
        coverage_mode=coverage_mode
    )

    return u_best, angles_best, float(J_best), history, float(best_cost)


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
        self.lambda_k1 = float(opt.get("lambda_k1", 1.0))
        self.n_mc = int(opt.get("n_mc", 200))
        self.n_restarts = int(opt.get("n_restarts", 1))
        self.jitter = opt.get("jitter", [1.0, -1.0, 0.0])
        self.restart_noise = opt.get("restart_noise", 0.05)
        self.nshell=int(opt.get("nshell", 400))
        self.max_iterations=int(opt.get("max_iterations", 100))
        self.ftolerance=float(opt.get("ftol", 1e-10))
        self.display=bool(opt.get("display", False))
        self.num_candidates=int(opt.get("num_candidates", 8))

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
        detecting_u: np.ndarray,
        detecting_idx: list,
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
        # fixed-agent optimizer mode (optional)
        use_fixed_agent: bool = False,
        fixed_agent_idx: Optional[int] = None,
        fixed_agent_u: Optional[np.ndarray] = None,
        # If "mean_los_per_epoch", recompute the fixed boresight for each
        # candidate epoch as LOS from the fixed spacecraft to p_hat_k.
        # Otherwise, use fixed_agent_u as provided.
        fixed_agent_u_mode: str = "provided",


        # coverage counting
        coverage_point: Optional[np.ndarray] = None,  # (3,) or (N,3)
    ) -> Tuple[AttCoordResult, AttCoordResult, List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Returns:
            (res_opt_best, res_mean_best, opt_series, mean_series)

        - res_opt_best: best feasible optimizer solution across dt_grid (min cost)
        - res_mean_best: earliest dt where all agents can point to mean (slew+keepout)
                         If infeasible across horizon -> coverage=-1, others NaN/u=None

        - opt_series: list length N; per-epoch optimizer diagnostics
        - mean_series: list length N; per-epoch mean-method diagnostics
        """

        # --- normalize / validate ---
        dt_grid = np.asarray(dt_grid, dtype=float).ravel()
        N = int(dt_grid.size)
        if N == 0:
            return self._nan_result(), self._nan_result(), [], []

        u_curr_agents = np.asarray(u_curr_agents, dtype=float)
        if u_curr_agents.ndim != 2 or u_curr_agents.shape[1] != 3:
            raise ValueError(f"u_curr_agents must be (M,3), got {u_curr_agents.shape}")
        M = int(u_curr_agents.shape[0])
        if M <= 0:
            return self._nan_result(), self._nan_result(), [], []

        # If using fixed-agent mode and no fixed vector provided, default to the agent's current boresight.
        fixed_agent_u_mode = str(fixed_agent_u_mode or "provided").lower().strip()
        if fixed_agent_u_mode not in ("provided", "mean_los_per_epoch"):
            raise ValueError(
                "fixed_agent_u_mode must be one of 'provided' or 'mean_los_per_epoch'"
            )

        tracking_cfg = self.cfg.get("attcoord_tracking", {}) if isinstance(self.cfg, dict) else {}
        fixed_agent_ems_infeasible_behavior = str(
            tracking_cfg.get("anchor_ems_infeasible_fallback", "free")
        ).lower().strip()
        if fixed_agent_ems_infeasible_behavior not in ("free", "skip"):
            raise ValueError(
                "attcoord_tracking.anchor_ems_infeasible_fallback must be 'free' or 'skip'"
            )

        if use_fixed_agent:
            if fixed_agent_idx is None:
                raise ValueError("use_fixed_agent=True requires fixed_agent_idx")
            if not (0 <= int(fixed_agent_idx) < M):
                raise ValueError(f"fixed_agent_idx out of range: {fixed_agent_idx} for M={M}")
            if fixed_agent_u is None:
                fixed_agent_u = u_curr_agents[int(fixed_agent_idx)].copy()

        if d_M is None:
            d_M = float(self.cfg.get("covariance", {}).get("d_mahal", 1.0))

        # EMS defaults
        if p_em is None:
            p_em = self.p_em_default
        else:
            p_em = np.asarray(p_em, dtype=float).reshape(3,)

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
        # (A) OPTIMIZER per-epoch diagnostics + best selection
        # ============================================================
        opt_series: List[Dict[str, Any]] = []
        best_opt = {"cost": np.inf}

        for k, dt in enumerate(dt_grid):
            dt = float(dt)
            p_agents_k = p_agents_ts[k]
            p_hat_k = p_hat_ts[k].reshape(3,)
            P_p_k = P_p_ts[k]

            # Slew limit for this dt
            theta_s_t = float(theta_s_of_dt(dt, alpha_max, omega_max))
            theta_s_t = float(np.clip(theta_s_t, 0.0, np.deg2rad(179.0)))
            theta_s_list_t = np.full(M, theta_s_t)

            # Fixed tracking-anchor handling.  In mean_los_per_epoch mode,
            # recompute the fixed anchor boresight for this candidate epoch
            # using the predicted target mean and the candidate spacecraft
            # position.  The candidate is skipped if this anchor direction
            # violates the available slew or EMS keepout constraints.
            fixed_agent_u_k = fixed_agent_u
            fixed_agent_feasible_k = True
            fixed_agent_infeasible_reason = ""
            fixed_agent_theta_req_rad = float("nan")
            use_fixed_agent_k = bool(use_fixed_agent)
            fixed_agent_idx_k = fixed_agent_idx
            fixed_agent_mode_k = fixed_agent_u_mode

            if bool(use_fixed_agent_k) and fixed_agent_idx_k is not None:
                idx_fix_i = int(fixed_agent_idx_k)

                if fixed_agent_u_mode == "mean_los_per_epoch":
                    r_fix = np.asarray(p_hat_k, dtype=float).reshape(3,) - np.asarray(p_agents_k[idx_fix_i], dtype=float).reshape(3,)
                    nr_fix = float(np.linalg.norm(r_fix))
                    if nr_fix <= 1e-12:
                        fixed_agent_u_k = u_curr_agents[idx_fix_i].copy()
                        fixed_agent_feasible_k = False
                        fixed_agent_infeasible_reason = "zero_anchor_mean_los"
                    else:
                        fixed_agent_u_k = r_fix / nr_fix

                fixed_agent_u_k = np.asarray(fixed_agent_u_k, dtype=float).reshape(3,)
                fixed_agent_u_k = fixed_agent_u_k / max(float(np.linalg.norm(fixed_agent_u_k)), 1e-12)

                fixed_agent_theta_req_rad = float(angle_between(u_curr_agents[idx_fix_i], fixed_agent_u_k))
                if fixed_agent_theta_req_rad > theta_s_t + 1e-12:
                    fixed_agent_feasible_k = False
                    fixed_agent_infeasible_reason = "fixed_anchor_slew_infeasible"

                if fixed_agent_feasible_k and p_em is not None and float(R_em) > 0.0:
                    if not keepout_safe_single(
                        p_agents_k[idx_fix_i], fixed_agent_u_k, float(theta_h),
                        p_em, float(R_em), float(alpha_s)
                    ):
                        fixed_agent_feasible_k = False
                        fixed_agent_infeasible_reason = "fixed_anchor_ems_infeasible"

                if (not fixed_agent_feasible_k) and fixed_agent_infeasible_reason == "fixed_anchor_ems_infeasible" and fixed_agent_ems_infeasible_behavior == "free":
                    # The tracking anchor's mean-LOS violates the conservative EMS/FOV-cone
                    # keepout for this candidate epoch.  Do not kill the candidate.
                    # Instead, revert to the pre-anchor/free attitude-coordination behaviour
                    # for this epoch.  This keeps EMS infeasibility from making the entire
                    # search horizon NaN while still logging why the anchor was released.
                    use_fixed_agent_k = False
                    fixed_agent_idx_k = None
                    fixed_agent_u_k = None
                    fixed_agent_mode_k = "free_due_to_anchor_ems_infeasible"
                    fixed_agent_feasible_k = True

                # print(fixed_agent_mode_k)

                if not fixed_agent_feasible_k:
                    opt_series.append(dict(
                        k=k, dt=dt, feasible=False,
                        u=None,
                        cost=float("nan"),
                        J=float("nan"),
                        theta_req_avg_deg=float("nan"),
                        theta_s_allowed=float(theta_s_t),
                        coverage=-1,
                        history=None,
                        fixed_agent_idx=idx_fix_i,
                        fixed_agent_u_mode=fixed_agent_mode_k,
                        fixed_agent_feasible=False,
                        fixed_agent_infeasible_reason=fixed_agent_infeasible_reason,
                        fixed_agent_theta_req_deg=float(np.rad2deg(fixed_agent_theta_req_rad)) if np.isfinite(fixed_agent_theta_req_rad) else float("nan"),
                    ))
                    continue

            u_star, ang_star, J_star, history, cost_star = optimize_pointing_lbfgs_joint(
                p_hat_k, P_p_k, p_agents_k, u_curr_agents,
                float(theta_h), theta_s_list_t, self.jitter,
                d_M=float(d_M),
                kappa_sigma=float(self.kappa_sigma),
                lambda_k1=float(self.lambda_k1),
                n_mc=int(self.n_mc),
                seed=int(trial_seed),
                n_restarts=int(self.n_restarts),
                p_em=p_em, R_em=float(R_em),
                alpha_s=float(alpha_s),
                lambda_em=float(lambda_em),
                beta_zeta=float(beta_zeta),
                restart_noise_scale=float(self.restart_noise),
                nshell=int(self.nshell),
                maxiter=int(self.max_iterations),
                ftol=float(self.ftolerance),
                display=bool(self.display),
                use_fixed_agent=bool(use_fixed_agent_k),
                idx_fix=fixed_agent_idx_k,
                u_fix=fixed_agent_u_k,
                num_candidates=int(self.num_candidates),
                detecting_u=detecting_u,
                detecting_idx=detecting_idx
            )

            if u_star is None or not np.isfinite(float(cost_star)):
                # infeasible / failed
                opt_series.append(dict(
                    k=k, dt=dt, feasible=False,
                    u=None,
                    cost=float("nan"),
                    J=float("nan"),
                    theta_req_avg_deg=float("nan"),
                    theta_s_allowed=float(theta_s_t),
                    coverage=-1,
                    history=history,
                    fixed_agent_idx=(None if fixed_agent_idx_k is None else int(fixed_agent_idx_k)),
                    fixed_agent_u_mode=fixed_agent_mode_k,
                    fixed_agent_feasible=bool(fixed_agent_feasible_k),
                    fixed_agent_infeasible_reason=fixed_agent_infeasible_reason,
                    fixed_agent_theta_req_deg=float(np.rad2deg(fixed_agent_theta_req_rad)) if np.isfinite(fixed_agent_theta_req_rad) else float("nan"),
                ))
                continue

            # average slew
            slews = np.array([angle_between(u_curr_agents[i], u_star[i]) for i in range(M)], dtype=float)
            slew_avg_deg = float(np.rad2deg(np.nanmean(slews)))

            cov_cnt = -1
            if covpt_ts is not None:
                cov_cnt = int(coverage_count_point(covpt_ts[k], p_agents_k, u_star, float(theta_h)))

            c = float(cost_star)
            Jv = float(J_star)
            coverage_mode = None
            n_mean_full_cover = -1
            mean_full_cover_flags = None
            if history:
                coverage_mode = history[0].get("coverage_mode")
                n_mean_full_cover = int(history[0].get("n_mean_full_cover", -1))
                mean_full_cover_flags = history[0].get("mean_full_cover_flags")

            row = dict(
                k=k, dt=dt, feasible=True,
                u=u_star,
                cost=c,
                J=Jv,
                theta_req_avg_deg=slew_avg_deg,
                theta_s_allowed=float(theta_s_t),
                coverage=int(cov_cnt),
                coverage_mode=coverage_mode,
                n_mean_full_cover=n_mean_full_cover,
                mean_full_cover_flags=mean_full_cover_flags,
                history=history,
                fixed_agent_idx=(None if fixed_agent_idx_k is None else int(fixed_agent_idx_k)),
                fixed_agent_u_mode=fixed_agent_mode_k,
                fixed_agent_feasible=bool(fixed_agent_feasible_k),
                fixed_agent_infeasible_reason=fixed_agent_infeasible_reason,
                fixed_agent_theta_req_deg=float(np.rad2deg(fixed_agent_theta_req_rad)) if np.isfinite(fixed_agent_theta_req_rad) else float("nan"),
            )
            opt_series.append(row)

            # update best
            if c < best_opt["cost"]:
                best_opt = dict(
                    cost=c, dt=dt, u=u_star, J=Jv,
                    theta_req_avg_deg=slew_avg_deg,
                    coverage=int(cov_cnt),
                    coverage_mode=coverage_mode,
                    n_mean_full_cover=n_mean_full_cover,
                    mean_full_cover_flags=mean_full_cover_flags,
                    history=history,
                    fixed_agent_idx=(None if fixed_agent_idx_k is None else int(fixed_agent_idx_k)),
                    fixed_agent_u_mode=fixed_agent_mode_k,
                    fixed_agent_theta_req_deg=float(np.rad2deg(fixed_agent_theta_req_rad)) if np.isfinite(fixed_agent_theta_req_rad) else float("nan"),
                )

        if not np.isfinite(best_opt.get("cost", np.inf)):
            res_opt_best = self._nan_result()
        else:
            res_opt_best = AttCoordResult(
                u_cmd=best_opt["u"],
                chosen_dt=float(best_opt["dt"]),
                cost=float(best_opt["cost"]),
                J=float(best_opt["J"]),
                coverage=int(best_opt["coverage"]),
                theta_req_avg_deg=float(best_opt["theta_req_avg_deg"]),
                extra={
                    "history": best_opt["history"],
                    "coverage_mode": best_opt.get("coverage_mode"),
                    "n_mean_full_cover": best_opt.get("n_mean_full_cover", -1),
                    "mean_full_cover_flags": best_opt.get("mean_full_cover_flags"),
                    "fixed_agent_idx": best_opt.get("fixed_agent_idx"),
                    "fixed_agent_u_mode": best_opt.get("fixed_agent_u_mode"),
                    "fixed_agent_theta_req_deg": best_opt.get("fixed_agent_theta_req_deg"),
                },
            )

        # ============================================================
        # (B) MEAN per-epoch diagnostics + earliest-feasible selection
        # ============================================================
        start_mean = time.time()

        mean_series: List[Dict[str, Any]] = []
        best_mean = None

        for k, dt in enumerate(dt_grid):
            dt = float(dt)
            p_agents_k = p_agents_ts[k]
            p_hat_k = p_hat_ts[k].reshape(3,)

            all_ok, per_ok, u_mean, theta_req, theta_s_t = all_agents_can_point_to_mean(
                dt,
                p_hat_k,
                p_agents_k, u_curr_agents,
                float(theta_h),
                float(alpha_max), float(omega_max),
                p_em, float(R_em), float(alpha_s),
            )

            if not all_ok:
                mean_series.append(dict(
                    k=k, dt=dt, feasible=False,
                    u=None,
                    theta_req_avg_deg=float("nan"),
                    theta_s_allowed=float(theta_s_t),
                    coverage=-1,
                    per_agent_ok=per_ok,
                    theta_required=theta_req,
                ))
                continue

            slew_avg_deg = float(np.rad2deg(np.nanmean(theta_req)))

            cov_cnt = -1
            if covpt_ts is not None:
                cov_cnt = int(coverage_count_point(covpt_ts[k], p_agents_k, u_mean, float(theta_h)))

            row = dict(
                k=k, dt=dt, feasible=True,
                u=u_mean,
                theta_req_avg_deg=slew_avg_deg,
                theta_s_allowed=float(theta_s_t),
                coverage=int(cov_cnt),
                per_agent_ok=per_ok,
                theta_required=theta_req,
            )
            mean_series.append(row)

            if best_mean is None:
                best_mean = row  # earliest feasible

        if best_mean is None:
            res_mean_best = self._nan_result()
        else:
            res_mean_best = AttCoordResult(
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

        mean_time = time.time() - start_mean

        return res_opt_best, res_mean_best, opt_series, mean_series, mean_time


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


def compute_J_grid_theta_phi_single_free(
    p_hat, P_p, p_agents, u_curr_agents,
    theta_h,
    *,
    idx_fix: int,
    u_fix,                      # (3,) fixed boresight for idx_fix (already in same frame)
    idx_free: int | None = None, # if None, inferred as the other agent when M=2
    theta_range_rad=None,        # tuple (lo, hi); if None -> (theta_lower_free, theta_upper_free) OR (-pi/2, pi/2)
    phi_range_rad=(0.0, 2*np.pi),
    d_M=3.0, kappa_sigma=100.0, lambda_k1 = 0.5,
    n_mc=20000, n_grid_theta=61, n_grid_phi=121,
    seed=0,
    use_cached_y=True,
):
    """
    Grid over (theta, phi) for ONE free agent, with the other agent fixed.

    Designed for fixed-mode attitude coordination (e.g., M=2).
    Uses u_from_cap(u_curr_free, theta, phi) for the free agent, and u_fix for idx_fix.

    Returns:
      TH_deg  : (n_grid_theta, n_grid_phi) meshgrid of theta in degrees
      PH_deg  : (n_grid_theta, n_grid_phi) meshgrid of phi in degrees
      J_grid  : (n_grid_theta, n_grid_phi) J values
    """
    rng = np.random.default_rng(seed)

    p_agents = np.asarray(p_agents, dtype=float)
    u_curr_agents = np.asarray(u_curr_agents, dtype=float)

    M = int(p_agents.shape[0])
    if M < 2:
        raise ValueError("Need at least 2 agents")
    if not (0 <= idx_fix < M):
        raise ValueError("idx_fix out of range")

    if idx_free is None:
        if M != 2:
            raise ValueError("idx_free must be provided when M != 2")
        idx_free = 1 - int(idx_fix)
    if idx_free == idx_fix:
        raise ValueError("idx_free must differ from idx_fix")
    if not (0 <= idx_free < M):
        raise ValueError("idx_free out of range")

    # normalize u_fix
    u_fix = np.asarray(u_fix, dtype=float).reshape(3,)
    u_fix = u_fix / max(np.linalg.norm(u_fix), 1e-12)

    # MC cache
    if use_cached_y:
        y_cached = make_cached_y(P_p, d_M, n_mc, seed=seed + 123)
    else:
        y_cached = None

    # theta range
    if theta_range_rad is None:
        # If you have per-agent theta bounds in your coordinator, pass them in explicitly.
        # Otherwise, a generic sweep:
        theta_range_rad = (-0.5*np.pi, 0.5*np.pi)

    th_lo, th_hi = map(float, theta_range_rad)
    ph_lo, ph_hi = map(float, phi_range_rad)

    th_vals = np.linspace(th_lo, th_hi, int(n_grid_theta))
    ph_vals = np.linspace(ph_lo, ph_hi, int(n_grid_phi))

    J_grid = np.zeros((th_vals.size, ph_vals.size), dtype=float)

    # full u_agents container
    u_agents = np.zeros_like(u_curr_agents, dtype=float)
    u_agents[idx_fix] = u_fix

    u_curr_free = u_curr_agents[idx_free]
    u_curr_free = np.asarray(u_curr_free, dtype=float).reshape(3,)
    u_curr_free = u_curr_free / max(np.linalg.norm(u_curr_free), 1e-12)

    for a, th in enumerate(th_vals):
        for b, ph in enumerate(ph_vals):
            u_agents[idx_free] = u_from_cap(u_curr_free, float(th), float(ph))

            J_val = J_t_dual_coverage(
                p_hat, P_p, p_agents, u_agents,
                theta_h, d_M=d_M, kappa_sigma=kappa_sigma,
                n_mc=n_mc, y_samples_cached=y_cached, lambda_k1=lambda_k1
            )
            J_grid[a, b] = float(J_val)

    TH, PH = np.meshgrid(th_vals, ph_vals, indexing="ij")
    return np.rad2deg(TH), np.rad2deg(PH), J_grid


