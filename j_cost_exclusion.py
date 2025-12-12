import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D EMS sphere plot
try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
import itertools


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


def finite_diff_grad(f, x, eps=1e-4):
    """Central finite-difference gradient."""
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xm = x.copy()
        xp[i] += eps; xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2*eps)
    return -g  # gradient of -J


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


def init_theta_phi_to_mean(p_hat, p_agents, u_curr_agents, theta_lower, theta_upper, seed, eps=1e-10):
    """
    Initialize (theta_i, phi_i) so u_i ~ direction from p_agents[i] to p_hat,
    respecting slew bounds.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    p_agents = np.asarray(p_agents, dtype=float)
    u_curr_agents = np.asarray(u_curr_agents, dtype=float)
    theta_lower = np.asarray(theta_lower, dtype=float)
    theta_upper = np.asarray(theta_upper, dtype=float)
    M = p_agents.shape[0]
    x0 = np.zeros(2 * M, dtype=float)

    rng = np.random.default_rng(seed=seed)

    for i in range(M):
        p_i = p_agents[i]
        u_curr = u_curr_agents[i] / max(np.linalg.norm(u_curr_agents[i]), eps)

        # Direction from spacecraft to mean
        d_vec = p_hat - p_i
        dist = np.linalg.norm(d_vec)

        if dist < eps:
            # Degenerate: spacecraft at the mean; just keep current pointing
            theta_star = 0.0
            phi_star = 0.0
        else:
            v = d_vec / dist  # desired pointing direction (unit)

            # Build local basis around u_curr
            e1, e2 = orthonormal_basis_from_u(u_curr)

            # Decompose v in {u_curr, e1, e2}
            a  = np.dot(v, u_curr)   # component along u_curr
            b1 = np.dot(v, e1)
            b2 = np.dot(v, e2)
            s  = np.sqrt(b1**2 + b2**2)

            # Ideal slew angle between u_curr and v
            # (angle in [0, pi])
            theta_star = np.arctan2(s, a)

            # Azimuth in the e1/e2 plane
            phi_star = np.arctan2(b2, b1)

            # Clamp theta to feasible slew interval
            # This enforces the hard slew constraint
            theta_star = np.clip(theta_star, theta_lower[i], theta_upper[i])

        # Store in x0 as (theta_i, phi_i)
        possible_values = np.deg2rad([0, 1, -1])
        x0[2*i]   = theta_star + rng.choice(possible_values, size=1)
        x0[2*i+1] = phi_star

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

        if SCIPY_OK:
            def cb(xk, restart_idx=r):
                log_state(xk, restart_idx)

            res = minimize(
                f, x0, method="L-BFGS-B",
                bounds=bounds,
                callback=cb,
                options=dict(maxiter=60, ftol=1e-10, disp=True)
            )
            x_star = res.x
            f_star = res.fun
        else:
            x_star = x0.copy()
            f_star = f(x_star)
            lr = 0.2
            for _ in range(40):
                g = finite_diff_grad(f, x_star, eps=2e-4)
                x_star -= lr * g
                for i in range(M):
                    x_star[2*i]   = np.clip(x_star[2*i], theta_lower[i], theta_upper[i])
                    x_star[2*i+1] = x_star[2*i+1] % (2*np.pi)

                f_new = f(x_star)
                log_state(x_star, r)

                if f_new < f_star:
                    f_star = f_new
                    lr *= 0.95
                else:
                    lr *= 0.5

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


def sample_agents_on_line(N, x_min=-5.0, x_max=5.0, y_line=0.0, seed=None):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x_min, x_max, size=N)
    ys = np.full(N, y_line)
    return np.stack([xs, ys], axis=1)


def sample_target_in_front(x_min=-5.0, x_max=5.0, y_min=2.0, y_max=8.0, seed=None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max)
    y = rng.uniform(y_min, y_max)
    return np.array([x, y])


def mahalanobis_ellipse_points(mu, Sigma, d_mahal=3.0, num_pts=200):
    """
    Points on the ellipse (x-mu)^T Sigma^{-1} (x-mu) = d_mahal^2
    """
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-12)

    t = np.linspace(0, 2*np.pi, num_pts)
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

    left_dir  = dir_from_plus_y(pointing_angle - half_angle)
    right_dir = dir_from_plus_y(pointing_angle + half_angle)

    left_pt  = agent_pos + ray_length * left_dir
    right_pt = agent_pos + ray_length * right_dir

    ax.plot([x0, left_pt[0]],  [y0, left_pt[1]],  color=color, lw=lw)
    ax.plot([x0, right_pt[0]], [y0, right_pt[1]], color=color, lw=lw)

    ax.fill([x0, left_pt[0], right_pt[0]],
            [y0, left_pt[1], right_pt[1]],
            color=color, alpha=alpha)


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

    M = len(p_agents_2d)
    dirs = dir_from_plus_y(pointing_angles)  # (M,2)
    cos_th = np.cos(theta_h)

    counts = np.zeros(len(grid_pts), dtype=int)

    for i in range(M):
        v = grid_pts - p_agents_2d[i]             # (N,2)
        v_norm = np.linalg.norm(v, axis=1)
        good = v_norm > 1e-9                      # avoid divide-by-zero
        v_unit = np.zeros_like(v)
        v_unit[good] = v[good] / v_norm[good,None]

        cosang = np.sum(v_unit * dirs[i], axis=1)
        inside = cosang >= cos_th                 # boolean mask
        counts += inside.astype(int)

    return counts


def sample_from_uncertainty_2d(mu, Sigma, d_mahal=None, rng=None, max_tries=1000):
    """
    Draw a random sample from N(mu, Sigma).
    If d_mahal is provided, reject samples with Mahalanobis distance > d_mahal.
    """
    if rng is None:
        rng = np.random.default_rng()

    L = np.linalg.cholesky(Sigma)

    for _ in range(max_tries):
        z = rng.normal(size=2)          # N(0, I)
        x = mu + L @ z                  # N(mu, Sigma)
        if d_mahal is None:
            return x
        # Mahalanobis distance
        dm2 = z @ z                     # since x = mu + L z => (x-mu)^T Sigma^{-1} (x-mu) = z^T z
        if dm2 <= d_mahal**2:
            return x

    # Fallback if rejection fails (very unlikely for moderate d_mahal)
    return x


def compute_J_grid_thetas_pair(
    p_hat, P_p, p_agents, u_curr_agents,
    theta_h, theta_s_list,
    idx_pair=(0, 1),          # which two thetas to sweep (i, j)
    fixed_thetas=None,        # length M array for the others
    d_M=3.0, kappa_sigma=100.0,
    n_mc=20000, n_grid=50, seed=0,
    p_em=None, R_em=0.3, alpha_s=1.0,
    lambda_em=10.0, beta_zeta=1500.0
):
    """
    Brute-force grid in (theta_i, theta_j) for arbitrary M, with phi_k = 0 for all k.
    All thetas not in idx_pair are held fixed at fixed_thetas[k].

    Returns:
      THI_deg, THJ_deg : meshgrids (in degrees)
      J_grid           : J_t(theta_i, theta_j | others fixed)
    """
    rng = np.random.default_rng(seed)
    M = len(p_agents)
    i, j = idx_pair
    assert 0 <= i < M and 0 <= j < M and i != j

    # MC cache
    y_cached = make_cached_y(P_p, d_M, n_mc, seed=seed+123)

    # default fixed_thetas: zero for all non-swept indices
    if fixed_thetas is None:
        fixed_thetas = np.zeros(M)

    # theta ranges for the swept pair
    # thi_vals = np.linspace(0.0, theta_s_list[i], n_grid)
    # thj_vals = np.linspace(0.0, theta_s_list[j], n_grid)
    thi_vals = np.linspace(np.deg2rad(-90), np.deg2rad(90), n_grid)
    thj_vals = np.linspace(np.deg2rad(-90), np.deg2rad(90), n_grid)

    J_grid = np.zeros((n_grid, n_grid))

    # temp array for all agents' pointings
    u_agents = np.zeros_like(u_curr_agents)

    # precompute the fixed directions for k ≠ i,j
    for k in range(M):
        if k not in idx_pair:
            u_agents[k] = u_from_cap(u_curr_agents[k], fixed_thetas[k], 0.0)

    for a, thi in enumerate(thi_vals):
        for b, thj in enumerate(thj_vals):
            # update just i, j
            u_agents[i] = u_from_cap(u_curr_agents[i], thi, 0.0)
            u_agents[j] = u_from_cap(u_curr_agents[j], thj, 0.0)


            J_val = J_t_dual_coverage(
                p_hat, P_p, p_agents, u_agents,
                theta_h, d_M=d_M, kappa_sigma=kappa_sigma,
                n_mc=n_mc, y_samples_cached=y_cached
            )

            penalty_em = ems_exclusion_penalty(
                p_agents, u_agents,
                p_em=p_em, R_em=R_em,
                theta_h=theta_h,
                alpha_s=alpha_s,
                lambda_em=lambda_em,
                beta_zeta=beta_zeta
            )


            J_grid[a, b] = J_val - penalty_em

    THI, THJ = np.meshgrid(thi_vals, thj_vals, indexing='ij')
    THI_deg = np.rad2deg(THI)
    THJ_deg = np.rad2deg(THJ)
    return THI_deg, THJ_deg, J_grid



# ============================================================
# main()
# ============================================================

def main():
    # =======================
    # User parameters (generalized / randomized)
    # =======================
    seed = int(time.time())
    # seed = 1765481444  # not workng without any penalty
    # seed = 1765490101
    # seed = 1765557699
    # seed = 1765561479
    # seed = 1765566576
    # seed = 1765569259
    # seed = 1765570072
    # seed = 1765570959
    print(seed)
    rng = np.random.default_rng(seed)

    M = 3  # number of spacecraft

    # Spatial region for agents / target (you can tweak these)
    x_line_min, x_line_max = -1.5, -1.5  # reuse as x-bounds for agents
    y_agents_min, y_agents_max = -0.8, 0.8  # reuse as x-bounds for agents

    x_t_min, x_t_max = -1.0, 4.5
    y_t_min, y_t_max = -4.5, 4.5

    # FOV half-angle theta_h: random in a specified range [deg]
    theta_h_min_deg = 2.5
    theta_h_max_deg = 2.5

    # Seed for other randomness (geometry, covariance, etc.)
    # seed = 1764870711  # for not mean when m=2
    # seed = 1764877550  # good for convex
    # seed = 1764965779  # good for all at same place
    # seed = 1765220306

    # seed = int(time.time())
    # print(seed)

    theta_h = np.deg2rad(
        rng.uniform(theta_h_min_deg, theta_h_max_deg)
    )

    # If you still need a generic theta_s_list somewhere:
    theta_s_list = np.array([np.deg2rad(360.0)] * M)



    # -----------------------
    # Agent positions: random in the planar region
    # -----------------------
    x_agents = rng.uniform(x_line_min, x_line_max, size=M)
    y_agents = rng.uniform(y_agents_min, y_agents_max, size=M)
    p_agents_2d = np.stack([x_agents, y_agents], axis=1)

    # -----------------------
    # Target mean position: random in its own box
    # -----------------------
    p_hat_2d = np.array([
        rng.uniform(x_t_min, x_t_max),
        rng.uniform(y_t_min, y_t_max),
    ])

    # -----------------------
    # Initial pointings: axis-aligned, chosen to best point to target
    # -----------------------
    # Angles are measured from +y axis CCW:
    #   0       -> +y
    #   pi/2    -> +x
    #   pi      -> -y
    #   3pi/2   -> -x
    axis_dirs = np.array([
        [0.0, 1.0],  # +y
        [1.0, 0.0],  # +x
        [0.0, -1.0],  # -y
        [-1.0, 0.0],  # -x
    ])
    axis_angles = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])

    pointing_angles = np.zeros(M)
    for i in range(M):
        rel = p_hat_2d - p_agents_2d[i]  # vector from agent to target
        n = np.linalg.norm(rel)
        if n < 1e-9:
            # if target is basically at the same point, just pick +y
            pointing_angles[i] = 0.0
            continue
        rel_unit = rel / n
        dots = axis_dirs @ rel_unit  # cosine with each axis direction
        idx_best = np.argmax(dots)  # most aligned axis
        pointing_angles[i] = axis_angles[idx_best]

    # -----------------------
    # 2D covariance: random eigenvalues + random rotation
    # -----------------------
    # Draw random eigenvalues (spread/scale of uncertainty)
    lambda1, lambda2 = rng.uniform(0.2, 1.5, size=2)
    D = np.diag([lambda1, lambda2])

    # Random 2D rotation
    phi_r = rng.uniform(0.0, 2.0 * np.pi)
    R = np.array([[np.cos(phi_r), -np.sin(phi_r)],
                  [np.sin(phi_r), np.cos(phi_r)]])

    # Covariance = R * D * R^T
    P_p_2d = R @ D @ R.T

    d_mahal = 3.0
    kappa = 1500

    # =======================
    # EMS configuration
    # =======================
    p_em = np.array([0.0, 0.0, 0.0])   # EMS center
    R_em = 0.5                        # EMS effective radius
    alpha_s = np.deg2rad(4.5)          # safety margin
    lambda_em = 1.0                    # penalty weight
    beta_zeta = 1500.0                   # softplus sharpness

    # =======================
    # Initial geometry
    # =======================
    ellipse_pts = mahalanobis_ellipse_points(p_hat_2d, P_p_2d, d_mahal=d_mahal)

    # ----- Embed into 3D for optimizer -----
    p_agents = np.hstack([p_agents_2d, np.zeros((M, 1))])  # (M,3)
    p_hat = np.array([p_hat_2d[0], p_hat_2d[1], 0.0])  # (3,)

    # Pad covariance to 3x3 (small z variance)
    P_p = np.array([[P_p_2d[0, 0], P_p_2d[0, 1], 0.0],
                    [P_p_2d[1, 0], P_p_2d[1, 1], 0.0],
                    [0.0, 0.0, 1e-4]])

    # Current pointing vectors from planar angles (measured from +y)
    # u = [sin(angle), cos(angle), 0]

    u_curr_agents = np.stack([np.sin(pointing_angles),
                              np.cos(pointing_angles),
                              np.zeros(M)], axis=1)

    # ----- Optimize jointly (θ, φ) with EMS penalty -----
    u_star, ang_star, J_star, history, cost_star = optimize_pointing_lbfgs_joint(
        p_hat, P_p, p_agents, u_curr_agents,
        theta_h, theta_s_list,
        d_M=d_mahal, kappa_sigma=kappa,
        n_mc=20000, seed=seed, n_restarts=1,
        p_em=p_em, R_em=R_em,
        alpha_s=alpha_s, lambda_em=lambda_em,
        beta_zeta=beta_zeta
    )

    if u_star is None:
        print("No feasible solution given the slew angle constraints.")
        return

    print("Best objective (−J):", J_star)

    # Per-agent report with slews
    for i, (theta_i, phi_i) in enumerate(ang_star):
        dot = np.dot(u_curr_agents[i], u_star[i])
        dot = np.clip(dot, -1.0, 1.0)
        slew = np.arccos(dot)

        print(f"\nAgent {i}:")
        print(f"   theta = {theta_i:.4f} rad   ({np.rad2deg(theta_i):.2f} deg)")
        print(f"   phi   = {phi_i:.4f} rad   ({np.rad2deg(phi_i):.2f} deg)")
        print(f"   u*    = {u_star[i]}")
        print(f"   slew  = {slew:.4f} rad   ({np.rad2deg(slew):.2f} deg)")

    # True
    sample_pt = sample_from_uncertainty_2d(p_hat_2d, P_p_2d,
                                           d_mahal=d_mahal, rng=rng)


    # ---- Extract history for plotting ----
    slew0 = [entry["slew"][0] for entry in history]
    slew1 = [entry["slew"][1] for entry in history]
    J_hist = [entry["J"] for entry in history]

    if M >= 3:
        slew2 = [entry["slew"][2] for entry in history]
    if M >= 4:
        slew3 = [entry["slew"][3] for entry in history]

    # Slew / J_t evolution plots
    if M == 2:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(np.rad2deg(slew0))
        plt.ylabel("$\\theta_0$ (deg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 2)
        plt.plot(np.rad2deg(slew1))
        plt.ylabel("$\\theta_1$ (deg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 3)
        plt.plot(J_hist)
        plt.xlabel("Logged step")
        plt.ylabel("$J_t$")
        plt.grid(True, alpha=0.3)

    elif M == 3:
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.plot(np.rad2deg(slew0))
        plt.ylabel("Slew 0 (deg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(4, 1, 2)
        plt.plot(np.rad2deg(slew1))
        plt.ylabel("Slew 1 (deg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(4, 1, 3)
        plt.plot(np.rad2deg(slew2))
        plt.ylabel("Slew 2 (deg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(4, 1, 4)
        plt.plot(J_hist)
        plt.xlabel("Logged step")
        plt.ylabel("J_t")
        plt.grid(True, alpha=0.3)

    else:
        plt.figure()
        plt.subplot(5, 1, 1)
        plt.plot(np.rad2deg(slew0))
        plt.ylabel("Slew 0 (deg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(5, 1, 2)
        plt.plot(np.rad2deg(slew1))
        plt.ylabel("Slew 1 (deg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(5, 1, 3)
        plt.plot(np.rad2deg(slew2))
        plt.ylabel("Slew 2 (deg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(5, 1, 4)
        plt.plot(np.rad2deg(slew3))
        plt.ylabel("Slew 3 (deg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(5, 1, 5)
        plt.plot(J_hist)
        plt.xlabel("Logged step")
        plt.ylabel("J_t")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert optimized 3D u back to planar pointing angles
    pointing_angles_opt = np.arctan2(u_star[:, 0], u_star[:, 1])

    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # ----- Coverage shading + geometry plot -----
    fig, ax = plt.subplots(figsize=(8, 6))

    # Background coverage map
    Nx = 500
    Ny = 500
    xg = np.linspace(x_line_min - 2, x_line_max + 10, Nx)
    yg = np.linspace(y_agents_min - 8, y_t_max + 8, Ny)
    XX, YY = np.meshgrid(xg, yg)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

    coverage = coverage_count_2d(
        grid,
        p_agents_2d,
        pointing_angles_opt,
        theta_h
    )

    coverage_img = coverage.reshape(Ny, Nx)

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap_colors = np.array([
        [1, 1, 1, 1],  # 0 coverage = white
        [0.6, 0.8, 1, 1],  # 1 coverage = light blue
        [0.2, 0.7, 0.2, 1],  # 2 coverage = green (dual)
        [1.0, 0.0, 0.0, 1]  # ≥3 coverage = red (triple+)
    ])

    cov_cmap = ListedColormap(cmap_colors)

    # boundaries for bins around 0,1,2,3
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cov_cmap.N)

    im = ax.imshow(
        coverage_img,
        extent=[xg.min(), xg.max(), yg.min(), yg.max()],
        origin='lower',
        cmap=cov_cmap,
        norm=norm,  # <-- key line
        alpha=0.25,
        zorder=-5
    )

    # FOV wedges and agents
    agent_scatter = None
    for i, pos in enumerate(p_agents_2d):
        # FOV wedge (no direct label; we'll make a proxy for legend)
        plot_fov_wedge(ax, pos, pointing_angles_opt[i], theta_h,
                       ray_length=10.0, color='tab:blue', alpha=0.12, lw=1.5)
        sc = ax.scatter(pos[0], pos[1], color='tab:blue', s=60,
                        label='Agent position' if i == 0 else None)
        if agent_scatter is None:
            agent_scatter = sc
        ax.text(pos[0], pos[1] - 0.35, f"A{i}", color='tab:blue',
                ha='center', va='top')

    # Draw current boresight axis directions (u_curr) as dotted lines,
    # and annotate slew angle between u_curr and optimized pointing.
    u_axis_proxy = None  # for legend handle

    for i, pos in enumerate(p_agents_2d):
        # Current boresight (initial attitude) in 2D
        u_curr_2d = u_curr_agents[i, :2]

        # Optimized pointing direction in 2D from the optimized angle
        u_opt_2d = np.array([
            np.sin(pointing_angles_opt[i]),
            np.cos(pointing_angles_opt[i])
        ])

        # Endpoint for the boresight line (a short segment)
        p_end = pos + u_curr_2d * 3.0

        ln, = ax.plot(
            [pos[0], p_end[0]],
            [pos[1], p_end[1]],
            linestyle=':',
            color='black',
            lw=1.2,
            label='Initial boresight' if i == 0 else None
        )
        if u_axis_proxy is None:
            u_axis_proxy = ln

        # Slew angle between current and optimized
        dot = np.dot(u_curr_2d, u_opt_2d)
        dot = np.clip(dot, -1.0, 1.0)
        slew_rad = np.arccos(dot)
        slew_deg = np.rad2deg(slew_rad)

        # Place text slightly above the agent to avoid overlap
        ax.text(
            pos[0],
            pos[1] + 0.5,
            f"{slew_deg:.1f}°",
            ha='center',
            va='bottom',
            fontsize=9,
            color='black'
        )


    # Target mean + ellipse
    unc_mean_sc = ax.scatter(p_hat_2d[0], p_hat_2d[1], color='tab:red', s=80,
                             marker='x', linewidths=2, label='Uncertainty mean')
    ellipse_line, = ax.plot(ellipse_pts[:, 0], ellipse_pts[:, 1],
                            color='tab:red', lw=2, label='Uncertainty ellipse')
    ax.fill(ellipse_pts[:, 0], ellipse_pts[:, 1],
            color='tab:red', alpha=0.10)

    # Random sample from uncertainty distribution (within same d_M)

    true_sc = ax.scatter(sample_pt[0], sample_pt[1],
                         s=60, facecolors='none', edgecolors='green',
                         linewidths=2, label='True position')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    # ax.set_title(...)  # removed title

    # ax.set_xlim(x_line_min - 10, x_line_max + 10)
    # ax.set_ylim(y_line - 10, y_t_max + 10)
    plt.grid(alpha=0.25)

    # ---- Comprehensive legend ----

    # Proxy artists for coverage levels
    single_cov_patch = Patch(facecolor=cmap_colors[1], alpha=0.25, label='Single coverage')
    double_cov_patch = Patch(facecolor=cmap_colors[2], alpha=0.25, label='Double coverage')
    triple_cov_patch = Patch(facecolor=cmap_colors[3], alpha=0.25, label='Triple+ coverage')

    # Proxy for FOV wedge (blue lines)
    fov_proxy = Line2D([0], [0], color='tab:blue', lw=1.5, label='Agent FOV')

    handles = [
        agent_scatter,
        fov_proxy,
        # u_axis_proxy,  # ← NEW: initial boresight
        unc_mean_sc,
        ellipse_line,
        # true_sc,
        single_cov_patch,
        double_cov_patch,
        triple_cov_patch,
    ]

    ax.legend(handles=handles, loc='upper right')

    if R_em > 0.0:
        z_plane = 0.0
        dz = z_plane - p_em[2]
        if abs(dz) <= R_em:
            r_xy = np.sqrt(R_em ** 2 - dz ** 2)
            theta_c = np.linspace(0, 2 * np.pi, 200)
            x_c = p_em[0] + r_xy * np.cos(theta_c)
            y_c = p_em[1] + r_xy * np.sin(theta_c)
            ax.plot(x_c, y_c, color='orange', lw=2, label='EMS sphere (cross-section)')
            ax.fill(x_c, y_c, color='orange', alpha=0.1)
            ax.scatter(p_em[0], p_em[1], color='orange', s=60, marker='o')
            ax.text(p_em[0], p_em[1] + 0.3, "EMS", color='orange',
                    ha='center', va='bottom')

    # ===== 3D visualization of agents, target mean, and EMS sphere =====
    if False:
        fig3d = plt.figure(figsize=(7, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')

        # EMS sphere
        u_s = np.linspace(0, 2*np.pi, 50)
        v_s = np.linspace(0, np.pi, 50)
        uu, vv = np.meshgrid(u_s, v_s)

        x_s = p_em[0] + R_em * np.cos(uu) * np.sin(vv)
        y_s = p_em[1] + R_em * np.sin(uu) * np.sin(vv)
        z_s = p_em[2] + R_em * np.cos(vv)

        ax3d.plot_surface(x_s, y_s, z_s, alpha=0.3, color='orange', edgecolor='none')
        ax3d.scatter(p_em[0], p_em[1], p_em[2], color='orange', s=40)

        # Agents
        ax3d.scatter(p_agents[:, 0], p_agents[:, 1], p_agents[:, 2],
                     color='tab:blue', s=40)
        for i in range(M):
            ax3d.text(p_agents[i, 0], p_agents[i, 1], p_agents[i, 2],
                      f"A{i}", color='tab:blue')

            # draw short line showing pointing direction
            L_dir = 2.0
            p_end = p_agents[i] + L_dir * u_star[i]
            ax3d.plot([p_agents[i, 0], p_end[0]],
                      [p_agents[i, 1], p_end[1]],
                      [p_agents[i, 2], p_end[2]],
                      color='tab:blue', lw=1.5)

        # Target mean
        ax3d.scatter(p_hat[0], p_hat[1], p_hat[2],
                     color='tab:red', s=50, marker='x')

        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.set_title("3D View: Agents, Pointing, and EMS Sphere")

        # make aspect roughly equal
        all_pts = np.vstack([p_agents, p_hat[None, :], p_em[None, :]])
        max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
        mid = all_pts.mean(axis=0)
        for axis, m in zip([ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim], mid):
            axis(m - max_range/2, m + max_range/2)

    # Optional: J_t(θ_1, θ_2) surface, like before
    if False:
        fixed_thetas = [np.deg2rad(0), 0]

        TH12_1, TH12_2, J12 = compute_J_grid_thetas_pair(
            p_hat, P_p, p_agents, u_curr_agents,
            theta_h, theta_s_list,
            idx_pair=(0, 1),
            fixed_thetas=fixed_thetas,
            d_M=d_mahal, kappa_sigma=kappa,
            n_mc=20000, n_grid=50, seed=seed,
            p_em=p_em, R_em=R_em, alpha_s=alpha_s,
            lambda_em=lambda_em, beta_zeta=beta_zeta
        )

        # 3D surface plot: theta1 vs theta2 vs J_t
        fig3d = plt.figure(figsize=(8, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')

        ax3d.plot_surface(
            TH12_1, TH12_2, J12,
            rstride=1, cstride=1, linewidth=0.2, alpha=0.9
        )
        ax3d.set_xlabel(r'$\theta_1$ (deg)')
        ax3d.set_ylabel(r'$\theta_2$ (deg)')
        ax3d.set_zlabel(r'$J_t$')
        ax3d.set_title(r'$J_t(\theta_0=0.2^{\circ}, \theta_1,\theta_2)$')

        plt.tight_layout()

        plt.figure(figsize=(6, 5))
        cs = plt.contourf(TH12_1, TH12_2, J12, levels=30)
        plt.colorbar(cs, label=r'$J_t$')
        plt.xlabel(r'$\theta_1$ (deg)')
        plt.ylabel(r'$\theta_2$ (deg)')
        plt.title(r'$J_t(\theta_0=0.2^{\circ}, \theta_1,\theta_2)$')
        plt.grid(alpha=0.3)

        # determine how many restarts we actually have in history
        restart_indices = sorted({entry["restart"] for entry in history})

        # choose colors and markers to cycle through
        colors = ['white', 'yellow', 'cyan', 'magenta', 'green', 'orange']
        markers = ['o', 's', '^', 'D', 'x', '+']

        for k, r in enumerate(restart_indices):
            # Extract all states for this restart
            path_entries = [entry for entry in history if entry["restart"] == r]
            if not path_entries:
                continue

            # Extract θ1, θ2 in *degrees* over the path
            theta1_path = []
            theta2_path = []
            for entry in path_entries:
                xk = entry["x"]
                theta1_path.append(np.rad2deg(xk[0]) * np.cos(xk[1]))  # agent 0 θ
                theta2_path.append(np.rad2deg(xk[2]) * np.cos(xk[3]))  # agent 1 θ

            theta1_path = np.array(theta1_path)
            theta2_path = np.array(theta2_path)

            col = colors[k % len(colors)]
            m = markers[k % len(markers)]

            label = f"Trial {r}"
            if r == 0:
                label += " (warm start)"

            # line + markers
            plt.plot(theta1_path, theta2_path,
                     linestyle='-',
                     marker=m,
                     color=col,
                     lw=1.5,
                     ms=5,
                     label=label)

            # optional: show arrow on last segment to emphasize direction
            # if len(theta1_path) > 1:
            #     plt.annotate("",
            #                  xy=(theta1_path[-1], theta2_path[-1]),
            #                  xytext=(theta1_path[-2], theta2_path[-2]),
            #                  arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

        plt.legend(loc='upper right')

    plt.show()


if __name__ == "__main__":
    main()
