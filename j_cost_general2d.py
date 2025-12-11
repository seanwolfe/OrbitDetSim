import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D  # at top of file, if not already
try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


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
        return C[0] * C[1]

    one_minus_C = 1.0 - C

    if M ==3:
        k2 = C[0] * C[1] * one_minus_C[2] + C[0] * C[2] * one_minus_C[1] + C[1] * C[2] * one_minus_C[0]
        k1 = 0.1 * (C[0] * one_minus_C[1] * one_minus_C[2] + C[2] * one_minus_C[0] * one_minus_C[1] + C[1] * one_minus_C[0] * one_minus_C[2])
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
        k1 = 0.1 * (
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
        k1 = 0.1 * (
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
        k1 = 0.1 * (
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
        for j in range(i+1, M):
            denom = one_minus_C[i] * one_minus_C[j]

            # Avoid 0/0 only by masking, not by changing the denominator value
            prod_excl = np.zeros_like(prod_all)
            mask = denom > 1e-30
            prod_excl[mask] = prod_all[mask] / denom[mask]
            # where denom ~ 0, prod_excl is irrelevant: either product_all is ~0 too,
            # or C[i]*C[j] is tiny/ill-defined; leaving it 0 is fine

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
                    y_cached):
    M = len(p_agents)
    u_agents = angles_to_pointings(x, p_agents, u_curr_agents, theta_lower, theta_upper, M)
    # We maximize J, so objective is -J
    return -J_t_dual_coverage(
        p_hat, P_p, p_agents, u_agents,
        theta_h, d_M=d_M, kappa_sigma=kappa_sigma,
        n_mc=y_cached.shape[0], y_samples_cached=y_cached
    )


def init_theta_phi_to_mean(p_hat, p_agents, u_curr_agents, theta_lower, theta_upper, seed, eps=1e-10):
    """
    Initialize (theta_i, phi_i) for each spacecraft i so that the resulting pointing
    vector u_i is as close as possible to the direction from p_agents[i] to p_hat,
    but respecting the slew constraints:
        theta_lower[i] <= theta_i <= theta_upper[i].

    Parameters
    ----------
    p_hat : array_like, shape (3,)
        Mean position of the uncertainty ellipsoid in 3D.
    p_agents : array_like, shape (M,3)
        Positions of the M spacecraft.
    u_curr_agents : array_like, shape (M,3)
        Current boresight unit vectors for each spacecraft.
    theta_lower : array_like, shape (M,)
        Lower bounds on the slew angle for each spacecraft (usually >= 0).
    theta_upper : array_like, shape (M,)
        Upper bounds on the slew angle for each spacecraft.

    Returns
    -------
    x0 : ndarray, shape (2*M,)
        Initial parameter vector:
            x0 = [theta_0, phi_0, theta_1, phi_1, ..., theta_{M-1}, phi_{M-1}]
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


### NEW: helper to estimate theta_min and theta_max from the uncertainty ellipsoid
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

    return theta_min, theta_max


def optimize_pointing_lbfgs_joint(
    p_hat, P_p, p_agents, u_curr_agents,
    theta_h, theta_s_list,
    d_M=3.0, kappa_sigma=120.0,
    n_mc=25000, seed=0,
    n_restarts=3
):
    """
    Joint L-BFGS-B over all agents’ (theta_i,phi_i), with θ-bounds derived from
    the uncertainty ellipsoid and slew constraints.

    Algorithm is the same as your original j_cost version (maximize J_t over
    θ,φ with ellipsoid-based θ-bounds). The only additions are:
      - A history log of iterations (slew angles, J_t, etc.)
      - An extra return value `history`.

    Returns:
      u_best      : (M,3) array of best pointing vectors
      angles_best : list[(theta_i, phi_i)] for each spacecraft
      J_best      : best J_t value (maximized)
      history     : list of dicts with entries:
                       {
                         "x":    parameter vector at that step (θ/φ),
                         "u":    pointing vectors (M,3),
                         "slew": slew angles (M,) [rad],
                         "J":    J_t at this point,
                         "restart": restart index
                       }

    If infeasible (slew too small to even reach the near edge of the ellipsoid
    for any spacecraft), returns (None, None, 0.0, []).
    """
    rng = np.random.default_rng(seed)
    M = len(p_agents)

    # --- θ-bounds from uncertainty ellipsoid + slew ---
    theta_min_ell, theta_max_ell = estimate_theta_bounds_from_ellipsoid(
        p_hat, P_p, p_agents, u_curr_agents, d_M,
        n_shell=400, seed=seed+999
    )

    theta_upper = np.minimum(theta_max_ell, theta_s_list)
    theta_lower = theta_min_ell.copy()
    theta_lower = np.maximum(theta_lower, 0.0)

    infeasible_mask = theta_upper < theta_lower
    if np.any(infeasible_mask):
        print("Infeasible: for at least one spacecraft, slew limit is smaller "
              "than the angle required to reach the near edge of the uncertainty ellipsoid.")
        return None, None, 0.0, []

    # MC points cache
    y_cached = make_cached_y(P_p, d_M, n_mc, seed=seed+123)

    # Bounds in (θ, φ) space
    bounds = []
    for i in range(M):
        bounds.append((theta_lower[i], theta_upper[i]))  # theta_i
        bounds.append((0.0, 2*np.pi))                    # phi_i

    # ---- History logger (new) ----
    history = []

    def log_state(x, restart_idx):
        """
        Record current state:
          - u from current (θ, φ)
          - slew between u_curr and u
          - J_t at this point
        """
        u = angles_to_pointings(x, p_agents, u_curr_agents, theta_lower, theta_upper, M)

        # Slew angles between current and candidate pointings
        dots = np.einsum("ij,ij->i", u_curr_agents, u)
        dots = np.clip(dots, -1.0, 1.0)
        slews = np.arccos(dots)

        # Pure J_t (no penalties)
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

    # ---- Optimization (same algorithm as before) ----
    best_x = None
    best_f = np.inf  # we minimize f = -J_t
    # before the restart loop, after you have theta_lower, theta_upper, etc.
    x0_mean = init_theta_phi_to_mean(
        p_hat, p_agents, u_curr_agents,
        theta_lower, theta_upper, seed
    )

    for r in range(n_restarts):
        if r == 0:
            # warm start: all agents roughly pointing toward the mean
            x0 = x0_mean.copy()
        else:
            # small random perturbation around the warm start
            noise = rng.normal(scale=0.05, size=2 * M)  # tweak scale if you want
            print(np.rad2deg(noise))
            x0 = x0_mean + noise
            # make sure θ stays in bounds and wrap φ
            for i in range(M):
                x0[2 * i] = np.clip(x0[2 * i], theta_lower[i], theta_upper[i])
                x0[2 * i + 1] = x0[2 * i + 1] % (2 * np.pi)

        # log initial state
        log_state(x0, r)

        f = lambda z: objective_joint(
            z, p_hat, P_p, p_agents, u_curr_agents,
            theta_lower, theta_upper,
            theta_h, d_M, kappa_sigma, y_cached
        )

        if SCIPY_OK:
            def cb(xk, restart_idx=r):
                # log each L-BFGS-B iteration
                log_state(xk, restart_idx)

            res = minimize(
                f, x0, method="L-BFGS-B",
                bounds=bounds,
                callback=cb,
                options=dict(maxiter=60, ftol=1e-10, disp=False)
            )
            x_star = res.x
            f_star = res.fun
        else:
            # fallback: crude joint random + gradient descent
            x_star = x0.copy()
            f_star = f(x_star)
            lr = 0.2
            for _ in range(40):
                g = finite_diff_grad(f, x_star, eps=2e-4)
                x_star -= lr * g
                # project to bounds
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

    # Unpack best and compute actual J_t (not −J)
    thetas_best, phis_best = unpack_angles(best_x, M)
    u_best = angles_to_pointings(best_x, p_agents, u_curr_agents, theta_lower, theta_upper, M)

    J_best = J_t_dual_coverage(
        p_hat, P_p, p_agents, u_best,
        theta_h, d_M=d_M, kappa_sigma=kappa_sigma,
        n_mc=y_cached.shape[0], y_samples_cached=y_cached
    )

    angles_best = [(thetas_best[i], phis_best[i]) for i in range(M)]
    return u_best, angles_best, J_best, history



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
    Returns points on the ellipse defined by (x-mu)^T Sigma^{-1} (x-mu) = d_mahal^2
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

    # direction vector for planar angle = rotation from +y
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
    n_mc=20000, n_grid=50, seed=0
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
            J_grid[a, b] = J_val

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
    seed = 1765220309
    seed = 1765481444
    rng = np.random.default_rng(seed)

    M = 2  # number of spacecraft

    # Spatial region for agents / target (you can tweak these)
    x_line_min, x_line_max = -3.0, 3.0  # reuse as x-bounds for agents
    y_agents_min, y_agents_max = -6.0, 6.0  # reuse as x-bounds for agents

    x_t_min, x_t_max = -6, 6.0
    y_t_min, y_t_max = -6.0, 6.0

    # FOV half-angle theta_h: random in a specified range [deg]
    theta_h_min_deg = 2.0
    theta_h_max_deg = 15.0

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


    # ----- Optimize jointly (θ, φ) with logging -----
    u_star, ang_star, J_star, history = optimize_pointing_lbfgs_joint(
        p_hat, P_p, p_agents, u_curr_agents,
        theta_h, theta_s_list,
        d_M=d_mahal, kappa_sigma=kappa,
        n_mc=20000, seed=seed, n_restarts=1
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
    xg = np.linspace(x_line_min - 5, x_line_max + 10, Nx)
    yg = np.linspace(y_agents_min - 1, y_t_max + 8, Ny)
    XX, YY = np.meshgrid(xg, yg)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

    coverage = coverage_count_2d(
        grid,
        p_agents_2d,
        pointing_angles_opt,
        theta_h
    )

    coverage_img = coverage.reshape(Ny, Nx)

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

    # Optional: J_t(θ_1, θ_2) surface, like before
    if False:
        fixed_thetas = [np.deg2rad(62.44), 0, 0]

        TH12_1, TH12_2, J12 = compute_J_grid_thetas_pair(
            p_hat, P_p, p_agents, u_curr_agents,
            theta_h, theta_s_list,
            idx_pair=(1, 2),
            fixed_thetas=fixed_thetas,
            d_M=d_mahal, kappa_sigma=kappa,
            n_mc=20000, n_grid=100, seed=seed
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
                theta1_path.append(np.rad2deg(xk[2]))  # agent 0 θ
                theta2_path.append(np.rad2deg(xk[4]))  # agent 1 θ

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
