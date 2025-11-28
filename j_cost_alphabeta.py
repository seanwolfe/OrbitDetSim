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
def unpack_ab(x, M):
    """x = [alpha1,beta1,...,alphaM,betaM] -> arrays (M,)."""
    alphas = x[0::2]
    betas  = x[1::2]
    return alphas, betas


def ellipsoid_point_from_angles(p_hat, Lp, d_M, alpha, beta):
    """
    Map (alpha, beta) -> point on ellipsoid boundary.
    alpha in [0, 2π), beta in [-π/2, π/2].
    y is on a sphere of radius d_M in whitened coords.
    """
    y = d_M * np.array([
        np.cos(alpha) * np.cos(beta),
        np.sin(alpha) * np.cos(beta),
        np.sin(beta)
    ])  # (3,)
    p = p_hat + Lp @ y
    return p


def angles_to_pointings_ellipsoid(x, Lp, p_hat, p_agents, d_M):
    """
    Given joint (alpha_i, beta_i) for all spacecraft, compute u_agents (M,3)
    where each u_i points from p_agents[i] to a point on the ellipsoid boundary.
    """
    M = len(p_agents)
    alphas, betas = unpack_ab(x, M)
    u_agents = np.zeros((M, 3))
    for i in range(M):
        p_target = ellipsoid_point_from_angles(p_hat, Lp, d_M, alphas[i], betas[i])
        r = p_target - p_agents[i]
        u_agents[i] = r / np.linalg.norm(r)
    return u_agents


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

    one_minus_C = 1.0 - C
    prod_all = np.prod(one_minus_C, axis=0)

    k2 = np.zeros(N)
    for i in range(M):
        for j in range(i+1, M):  # <-- i < j (no i=j terms)
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


### NEW: helper to estimate theta_min and theta_max from the uncertainty ellipsoid
def objective_joint_ellipsoid(
    x, Lp, p_hat, P_p, p_agents, u_curr_agents, theta_s_list,
    theta_h, d_M, kappa_sigma, y_cached,
    lambda_slew=0.0
):
    """
    Objective for L-BFGS-B in the ellipsoid-param space.
    f(x) = -(J_t(x) - slew_penalty(x))  -> minimize f => maximize J_t - penalty
    """
    # Current pointings from ellipsoid parameters
    u_agents = angles_to_pointings_ellipsoid(x, Lp, p_hat, p_agents, d_M)

    # Dual-coverage cost J_t
    J_val = J_t_dual_coverage(
        p_hat, P_p, p_agents, u_agents,
        theta_h, d_M=d_M, kappa_sigma=kappa_sigma,
        n_mc=y_cached.shape[0], y_samples_cached=y_cached
    )

    # Optional slew penalty
    penalty = 0.0
    if lambda_slew > 400.0:
        # angle between u_curr and u_agents
        dots = np.einsum('ij,ij->i', u_curr_agents, u_agents)
        dots = np.clip(dots, -1.0, 1.0)
        thetas = np.arccos(dots)  # [0, π]

        excess = np.maximum(thetas - theta_s_list, 0.0)
        penalty = lambda_slew * np.sum(excess**2)

    # Minimize f = -(J - penalty) = penalty - J
    return -(J_val - penalty)


def init_ab_to_mean(p_hat, P_p, p_agents, d_M, eps=1e-10):
    """
    Initialize (alpha_i, beta_i) so that for each spacecraft i, the pointing
    direction u_i goes along the line from spacecraft i to the mean p_hat,
    and hits the ellipsoid boundary where (p - p_hat)^T P_p^{-1} (p - p_hat) = d_M^2.

    Returns:
        x0 : shape (2*M,) = [alpha_0, beta_0, alpha_1, beta_1, ...]
    """
    M = len(p_agents)
    x0 = np.zeros(2 * M)

    # Cholesky and inverse of covariance
    Lp = np.linalg.cholesky(P_p)
    Pinv = np.linalg.inv(P_p)

    for i in range(M):
        p_i = p_agents[i]

        # Direction from spacecraft to mean
        d_vec = p_hat - p_i
        dist = np.linalg.norm(d_vec)
        if dist < eps:
            # Spacecraft at the mean (degenerate case) -> pick arbitrary direction
            d_hat = np.array([0.0, 0.0, 1.0])
        else:
            d_hat = d_vec / dist   # unit vector from s/c to mean

        # Line: p(λ) = p_i + λ d_hat, λ >= 0
        # Ellipsoid condition: (p(λ)-p_hat)^T P^{-1} (p(λ)-p_hat) = d_M^2
        # Let Δ = p_i - p_hat
        # => (Δ + λ d_hat)^T P^{-1} (Δ + λ d_hat) = d_M^2
        # => A λ^2 + 2 B λ + C = 0, with
        #    A = d_hat^T P^{-1} d_hat
        #    B = Δ^T P^{-1} d_hat
        #    C = Δ^T P^{-1} Δ - d_M^2

        Delta = p_i - p_hat
        A = d_hat @ (Pinv @ d_hat)
        B = Delta @ (Pinv @ d_hat)
        C = (Delta @ (Pinv @ Delta)) - d_M**2

        disc = B**2 - A * C

        if disc < 0:
            # No real intersection (numerical issues or geometry weird) fallback:
            # Take ray from center outwards in direction d_hat and intersect that.
            # p(τ) = p_hat + τ d_hat
            # => τ^2 d_hat^T P^{-1} d_hat = d_M^2 => τ = d_M / sqrt(A)
            tau = d_M / np.sqrt(max(A, eps))
            p_target = p_hat + tau * d_hat
        else:
            sqrt_disc = np.sqrt(disc)
            lambda1 = (-B + sqrt_disc) / max(A, eps)
            lambda2 = (-B - sqrt_disc) / max(A, eps)

            # We want the intersection further along the line from the spacecraft
            # going past the mean: typically the larger positive λ.
            lambdas = [lambda1, lambda2]
            lambdas_pos = [lam for lam in lambdas if lam > 0]

            if not lambdas_pos:
                # no positive solution? fallback to center-ray as above
                tau = d_M / np.sqrt(max(A, eps))
                p_target = p_hat + tau * d_hat
            else:
                lam = max(lambdas_pos)
                p_target = p_i + lam * d_hat

        # Now p_target lies on the ellipsoid. Convert to whitened coords:
        y = np.linalg.solve(Lp, p_target - p_hat)   # (3,)
        # Norm should be ~ d_M
        y_norm = np.linalg.norm(y)
        if y_norm < eps:
            # Degenerate; just pick some direction
            y_dir = np.array([1.0, 0.0, 0.0])
        else:
            y_dir = y / y_norm

        # Map y_dir to (alpha, beta) via your spherical parameterization:
        # y_dir = [cos(alpha)*cos(beta), sin(alpha)*cos(beta), sin(beta)]
        beta = np.arcsin(y_dir[2])
        alpha = np.arctan2(y_dir[1], y_dir[0])

        x0[2*i]   = alpha
        x0[2*i+1] = beta

    return x0


def optimize_pointing_lbfgs_joint(
    p_hat, P_p, p_agents, u_curr_agents,
    theta_h, theta_s_list,
    d_M=3.0, kappa_sigma=120.0,
    n_mc=25000, seed=0,
    n_restarts=3,
    lambda_slew=0.0   # >0 to enforce slew softly, 0 to ignore slew
):
    """
    Joint L-BFGS-B over all agents’ (alpha_i, beta_i) on the ellipsoid boundary.

    Parametrization:
      - For each spacecraft i, (alpha_i, beta_i) picks a point on the ellipsoid
        boundary in whitened space, which is then mapped to R^3 and used as
        the look point for that spacecraft.
      - The pointing vector u_i is the unit vector from p_i^o to that point.

    Bounds:
      alpha_i ∈ [0, 2π),  beta_i ∈ [-π/2, π/2].

    Returns:
      u_best      : (M, 3) array of best pointing vectors
      angles_best : list of (alpha_i, beta_i) for each spacecraft
      J_best      : best J_t value (maximized)
      history     : list of dicts with entries:
                       {
                         "x":    current parameter vector,
                         "u":    current pointing vectors (M,3),
                         "slew": slew angles (M,) [rad],
                         "J":    J_t at this point,
                         "restart": restart index
                       }
    """
    rng = np.random.default_rng(seed)
    M = len(p_agents)

    # Cholesky of covariance (for ellipsoid mapping)
    Lp = np.linalg.cholesky(P_p)

    # Cache MC points to smooth objective
    y_cached = make_cached_y(P_p, d_M, n_mc, seed=seed+123)

    # Box bounds in (alpha, beta) for each spacecraft
    bounds = []
    for i in range(M):
        bounds.append((0.0, 2*np.pi))           # alpha_i
        bounds.append((-0.5*np.pi, 0.5*np.pi))  # beta_i

    # Global history of all restarts / iterations
    history = []

    # Helper to log the current optimization state
    def log_state(x, restart_idx):
        # Pointings from current parameters
        u = angles_to_pointings_ellipsoid(x, Lp, p_hat, p_agents, d_M)

        # Slew angles between current and optimized pointings
        dots = np.einsum("ij,ij->i", u_curr_agents, u)
        dots = np.clip(dots, -1.0, 1.0)
        slews = np.arccos(dots)  # [0, π]

        # Dual-coverage cost J_t (no slew penalty)
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
    best_f = np.inf  # we are minimizing f = -(J - penalty)

    Lp = np.linalg.cholesky(P_p)

    # Warm-start: each spacecraft points toward the mean, intersecting the ellipsoid
    x0_mean = init_ab_to_mean(p_hat, P_p, p_agents, d_M)

    for r in range(n_restarts):

        if r == 0:
            x0 = x0_mean.copy()  # deterministic warm-start
        else:
            # small random perturbation around that
            noise = rng.normal(scale=0.1, size=2 * M)
            x0 = x0_mean + noise
        # Random initial guess inside bounds
        # x0 = np.zeros(2*M)
        # for i in range(M):
        #     x0[2*i]   = rng.uniform(0.0, 2*np.pi)          # alpha_i
        #     x0[2*i+1] = rng.uniform(-0.5*np.pi, 0.5*np.pi) # beta_i

        # Log the initial state for this restart
        log_state(x0, r)

        # Objective with slew penalty
        f = lambda z: objective_joint_ellipsoid(
            z, Lp, p_hat, P_p, p_agents, u_curr_agents, theta_s_list,
            theta_h, d_M, kappa_sigma, y_cached,
            lambda_slew=lambda_slew
        )

        if SCIPY_OK:
            # Define callback that logs each L-BFGS-B iteration
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
        else:
            # Fallback: crude random + gradient descent
            x_star = x0.copy()
            f_star = f(x_star)
            lr = 0.2
            for _ in range(40):
                # finite_diff_grad in your code currently returns -∇f,
                # so stepping x -= lr * g moves *uphill* in J (downhill in f)
                g = finite_diff_grad(f, x_star, eps=2e-4)
                x_star -= lr * g

                # Project to bounds
                for i in range(M):
                    x_star[2*i]   = np.clip(x_star[2*i],     0.0,       2*np.pi)
                    x_star[2*i+1] = np.clip(x_star[2*i+1], -0.5*np.pi,  0.5*np.pi)

                f_new = f(x_star)
                # Log state after this update
                log_state(x_star, r)

                if f_new < f_star:
                    f_star = f_new
                    lr *= 0.95
                else:
                    lr *= 0.5

        if f_star < best_f:
            best_f = f_star
            best_x = x_star.copy()

    # Unpack best solution and compute actual J_t (without slew penalty) for reporting
    alphas_best, betas_best = unpack_ab(best_x, M)
    u_best = angles_to_pointings_ellipsoid(best_x, Lp, p_hat, p_agents, d_M)

    J_best = J_t_dual_coverage(
        p_hat, P_p, p_agents, u_best,
        theta_h, d_M=d_M, kappa_sigma=kappa_sigma,
        n_mc=y_cached.shape[0], y_samples_cached=y_cached
    )

    angles_best = [(alphas_best[i], betas_best[i]) for i in range(M)]

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
    thi_vals = np.linspace(0.0, theta_s_list[i], n_grid)
    thj_vals = np.linspace(0.0, theta_s_list[j], n_grid)

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
# Your main(), now working
# ============================================================

def main():
    # =======================
    # User parameters
    # =======================
    M = 5

    x_line_min, x_line_max = -1.0, 1.0
    y_line = 0.0

    x_t_min, x_t_max = 2.0, 4.0
    y_t_min, y_t_max = 1.0, 5.0

    theta_h = np.deg2rad(2.5)
    theta_s_list = np.array([np.deg2rad(90.0)] * M)

    rng = np.random.default_rng(0)
    # near +y
    pointing_angles = np.deg2rad(rng.uniform(-1e-5, 1e-5, size=M))

    # 2D covariance of target
    P_p_2d = np.array([[0.25, 0.1],
                       [-0.1, 0.5]])

    d_mahal = 3.0

    # seed = int(time.time())
    # print(seed)
    seed = 1764359905
    kappa = 2000
    # =======================

    # 2D positions
    p_agents_2d = sample_agents_on_line(M, x_line_min, x_line_max, y_line, seed=seed)
    p_hat_2d = sample_target_in_front(x_t_min, x_t_max, y_t_min, y_t_max, seed=seed+10)

    ellipse_pts = mahalanobis_ellipse_points(p_hat_2d, P_p_2d, d_mahal=d_mahal)

    # ----- Embed into 3D for optimizer -----
    p_agents = np.hstack([p_agents_2d, np.zeros((M,1))])           # (M,3)
    p_hat = np.array([p_hat_2d[0], p_hat_2d[1], 0.0])              # (3,)

    # Pad covariance to 3x3 (small z variance)
    P_p = np.array([[P_p_2d[0, 0], P_p_2d[0, 1], 0.0],
                    [P_p_2d[1, 0], P_p_2d[1, 1], 0.0],
                    [0.0,          0.0,          1e-4]])

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
        n_mc=20000, seed=seed, n_restarts=1,
        lambda_slew=0.0  # or >0 if you want soft slew constraint
    )

    if u_star is None:
        print("No feasible solution given the slew angle constraints.")
        return

    print("Best objective (−J):", J_star)
    # Optional: compute and print real slew angles
    for i, (alpha_i, beta_i) in enumerate(ang_star):
        # angle between current and optimized pointing vectors
        dot = np.dot(u_curr_agents[i], u_star[i])
        dot = np.clip(dot, -1.0, 1.0)
        slew = np.arccos(dot)

        print(f"\nAgent {i}:")
        print(f"   alpha = {alpha_i:.4f} rad   ({np.rad2deg(alpha_i):.2f} deg)")
        print(f"   beta  = {beta_i:.4f} rad   ({np.rad2deg(beta_i):.2f} deg)")
        print(f"   u*    = {u_star[i]}")
        print(f"   slew  = {slew:.4f} rad   ({np.rad2deg(slew):.2f} deg)")

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
        plt.ylabel("Slew 0 (deg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 2)
        plt.plot(np.rad2deg(slew1))
        plt.ylabel("Slew 1 (deg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 3)
        plt.plot(J_hist)
        plt.xlabel("Logged step")
        plt.ylabel("J_t")
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

    # ----- Coverage shading + geometry plot -----
    fig, ax = plt.subplots(figsize=(8, 6))

    # Background coverage map
    Nx = 500
    Ny = 500
    xg = np.linspace(x_line_min - 5, x_line_max + 5, Nx)
    yg = np.linspace(y_line - 1, y_t_max + 8, Ny)
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

    ax.imshow(
        coverage_img,
        extent=[xg.min(), xg.max(), yg.min(), yg.max()],
        origin='lower',
        cmap=cov_cmap,
        norm=norm,  # <-- key line
        alpha=0.25,
        zorder=-5
    )

    # FOV wedges and agents
    for i, pos in enumerate(p_agents_2d):
        plot_fov_wedge(ax, pos, pointing_angles_opt[i], theta_h,
                       ray_length=10.0, color='tab:blue', alpha=0.12, lw=1.5)
        ax.scatter(pos[0], pos[1], color='tab:blue', s=60)
        ax.text(pos[0], pos[1] - 0.35, f"A{i}", color='tab:blue',
                ha='center', va='top')

    # Target mean + ellipse
    ax.scatter(p_hat_2d[0], p_hat_2d[1], color='tab:red', s=80,
               marker='x', linewidths=2)
    ax.plot(ellipse_pts[:, 0], ellipse_pts[:, 1], color='tab:red', lw=2)
    ax.fill(ellipse_pts[:, 0], ellipse_pts[:, 1],
            color='tab:red', alpha=0.10)

    # Agent line
    ax.plot([x_line_min, x_line_max], [y_line, y_line],
            color='k', lw=1, alpha=0.3)

    # Random sample from uncertainty distribution (within same d_M)
    rng2 = np.random.default_rng(seed + 42)
    sample_pt = sample_from_uncertainty_2d(p_hat_2d, P_p_2d,
                                           d_mahal=d_mahal, rng=rng2)
    ax.scatter(sample_pt[0], sample_pt[1],
               s=60, facecolors='none', edgecolors='green',
               linewidths=2, label='Random sample')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Agents with Optimized Dual-Coverage FOVs (θ/φ parametrization)")

    ax.set_xlim(x_line_min - 5, x_line_max + 5)
    ax.set_ylim(y_line - 1, y_t_max + 8)
    plt.grid(alpha=0.25)
    plt.legend(loc='upper right')

    # Optional: J_t(θ_1, θ_2) surface, like before
    if False:
        fixed_thetas = [0, 0, 17.08]

        TH12_1, TH12_2, J12 = compute_J_grid_thetas_pair(
            p_hat, P_p, p_agents, u_curr_agents,
            theta_h, theta_s_list,
            idx_pair=(0, 1),
            fixed_thetas=fixed_thetas,
            d_M=d_mahal, kappa_sigma=kappa,
            n_mc=20000, n_grid=50, seed=seed
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
        ax3d.set_title(r'$J_t(\theta_1,\theta_2)$ for $\phi_1=\phi_2=0$')

        plt.tight_layout()

        plt.figure(figsize=(6, 5))
        cs = plt.contourf(TH12_1, TH12_2, J12, levels=30)
        plt.colorbar(cs, label=r'$J_t$')
        plt.xlabel(r'$\theta_1$ (deg)')
        plt.ylabel(r'$\theta_2$ (deg)')
        plt.title(r'$J_t$ contour for $\phi_1=\phi_2=0$')
        plt.grid(alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
