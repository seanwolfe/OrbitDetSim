import numpy as np
import matplotlib.pyplot as plt
import time
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

    one_minus_C = 1.0 - C
    prod_all = np.prod(one_minus_C, axis=0)

    k2 = np.zeros(N)
    for i in range(M):
        for j in range(i, M):
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


def finite_diff_grad(f, x, eps=1e-4):
    """Central finite-difference gradient."""
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xm = x.copy()
        xp[i] += eps; xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2*eps)
    return -g  # gradient of -J


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

    Returns:
      u_best (M,3), angles_best list[(theta,phi)], best_objective

    If infeasible (slew too small to even reach the near edge of the ellipsoid
    for any spacecraft), returns (None, None, 0.0).
    """
    rng = np.random.default_rng(seed)
    M = len(p_agents)

    # --- NEW: estimate θ_min and θ_max^{(ell)} for each spacecraft ---
    theta_min_ell, theta_max_ell = estimate_theta_bounds_from_ellipsoid(
        p_hat, P_p, p_agents, u_curr_agents, d_M,
        n_shell=400, seed=seed+999
    )

    # Effective upper bound: min(θ_max^{ell}, θ_s)
    theta_upper = np.minimum(theta_max_ell, theta_s_list)
    theta_lower = theta_min_ell.copy()  # "near edge" of ellipse

    # Ensure lower bound non-negative (just in case)
    theta_lower = np.maximum(theta_lower, 0.0)

    # Feasibility check: if θ_s < θ_min^{ell} => can't reach the ellipse at all
    infeasible_mask = theta_upper < theta_lower
    if np.any(infeasible_mask):
        print("Infeasible: for at least one spacecraft, slew limit is smaller "
              "than the angle required to reach the near edge of the uncertainty ellipsoid.")
        return None, None, 0.0

    # cache MC points to smooth objective
    y_cached = make_cached_y(P_p, d_M, n_mc, seed=seed+123)

    # bounds per agent in (θ, φ)
    bounds = []
    for i in range(M):
        bounds.append((theta_lower[i], theta_upper[i]))    # theta_i
        bounds.append((0.0, 2 * np.pi))                    # phi_i

    best = None
    best_J = np.inf

    for r in range(n_restarts):
        # random init inside θ-bounds
        x0 = np.zeros(2*M)
        for i in range(M):
            x0[2*i]   = rng.uniform(theta_lower[i], theta_upper[i])
            x0[2*i+1] = rng.uniform(0.0, 2*np.pi)

        f = lambda z: objective_joint(
            z, p_hat, P_p, p_agents, u_curr_agents,
            theta_lower, theta_upper,
            theta_h, d_M, kappa_sigma, y_cached
        )

        if SCIPY_OK:
            res = minimize(
                f, x0, method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=60, ftol=1e-10, disp=False)
            )
            x_star = res.x
            J_star = res.fun
        else:
            # fallback: crude joint random + gradient descent
            x_star = x0.copy()
            J_star = f(x_star)
            lr = 0.2
            for _ in range(40):
                g = finite_diff_grad(f, x_star, eps=2e-4)
                x_star -= lr * g
                # project to bounds
                for i in range(M):
                    x_star[2*i]   = np.clip(x_star[2*i], theta_lower[i], theta_upper[i])
                    x_star[2*i+1] = x_star[2*i+1] % (2*np.pi)
                J_new = f(x_star)
                if J_new < J_star:
                    J_star = J_new
                    lr *= 0.95
                else:
                    lr *= 0.5

        if J_star < best_J:
            best_J = J_star
            best = x_star.copy()

    # unpack best solution
    thetas_best, phis_best = unpack_angles(best, M)
    u_best = angles_to_pointings(best, p_agents, u_curr_agents, theta_lower, theta_upper, M)
    angles_best = [(thetas_best[i], phis_best[i]) for i in range(M)]

    return u_best, angles_best, best_J


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


# ============================================================
# Your main(), now working
# ============================================================

def main():
    # =======================
    # User parameters
    # =======================
    M = 3

    x_line_min, x_line_max = -1.0, 1.0
    y_line = 0.0

    x_t_min, x_t_max = -2.0, 2.0
    y_t_min, y_t_max = 1.0, 5.0

    theta_h = np.deg2rad(2.5)
    theta_s_list = np.array([np.deg2rad(90.0)] * M)

    rng = np.random.default_rng(0)
    pointing_angles = np.deg2rad(rng.uniform(-1, 1, size=M))

    # 2D covariance
    P_p_2d = np.array([[0.25, 0.1],
                       [-0.1, 0.5]])

    d_mahal = 3.0

    # seed = int(time.time())
    # print(seed)
    seed = 1764104794
    # =======================

    # 2D positions
    p_agents_2d = sample_agents_on_line(M, x_line_min, x_line_max, y_line, seed=seed)
    p_hat_2d = sample_target_in_front(x_t_min, x_t_max, y_t_min, y_t_max, seed=seed+10)

    ellipse_pts = mahalanobis_ellipse_points(p_hat_2d, P_p_2d, d_mahal=d_mahal)

    # ----- Embed into 3D for optimizer -----
    p_agents = np.hstack([p_agents_2d, np.zeros((M,1))])           # (M,3)
    p_hat = np.array([p_hat_2d[0], p_hat_2d[1], 0.0])              # (3,)

    # Pad covariance to 3x3 (tiny z variance)
    P_p = np.array([[P_p_2d[0,0], P_p_2d[0,1], 0.0],
                    [P_p_2d[1,0], P_p_2d[1,1], 0.0],
                    [0.0,         0.0,         1e-4]])

    # Current pointing vectors from planar angles (measured from +y)
    # u = [sin(angle), cos(angle), 0]
    u_curr_agents = np.stack([np.sin(pointing_angles),
                              np.cos(pointing_angles),
                              np.zeros(M)], axis=1)

    # ----- Optimize jointly -----
    u_star, ang_star, J_star = optimize_pointing_lbfgs_joint(
        p_hat, P_p, p_agents, u_curr_agents,
        theta_h, theta_s_list,
        d_M=d_mahal, kappa_sigma=100.0,
        n_mc=20000, seed=seed, n_restarts=100
    )

    if u_star is None:
        print("No feasible solution given the slew angle constraints.")
        return

    print("Best objective (−J):", J_star)
    for i, (th, ph) in enumerate(ang_star):
        print(f"Agent {i}: theta={np.rad2deg(th):.2f} deg, phi={np.rad2deg(ph):.2f} deg")
        print("   u =", u_star[i])

    # Convert optimized 3D u back to planar pointing angles
    pointing_angles_opt = np.arctan2(u_star[:,0], u_star[:,1])

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, pos in enumerate(p_agents_2d):
        plot_fov_wedge(ax, pos, pointing_angles_opt[i], theta_h,
                       ray_length=50.0, color='tab:blue', alpha=0.12, lw=1.5)
        ax.scatter(pos[0], pos[1], color='tab:blue', s=60)
        ax.text(pos[0], pos[1]-0.35, f"A{i}", color='tab:blue', ha='center', va='top')

    ax.scatter(p_hat_2d[0], p_hat_2d[1], color='tab:red', s=80, marker='x', linewidths=2)
    ax.plot(ellipse_pts[:, 0], ellipse_pts[:, 1], color='tab:red', lw=2)
    ax.fill(ellipse_pts[:, 0], ellipse_pts[:, 1], color='tab:red', alpha=0.10)

    ax.plot([x_line_min, x_line_max], [y_line, y_line], color='k', lw=1, alpha=0.3)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Agents with Optimized Dual-Coverage FOVs")

    ax.set_xlim(x_line_min-5, x_line_max+5)
    ax.set_ylim(y_line-1, y_t_max+8)

    plt.grid(alpha=0.25)
    plt.show()


if __name__ == "__main__":
    main()
