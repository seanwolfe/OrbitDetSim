import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# 2D scenario helpers
# ============================================================

def sample_agents_on_line(N, x_min=-5.0, x_max=5.0, y_line=0.0, seed=None):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x_min, x_max, size=N)
    ys = np.full(N, y_line)
    return np.stack([xs, ys], axis=1)

def sample_target_in_front(x_min=-5.0, x_max=5.0, y_min=2.0, y_max=8.0, seed=None):
    rng = np.random.default_rng(seed)
    return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)])

def mahalanobis_ellipse_points(mu2, Sigma2, d_mahal=3.0, num_pts=200):
    eigvals, eigvecs = np.linalg.eigh(Sigma2)
    eigvals = np.maximum(eigvals, 1e-12)
    t = np.linspace(0, 2*np.pi, num_pts)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)
    axes_lengths = d_mahal * np.sqrt(eigvals)
    ellipse_local = np.diag(axes_lengths) @ circle
    ellipse_world = (eigvecs @ ellipse_local).T + mu2
    return ellipse_world

def plot_fov_wedge(ax, agent_pos, pointing_angle, half_angle,
                   ray_length=50.0, color='tab:blue', alpha=0.15, lw=1.5):
    """
    2D infinite-range cone -> wedge with two rays.
    pointing_angle measured from +y axis (front), CCW positive.
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
# 3D dual coverage + analytic-grad optimizer (from earlier)
# ============================================================

def sample_uniform_ball(n_samples, radius=1.0, dim=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = rng.normal(size=(n_samples, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    u = rng.random(n_samples)
    r = radius * (u ** (1.0 / dim))
    return x * r[:, None]

def orthonormal_basis_from_u(u):
    u = u / np.linalg.norm(u)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, u)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = a - np.dot(a, u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def cap_pointing_and_derivs(u_curr, theta, phi):
    u_curr = u_curr / np.linalg.norm(u_curr)
    e1, e2 = orthonormal_basis_from_u(u_curr)

    u_phi = np.cos(phi)*e1 + np.sin(phi)*e2
    u = np.cos(theta)*u_curr + np.sin(theta)*u_phi
    u = u / np.linalg.norm(u)

    du_dtheta = -np.sin(theta)*u_curr + np.cos(theta)*u_phi
    du_dphi   = np.sin(theta)*(-np.sin(phi)*e1 + np.cos(phi)*e2)
    return u, du_dtheta, du_dphi

def unpack_angles(x, M):
    return x[0::2], x[1::2]

def compute_C_and_dC(y, Lp, p_hat, p_agents, u_curr_agents, thetas, phis,
                     cos_theta_h, kappa_sigma):
    M = len(p_agents)
    N = y.shape[0]
    p_samples = (Lp @ y.T).T + p_hat[None, :]

    C      = np.zeros((M, N))
    dC_dth = np.zeros((M, N))
    dC_dph = np.zeros((M, N))

    for i in range(M):
        u_i, du_dth_i, du_dph_i = cap_pointing_and_derivs(
            u_curr_agents[i], thetas[i], phis[i]
        )

        v = p_samples - p_agents[i][None, :]
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        v_unit = v / np.maximum(v_norm, 1e-12)

        cos_ang = v_unit @ u_i
        z = kappa_sigma * (cos_ang - cos_theta_h)

        s = sigmoid(z)
        sp = s * (1.0 - s)

        dcos_dth = v_unit @ du_dth_i
        dcos_dph = v_unit @ du_dph_i

        C[i]      = s
        dC_dth[i] = sp * kappa_sigma * dcos_dth
        dC_dph[i] = sp * kappa_sigma * dcos_dph

    return C, dC_dth, dC_dph

def k2_and_grad_wrt_C(C):
    M, N = C.shape
    eps = 1e-12
    one_minus = 1.0 - C
    prod_all = np.prod(one_minus, axis=0)

    k2 = np.zeros(N)
    gradC = np.zeros((M, N))

    for i in range(M):
        for j in range(i, M):
            denom = one_minus[i] * one_minus[j]
            prod_excl = prod_all / np.maximum(denom, eps)
            T = C[i] * C[j] * prod_excl
            k2 += T

            if i == j:
                gradC[i] += 2.0 * T / np.maximum(C[i], eps)
                for m in range(M):
                    if m != i:
                        gradC[m] += -T / np.maximum(one_minus[m], eps)
            else:
                gradC[i] += T / np.maximum(C[i], eps)
                gradC[j] += T / np.maximum(C[j], eps)
                for m in range(M):
                    if m != i and m != j:
                        gradC[m] += -T / np.maximum(one_minus[m], eps)

    return k2, gradC

def J_and_grad_joint(x, p_hat, P_p, p_agents, u_curr_agents,
                     theta_s_list, theta_h, d_M, kappa_sigma,
                     y_cached):
    M = len(p_agents)
    thetas, phis = unpack_angles(x, M)

    thetas = np.clip(thetas, 0.0, theta_s_list)
    phis   = phis % (2*np.pi)

    Lp = np.linalg.cholesky(P_p)
    cos_theta_h = np.cos(theta_h)

    C, dC_dth, dC_dph = compute_C_and_dC(
        y_cached, Lp, p_hat, p_agents, u_curr_agents,
        thetas, phis, cos_theta_h, kappa_sigma
    )

    k2, gradC = k2_and_grad_wrt_C(C)

    y2 = np.sum(y_cached**2, axis=1)
    w = np.exp(-0.5 * y2)

    V_ball = (4.0/3.0) * np.pi * d_M**3
    const = V_ball / ((2*np.pi)**1.5)

    J = const * np.mean(w * k2)

    grad_theta = np.zeros(M)
    grad_phi   = np.zeros(M)

    for i in range(M):
        grad_theta[i] = const * np.mean(w * gradC[i] * dC_dth[i])
        grad_phi[i]   = const * np.mean(w * gradC[i] * dC_dph[i])

    grad_x = np.zeros_like(x)
    grad_x[0::2] = grad_theta
    grad_x[1::2] = grad_phi

    return J, grad_x

def optimize_pointing_lbfgs_joint_analytic(
    p_hat, P_p, p_agents, u_curr_agents,
    theta_h, theta_s_list,
    d_M=3.0, kappa_sigma=120.0,
    n_mc=20000, seed=0,
    n_restarts=3
):
    rng = np.random.default_rng(seed)
    M = len(p_agents)

    y_cached = sample_uniform_ball(n_mc, radius=d_M, dim=3, rng=rng)

    bounds = []
    for i in range(M):
        bounds.append((0.0, theta_s_list[i]))
        bounds.append((0.0, 2*np.pi))

    best_x = None
    best_J = np.inf

    for _ in range(n_restarts):
        x0 = np.zeros(2*M)
        for i in range(M):
            x0[2*i]   = rng.uniform(0.0, theta_s_list[i])
            x0[2*i+1] = rng.uniform(0.0, 2*np.pi)

        def f(z):
            J, _ = J_and_grad_joint(
                z, p_hat, P_p, p_agents, u_curr_agents,
                theta_s_list, theta_h, d_M, kappa_sigma, y_cached
            )
            return -J

        def g(z):
            _, grad = J_and_grad_joint(
                z, p_hat, P_p, p_agents, u_curr_agents,
                theta_s_list, theta_h, d_M, kappa_sigma, y_cached
            )
            return -grad

        res = minimize(
            f, x0, method="L-BFGS-B",
            jac=g, bounds=bounds,
            options=dict(maxiter=80, ftol=1e-7)
        )

        if res.fun < best_J:
            best_J = res.fun
            best_x = res.x.copy()

    thetas_best, phis_best = unpack_angles(best_x, M)
    u_best = np.zeros_like(u_curr_agents)
    angles_best = []
    for i in range(M):
        u_i, _, _ = cap_pointing_and_derivs(
            u_curr_agents[i], thetas_best[i], phis_best[i]
        )
        u_best[i] = u_i
        angles_best.append((thetas_best[i], phis_best[i]))

    return u_best, angles_best, best_J


# ============================================================
# Your main(), now working
# ============================================================

def main():
    # =======================
    # User parameters
    # =======================
    M = 2

    x_line_min, x_line_max = -6.0, 6.0
    y_line = 0.0

    x_t_min, x_t_max = -4.0, 4.0
    y_t_min, y_t_max = 3.0, 9.0

    theta_h = np.deg2rad(5.0)
    theta_s_list = np.array([np.deg2rad(180.0)] * M)

    rng = np.random.default_rng(0)
    pointing_angles = np.deg2rad(rng.uniform(-25, 25, size=M))

    # 2D covariance
    P_p_2d = np.array([[0.8, 0.3],
                       [0.3, 0.5]])

    d_mahal = 3.0
    seed = 1
    # =======================

    # 2D positions
    p_agents_2d = sample_agents_on_line(M, x_line_min, x_line_max, y_line, seed=seed)
    p_hat_2d = sample_target_in_front(x_t_min, x_t_max, y_t_min, y_t_max, seed=seed+10)

    ellipse_pts = mahalanobis_ellipse_points(p_hat_2d, P_p_2d, d_mahal=d_mahal)

    # ----- Embed into 3D for optimizer -----
    p_agents = np.hstack([p_agents_2d, np.zeros((M,1))])           # (M,3)
    p_hat = np.array([p_hat_2d[0], p_hat_2d[1], 0.0])             # (3,)

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
    u_star, ang_star, J_star = optimize_pointing_lbfgs_joint_analytic(
        p_hat, P_p, p_agents, u_curr_agents,
        theta_h, theta_s_list,
        d_M=d_mahal, kappa_sigma=120.0,
        n_mc=20000, seed=1, n_restarts=4
    )

    print("Best J_t:", J_star)
    for i, (th, ph) in enumerate(ang_star):
        print(f"Agent {i}: theta={np.rad2deg(th):.2f} deg, phi={np.rad2deg(ph):.2f} deg")
        print("   u =", u_star[i])

    # Convert optimized 3D u back to planar pointing angles
    # planar angle from +y axis: angle = atan2(u_x, u_y)
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

    ax.set_xlim(x_line_min-1, x_line_max+1)
    ax.set_ylim(y_line-1, y_t_max+3)

    plt.grid(alpha=0.25)
    plt.show()


if __name__ == "__main__":
    main()
