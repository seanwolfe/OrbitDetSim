import numpy as np
import matplotlib.pyplot as plt
from j_cost import sample_uniform_ball,J_t_dual_coverage

# ---------- assume these are imported from your module ----------
# sample_uniform_ball
# cap_pointing_and_derivs (or u_from_cap)
# J_t_dual_coverage (or objective_joint)
# ---------------------------------------------------------------

def cap_basis(u_curr):
    """
    Build orthonormal basis {u_curr, e1, e2} for spherical-cap parametrization.
    """
    u_curr = u_curr / np.linalg.norm(u_curr)

    # pick a vector not parallel to u_curr
    if abs(u_curr[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(u_curr, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u_curr, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def cap_pointing_and_derivs(u_curr, theta, phi):
    """
    Maps spherical-cap parameters (theta, phi) into a pointing vector.
    Also returns analytic derivatives wrt theta and phi.
    """
    u_curr = u_curr / np.linalg.norm(u_curr)
    e1, e2 = cap_basis(u_curr)

    # unit vector inside tangent plane
    u_phi = np.cos(phi) * e1 + np.sin(phi) * e2

    # pointing vector
    u = np.cos(theta) * u_curr + np.sin(theta) * u_phi

    # analytic derivatives:
    du_dtheta = -np.sin(theta) * u_curr + np.cos(theta) * u_phi
    du_dphi = np.sin(theta) * (-np.sin(phi) * e1 + np.cos(phi) * e2)

    return u, du_dtheta, du_dphi


def plot_J_theta_surface(
    p_hat, P_p, p_agents, u_curr_agents,
    theta_h, theta_s_list,
    d_M=3.0, kappa_sigma=120.0,
    n_mc=20000, n_grid=60,
    phi1_fixed=0.0, phi2_fixed=0.0,
    seed=0
):
    """
    Plots J(theta1, theta2) for M=2, with phi1,phi2 fixed.
    theta_i in [0, theta_s_i].
    """

    assert len(p_agents) == 2, "This plotter is for M=2 only."

    rng = np.random.default_rng(seed)

    # Cache MC samples once for smoother plotting
    y_cached = sample_uniform_ball(n_mc, radius=d_M, dim=3, rng=rng)

    # Helper to evaluate J with fixed phis
    def J_for_thetas(th1, th2):
        # Build pointing vectors using cap model
        u1, _, _ = cap_pointing_and_derivs(u_curr_agents[0], th1, phi1_fixed)
        u2, _, _ = cap_pointing_and_derivs(u_curr_agents[1], th2, phi2_fixed)
        u_agents = np.vstack([u1, u2])

        # Evaluate cost using cached y
        return J_t_dual_coverage(
            p_hat, P_p, p_agents, u_agents,
            theta_h, d_M=d_M, kappa_sigma=kappa_sigma,
            n_mc=y_cached.shape[0], y_samples_cached=y_cached
        )

    # Theta grids
    th1_vals = np.linspace(0, theta_s_list[0], n_grid)
    th2_vals = np.linspace(0, theta_s_list[1], n_grid)

    TH1, TH2 = np.meshgrid(th1_vals, th2_vals, indexing="xy")
    J_grid = np.zeros_like(TH1)

    # Evaluate J on grid
    for i in range(n_grid):
        for j in range(n_grid):
            J_grid[j, i] = J_for_thetas(TH1[j, i], TH2[j, i])

    # ----- Plotting -----
    fig = plt.figure(figsize=(12, 5))

    # 3D surface
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        np.rad2deg(TH1), np.rad2deg(TH2), J_grid,
        rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.9
    )
    ax1.set_xlabel(r"$\theta_1$ [deg]")
    ax1.set_ylabel(r"$\theta_2$ [deg]")
    ax1.set_zlabel(r"$J$")
    ax1.set_title("Dual-coverage objective surface (M=2)")

    # 2D contours (contour-ish view)
    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(np.rad2deg(TH1), np.rad2deg(TH2), J_grid, levels=25)
    fig.colorbar(cs, ax=ax2, label="J")
    ax2.set_xlabel(r"$\theta_1$ [deg]")
    ax2.set_ylabel(r"$\theta_2$ [deg]")
    ax2.set_title("Dual-coverage objective contours (M=2)")

    plt.tight_layout()
    plt.show()

    return TH1, TH2, J_grid


# ------------------ example usage ------------------
if __name__ == "__main__":
    # Example 2D scenario embedded in 3D (same as your main)
    M = 2
    rng = np.random.default_rng(1)

    p_agents_2d = np.array([[-3.0, 0.0],
                            [ 3.0, 0.0]])
    p_agents = np.hstack([p_agents_2d, np.zeros((M,1))])

    p_hat_2d = np.array([0.5, 5.0])
    p_hat = np.array([p_hat_2d[0], p_hat_2d[1], 0.0])

    P_p_2d = np.array([[0.8, 0.3],
                       [0.3, 0.5]])
    P_p = np.array([[P_p_2d[0,0], P_p_2d[0,1], 0.0],
                    [P_p_2d[1,0], P_p_2d[1,1], 0.0],
                    [0.0,         0.0,         1e-4]])

    # Current pointing vectors (random)
    pointing_angles = np.deg2rad(rng.uniform(-20, 20, size=M))
    u_curr_agents = np.stack([np.sin(pointing_angles),
                              np.cos(pointing_angles),
                              np.zeros(M)], axis=1)

    theta_h = np.deg2rad(5.0)
    theta_s_list = [np.deg2rad(360.0), np.deg2rad(360.0)]

    plot_J_theta_surface(
        p_hat, P_p, p_agents, u_curr_agents,
        theta_h, theta_s_list,
        d_M=3.0, kappa_sigma=120.0,
        n_mc=15000, n_grid=50,
        phi1_fixed=0.0, phi2_fixed=0.0,
        seed=2
    )
