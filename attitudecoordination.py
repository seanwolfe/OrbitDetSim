import numpy as np
import matplotlib.pyplot as plt
from j_cost import optimize_pointing_lbfgs_joint


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
    # Eigen-decomposition of covariance
    eigvals, eigvecs = np.linalg.eigh(Sigma)  # eigvals sorted ascending
    eigvals = np.maximum(eigvals, 1e-12)      # safety against tiny/negative numeric issues

    # Param angle
    t = np.linspace(0, 2*np.pi, num_pts)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)  # shape (2, num_pts)

    # Scale circle to ellipse: sqrt(eigvals) * d_mahal along principal axes
    axes_lengths = d_mahal * np.sqrt(eigvals)          # (2,)
    ellipse_local = np.diag(axes_lengths) @ circle     # (2, num_pts)
    ellipse_world = (eigvecs @ ellipse_local).T + mu   # (num_pts, 2)

    return ellipse_world


def plot_fov_wedge(ax, agent_pos, pointing_angle, half_angle,
                   ray_length=50.0, color='tab:blue', alpha=0.15, lw=1.5):
    """
    2D infinite-range cone -> wedge with two rays.
    pointing_angle is in radians, measured from +y axis (front) CCW.
    """
    x0, y0 = agent_pos

    # Base boresight direction: +y axis
    # Direction vector for angle theta from +y:
    def dir_from_plus_y(theta):
        return np.array([np.sin(theta), np.cos(theta)])

    left_dir  = dir_from_plus_y(pointing_angle - half_angle)
    right_dir = dir_from_plus_y(pointing_angle + half_angle)

    left_pt  = agent_pos + ray_length * left_dir
    right_pt = agent_pos + ray_length * right_dir

    # Draw rays
    ax.plot([x0, left_pt[0]],  [y0, left_pt[1]],  color=color, lw=lw)
    ax.plot([x0, right_pt[0]], [y0, right_pt[1]], color=color, lw=lw)

    # Fill wedge triangle for visualization
    ax.fill([x0, left_pt[0], right_pt[0]],
            [y0, left_pt[1], right_pt[1]],
            color=color, alpha=alpha)


def main():
    # =======================
    # User parameters
    # =======================
    M = 2

    # Line where agents live
    x_line_min, x_line_max = -6.0, 6.0
    y_line = 0.0

    # Target sampling region (in front of line)
    x_t_min, x_t_max = -4.0, 4.0
    y_t_min, y_t_max = 3.0, 9.0

    # FOV half-angle (deg -> rad)
    theta_h = np.deg2rad(5.0)
    theta_s_list = [np.deg2rad(180.0)] * M

    # Agent pointing angles (control angles) in their body frames
    # Measured from +y axis, CCW positive.
    # Example: random small slews about +y
    rng = np.random.default_rng(0)
    pointing_angles = rng.uniform(-25, 25, size=M)
    print(pointing_angles)
    pointing_angles = np.deg2rad(pointing_angles)

    # Target covariance matrix (example)
    P_p = np.array([[0.8, 0.3],
                             [0.3, 0.5]])

    # Mahalanobis distance for ellipse (e.g., 1, 2, 3 ...)
    d_mahal = 3.0

    seed = 1
    # =======================

    p_agents = sample_agents_on_line(M, x_line_min, x_line_max, y_line, seed=seed)
    p_hat = sample_target_in_front(x_t_min, x_t_max, y_t_min, y_t_max, seed=seed+10)

    # Ellipse points
    ellipse_pts = mahalanobis_ellipse_points(p_hat, P_p, d_mahal=d_mahal)



    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # FOVs + agents
    for i, pos in enumerate(p_agents):
        plot_fov_wedge(ax, pos, pointing_angles[i], theta_h,
                       ray_length=50.0, color='tab:blue', alpha=0.12, lw=1.5)
        ax.scatter(pos[0], pos[1], color='tab:blue', s=60)
        ax.text(pos[0], pos[1]-0.35, f"A{i}", color='tab:blue', ha='center', va='top')

    # Target + ellipse
    ax.scatter(p_hat[0], p_hat[1], color='tab:red', s=80, marker='x', linewidths=2)
    ax.plot(ellipse_pts[:, 0], ellipse_pts[:, 1], color='tab:red', lw=2)
    ax.fill(ellipse_pts[:, 0], ellipse_pts[:, 1], color='tab:red', alpha=0.10)

    # Line of agents
    ax.plot([x_line_min, x_line_max], [y_line, y_line], color='k', lw=1, alpha=0.3)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Agents with Infinite-Range FOVs and Target Uncertainty Ellipse")

    # Nice bounds
    ax.set_xlim(x_line_min-1, x_line_max+1)
    ax.set_ylim(y_line-1, y_t_max+3)

    plt.grid(alpha=0.25)
    plt.show()


if __name__ == "__main__":
    main()
