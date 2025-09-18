import yaml
import os
import pandas as pd
from Asteroid import Asteroid
from Formation import Formation
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import ast
import spiceypy as spice
import argparse
from matplotlib.collections import LineCollection
import n_body_integrator as nbody
from astropy.time import Time
from scipy.integrate import odeint

# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')
pd.options.mode.chained_assignment = None


import numpy as np
import pandas as pd

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
            *(iod_data.loc[:, ["SC_GEO_X(KM)_PHYS", "SC_GEO_Y(KM)_PHYS"]].values * config['AU_TO_M'] / config['KM_TO_M']).T,
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
        ax7.plot(*(asteroid_cr3bp_position[:, :2] * config['AU_TO_M'] / config['KM_TO_M']).T, label='CR3BP Integrated', linestyle='--', linewidth=3, zorder=5)
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
        asteroid_state_helio = eme_to_ecliptic_batch(asteroid_state_geo) + earth_state

        # integrate s/c traj
        asteroid_integrated_states, asteroid_earth_states = nbody.integrate_n_body(asteroid_state_helio,
                                                                                   asteroid_epoch,
                                                                                   total_observation_window *
                                                                                   config['SECONDS_PER_DAY'],
                                                                                   config['time_between_frames'],
                                                                                   type="ASTEROID")  # integrator takes seconds

        asteroid_int_geo = (asteroid_integrated_states - asteroid_earth_states)
        asteroid_eme = ecliptic_to_eme_batch(asteroid_int_geo)
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
        pos_rmse = rmse_df.loc[:, ['IOD_X_NLLS', 'IOD_Y_NLLS', 'IOD_Z_NLLS']].values * config['AU_TO_M'] / config['KM_TO_M']
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
                                                                                      "Earth z (Helio)", "Earth vx (Helio)",
                                                                                      "Earth vy (Helio)",
                                                                                      "Earth vz (Helio)"]].values)
    moon_pos[:, :3] *= configs['AU_TO_M'] / configs['KM_TO_M']
    moon_pos[:, 3:] *= (configs['AU_TO_M'] / configs['KM_TO_M'] / configs['SECONDS_PER_DAY'])


    asteroid_pos_eme = ecliptic_to_eme_batch(asteroid_pos.T).T
    moon_pos_eme = ecliptic_to_eme_batch(moon_pos.T).T
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
            ]].values * (configs['AU_TO_M'] / configs['KM_TO_M']) # Reshape to (6, 1)

            # Call the function with correctly shaped inputs
            geo_boresight = sun_earth_corotating_to_geo_eclip_single(boresight_vec, earth_state)
            geo_eme_boresight = ecliptic_to_eme_single(geo_boresight)

            fov_corners = plot_fov_projection_geo(geo_eme_boresight[:3], spacecraft_pos, asteroid_pos_eme[traj_index, :],
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
    ax2.scatter(sc_eme_states_physically_sound[0, 0], sc_eme_states_physically_sound[1, 0], sc_eme_states_physically_sound[2, 0], color=colors[-3], s=50,
                label='Physically sound start', zorder=18)  # s/c integration trajectory and initial position
    ax2.plot(sc_eme_states_physically_sound[0, :], sc_eme_states_physically_sound[1, :], sc_eme_states_physically_sound[2, :], color=colors[-3], linewidth=5,
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


def helio_eclip_to_sun_earth_corotating_batch_full(states, earth_states):
    """
    Converts a batch of position and velocity state vectors from heliocentric ECLIPJ2000
    to the Earth-centered Sun-Earth co-rotating frame with fixed ecliptic north.

    Parameters:
    - states: (M, 6, N) array of states in heliocentric ECLIPJ2000, for M objects and N timesteps
    - earth_states: (6, N) array of Earth state vectors in the same frame at each timestep

    Returns:
    - states_corotating: (M, 6, N) array of transformed states in the SECR frame
    """

    _, N = states.shape

    h_r_E = earth_states[:3, :].T  # (N, 3)
    h_v_E = earth_states[3:, :].T  # (N, 3)

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angles = np.arctan2(-h_r_E[:, 1], -h_r_E[:, 0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angles = np.cos(-angles)
    sin_angles = np.sin(-angles)

    # Construct rotation matrices (shape: Nx3x3), Z axis is fixed along ecliptic north
    rotation_matrices = np.zeros((N, 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # Z remains unchanged (ecliptic north)

    # Angular velocity vector assuming uniform circular motion in ecliptic plane
    h_omega_mag = np.linalg.norm(np.cross(h_r_E, h_v_E), axis=1) / (np.linalg.norm(h_r_E, axis=1) ** 2)  # (N,)
    h_omega = np.zeros((N, 3))
    h_omega[:, 2] = h_omega_mag  # Only z-component for ecliptic plane rotation

    states_corotating = np.zeros_like(states)

    h_r_O = states[:3, :].T  # (N, 3)
    h_v_O = states[3:, :].T  # (N, 3)

    h_rel_r = h_r_O - h_r_E  # position relative to Earth (in inertial)
    h_rel_v = h_v_O - h_v_E  # velocity relative to Earth (in inertial)

    E_r_o_prime = np.einsum('nij,nj->ni', rotation_matrices, h_rel_r)  # now co-rotating frame
    v_rel_rot = np.einsum('nij,nj->ni', rotation_matrices, h_rel_v)
    E_omega = np.einsum('nij,nj->ni', rotation_matrices, h_omega)
    v_rot = np.cross(E_omega, E_r_o_prime)  # correct: using rotated position
    E_v_o_prime = v_rel_rot - v_rot  # total velocity in rotating frame

    states_corotating[:3, :] = E_r_o_prime.T
    states_corotating[3:, :] = E_v_o_prime.T

    return states_corotating


def sun_earth_corotating_to_geo_eclip_batch_full(states_corotating, earth_states):
    """
    Converts a batch of state vectors from the Sun-Earth co-rotating (SECR) frame
    to geocentric ECLIPJ2000 coordinates.

    Parameters:
    - states_corotating: (6, N) array in SECR frame; first 3 rows = position, last 3 = velocity
    - earth_states: (6, N) array in heliocentric ECLIPJ2000; used only for rotation angle

    Returns:
    - states_geocentric: (6, N) array in geocentric ECLIPJ2000
    """


    _, N = states_corotating.shape


    # Earth's heliocentric position (used only for rotation)
    h_r_E = earth_states[:3, :].T  # (N, 3)
    h_v_E = earth_states[3:, :].T  # (N, 3)

    # Co-rotating frame state
    E_r_o_prime = states_corotating[:3, :].T  # (N, 3)
    E_v_o_prime = states_corotating[3:, :].T  # (N, 3)

    # Compute rotation angles from Earth-Sun vector (negate for SECR to inertial)
    angles = np.arctan2(-h_r_E[:, 1], -h_r_E[:, 0])  # Shape: (N,)


    # Rotation matrices: from SECR to ECLIPJ2000
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    rotation_matrices = np.zeros((N, 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # z-axis (ecliptic north) unchanged

    # Rotate position to ECLIPJ2000 (geocentric)
    geo_r_o = np.einsum('nij,nj->ni', rotation_matrices, E_r_o_prime)  # (N, 3)

    # Angular velocity vector (magnitude from Earth's motion)
    h_omega_mag = np.linalg.norm(np.cross(h_r_E, h_v_E), axis=1) / (np.linalg.norm(h_r_E, axis=1) ** 2)  # (N,)
    h_omega = np.zeros((N, 3))
    h_omega[:, 2] = h_omega_mag  # z-axis angular velocity

    # Rotate angular velocity to SECR frame
    E_omega = np.einsum('nij,nj->ni', rotation_matrices, h_omega)

    # Add Coriolis term to get inertial velocity
    v_rot = np.cross(E_omega, E_r_o_prime)  # (N, 3)
    v_rel_rot = E_v_o_prime + v_rot  # (N, 3)
    geo_v_o = np.einsum('nij,nj->ni', rotation_matrices, v_rel_rot)  # (N, 3)

    # Assemble full state vector
    states_geocentric = np.zeros_like(states_corotating)
    states_geocentric[:3, :] = geo_r_o.T
    states_geocentric[3:, :] = geo_v_o.T

    return states_geocentric


def sun_earth_corotating_to_geo_eclip_single(pos_corot, earth_state_helio):
    """
    Convert a single position from Sun-Earth co-rotating frame to geocentric ECLIPJ2000.

    Parameters:
    - pos_corot: np.array shape (3,), position in Sun-Earth co-rotating frame
    - earth_state_helio: np.array shape (6,), Earth's heliocentric state vector [x,y,z,vx,vy,vz] in ECLIPJ2000

    Returns:
    - pos_geo_eclip: np.array shape (3,), position in geocentric ECLIPJ2000 frame
    """

    # Earth's heliocentric position
    h_r_E = earth_state_helio[:3]

    # Compute rotation angle: angle of Earth relative to Sun in XY plane (negated for SECR to inertial)
    angle = np.arctan2(-h_r_E[1], -h_r_E[0])

    # Rotation matrix about Z-axis by "angle"
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])

    # Rotate position vector from co-rotating to inertial ECLIPJ2000 frame
    pos_inertial = R @ pos_corot

    return pos_inertial


def ecliptic_to_eme_batch(state_vectors_ecliptic):
    """
    Transforms a batch of full state vectors from Ecliptic J2000 to EME J2000.

    Parameters:
    - state_vectors_ecliptic (numpy array): 6xN array representing N state vectors
      in Ecliptic J2000 (rows: [x, y, z, vx, vy, vz]).

    Returns:
    - numpy array: 6xN array representing N state vectors in EME J2000.
    """

    # Obliquity of the ecliptic at J2000 (in degrees)
    epsilon = 23.439281
    epsilon_rad = np.radians(epsilon)

    # Rotation matrix about the x-axis (−epsilon for Ecliptic to EME)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(-epsilon_rad), np.sin(-epsilon_rad)],
        [0, -np.sin(-epsilon_rad), np.cos(-epsilon_rad)]
    ])

    # Separate position and velocity (each 3xN)
    pos = state_vectors_ecliptic[0:3, :]
    vel = state_vectors_ecliptic[3:6, :]

    # Apply rotation
    pos_eme = R @ pos
    vel_eme = R @ vel

    # Stack back into 6xN
    return np.vstack((pos_eme, vel_eme))


def ecliptic_to_eme_single(state_vectors_ecliptic):
    """
    Transforms a batch of full state vectors from Ecliptic J2000 to EME J2000.

    Parameters:
    - state_vectors_ecliptic (numpy array): 6xN array representing N state vectors
      in Ecliptic J2000 (rows: [x, y, z, vx, vy, vz]).

    Returns:
    - numpy array: 6xN array representing N state vectors in EME J2000.
    """

    # Obliquity of the ecliptic at J2000 (in degrees)
    epsilon = 23.439281
    epsilon_rad = np.radians(epsilon)

    # Rotation matrix about the x-axis (−epsilon for Ecliptic to EME)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(-epsilon_rad), np.sin(-epsilon_rad)],
        [0, -np.sin(-epsilon_rad), np.cos(-epsilon_rad)]
    ])

    # Separate position and velocity (each 3xN)
    pos = state_vectors_ecliptic[0:3]
    vel = state_vectors_ecliptic[3:6]

    # Apply rotation
    pos_eme = R @ pos

    return pos_eme


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
        geo_eclip_state = eme_to_ecliptic_batch(geo_eme_state)
        scs_helio.append(geo_eclip_state)
        closest_indices.append(closest_position_index)
        sc_epochs.append(sc_time.strftime("%Y-%m-%d %H:%M:%S"))

    detected_pop.loc[:, 'detecting_sc_lpf_orbit_index'] = closest_indices
    detected_pop.loc[:, 'sc_epoch'] = sc_epochs
    detected_pop.loc[:,
    ['GEO_ECLIP_X_(km)', 'GEO_ECLIP_Y_(km)', 'GEO_ECLIP_Z_(km)', 'GEO_ECLIP_Vx_(km/s)', 'GEO_ECLIP_Vy_(km/s)',
     'GEO_ECLIP_Vz_(km/s)']] = np.array(scs_helio)

    return detected_pop


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


def eme_to_ecliptic_batch(state_vectors_eme):
    """
    Transforms a batch of full state vectors from EME J2000 to Ecliptic J2000.

    Parameters:
    - state_vectors_eme (numpy array): 6 x N array representing N state vectors
      in EME J2000 (each row: [x, y, z, vx, vy, vz]).

    Returns:
    - numpy array: Nx6 array representing N state vectors in Ecliptic J2000.
    """
    # Obliquity of the ecliptic at J2000 (in degrees)
    epsilon = 23.439281  # Mean obliquity of the ecliptic at J2000 epoch
    epsilon_rad = np.radians(epsilon)

    # Rotation matrix about the x-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(epsilon_rad), np.sin(epsilon_rad)],
        [0, -np.sin(epsilon_rad), np.cos(epsilon_rad)]
    ])

    # Split into position and velocity
    positions = state_vectors_eme[0:3]
    velocities = state_vectors_eme[3:6]

    # Rotate both
    pos_ecliptic = rotation_matrix @ positions
    vel_ecliptic = velocities @ rotation_matrix.T

    # Concatenate position and velocity back
    state_vectors_ecliptic = np.hstack((pos_ecliptic, vel_ecliptic))

    return state_vectors_ecliptic


def eclip_to_sun_earth_corotating_batch_n_body_integrator_output(states, earth_states):
    """
    Converts a batch of position and velocity state vectors from heliocentric ECLIPJ2000
    to the Earth-centered Sun-Earth co-rotating frame (X toward Sun, Z along orbital angular momentum).

    Parameters:
    - states: (M, 6, N) array of states in heliocentric ECLIPJ2000, for M objects and N timesteps
    - earth_states: (6, N) array of Earth state vectors in the same frame at each timestep

    Returns:
    - states_corotating: (M, 6, N) array of transformed states in the SECR frame
    """

    earth_positions = earth_states[:3, :].T

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angles = np.arctan2(earth_positions[:, 1], earth_positions[:, 0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angles = np.cos(-angles)
    sin_angles = np.sin(-angles)

    # Construct rotation matrices (shape: Nx3x3)
    rotation_matrices = np.zeros((len(angles), 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # No rotation in the Z direction

    object_i_pos = states[:3, :].T + earth_positions

    rotation_matrices = np.asarray(rotation_matrices, dtype=np.float64)
    relative_positions = np.asarray(object_i_pos, dtype=np.float64)

    # Apply the rotation to transform positions
    position_corotating = np.einsum("nij,nj->ni", rotation_matrices, relative_positions)

    return position_corotating


def eclip_to_sun_earth_corotating_batch(minimoon_df):
    """
    Converts a batch of positions and velocities from the heliocentric ECLIPJ2000 frame
    to the Sun-Earth co-rotating frame.

    Parameters:
    - positions_eclip (numpy array): Nx3 array of positions in ECLIPJ2000 (AU).
    - et_times (numpy array): N-element array of ephemeris times.

    Returns:
    - positions_corotating (numpy array): Nx3 array of positions in Sun-Earth co-rotating frame (AU).
    """

    moon_pos = minimoon_df.loc[:, ["Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)"]].to_numpy()
    earth_positions = minimoon_df.loc[:, ["Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)"]].to_numpy()

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angles = np.arctan2(earth_positions[:, 1], earth_positions[:, 0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angles = np.cos(-angles)
    sin_angles = np.sin(-angles)

    # Construct rotation matrices (shape: Nx3x3)
    rotation_matrices = np.zeros((len(angles), 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # No rotation in the Z direction

    # Apply the rotation to transform positions
    positions_corotating = np.einsum("nij,nj->ni", rotation_matrices, moon_pos - earth_positions)

    old_file_convention = positions_corotating.copy()
    old_file_convention[:, :2] *= -1

    minimoon_df[['Moon Synodic x', 'Moon Synodic y', 'Moon Synodic z']] = old_file_convention
    file_path = '/media/aeromec/Seagate Desktop Drive/minimoon_files_oorb/' + str(
        minimoon_df['Object id'].iloc[0]) + '.csv'
    minimoon_df.to_csv(file_path, sep=' ', header=True, index=False)

    return positions_corotating


def eclip_to_sun_earth_corotating_batch_full_original_asteroid(states, earth_states):
    """
    Converts a batch of positions and velocities from the heliocentric ECLIPJ2000 frame
    to the Sun-Earth co-rotating frame.

    Parameters:
    - positions_eclip (numpy array): Nx3 array of positions in ECLIPJ2000 (AU).
    - et_times (numpy array): N-element array of ephemeris times.

    Returns:
    - positions_corotating (numpy array): Nx3 array of positions in Sun-Earth co-rotating frame (AU).
    """

    h_r_E = earth_states[:3, :].T  # (N, 3)
    h_v_E = earth_states[3:, :].T  # (N, 3)
    h_r_o = states[:3, :].T
    h_v_o = states[3:, :].T

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angles = np.arctan2(h_r_E[:, 1], h_r_E[:, 0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angles = np.cos(-angles)
    sin_angles = np.sin(-angles)

    # Construct rotation matrices (shape: Nx3x3)
    rotation_matrices = np.zeros((len(angles), 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # No rotation in the Z direction

    # Apply the rotation to transform positions
    positions_corotating = np.einsum("nij,nj->ni", rotation_matrices, h_r_o - h_r_E)

    return positions_corotating.T


def sun_earth_corotating_to_helio_eclip_batch_full(states_corotating, earth_states):
    """
    Converts a batch of position and velocity state vectors from the Earth-centered
    Sun-Earth co-rotating frame (SECR) to heliocentric ECLIPJ2000, accounting for the Coriolis term.

    Parameters:
    - states_corotating: (6, N) array of states in the SECR frame, where M is the number of objects
      and N is the number of timesteps. The first 3 rows represent positions, and the last 3 rows represent velocities.
    - earth_states: (6, N) array of Earth state vectors in the heliocentric ECLIPJ2000 frame at each timestep.
      The first 3 rows represent Earth's position, and the last 3 rows represent Earth's velocity.

    Returns:
    - states_heliocentric: (6, N) array of transformed states in heliocentric ECLIPJ2000 frame.
      The first 3 rows represent positions, and the last 3 rows represent velocities.
    """

    _, N = states_corotating.shape

    # Extract Earth's position and velocity from heliocentric ECLIPJ2000
    h_r_E = earth_states[:3, :].T  # (N, 3)
    h_v_E = earth_states[3:, :].T  # (N, 3)
    E_r_o_prime = states_corotating[:3, :].T
    E_v_o_prime = states_corotating[3:, :].T

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angles = np.arctan2(-h_r_E[:, 1], -h_r_E[:, 0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Construct rotation matrices (shape: Nx3x3), Z axis is fixed along ecliptic north
    rotation_matrices = np.zeros((N, 3, 3))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1  # Z remains unchanged (ecliptic north)

    states_heliocentric = np.zeros_like(states_corotating)

    h_rel_o = np.einsum('nij,nj->ni', rotation_matrices, E_r_o_prime)  # now co-rotating frame
    h_r_o = h_rel_o + h_r_E

    # Angular velocity vector assuming uniform circular motion in ecliptic plane
    h_omega_mag = np.linalg.norm(np.cross(h_r_E, h_v_E), axis=1) / (np.linalg.norm(h_r_E, axis=1) ** 2)  # (N,)
    h_omega = np.zeros((N, 3))
    h_omega[:, 2] = h_omega_mag  # Only z-component for ecliptic plane rotation

    E_omega = np.einsum('nij,nj->ni', rotation_matrices.transpose(0, 2, 1), h_omega)
    v_rot = np.cross(E_omega, E_r_o_prime)  # correct: using rotated position
    v_rel_rot = E_v_o_prime + v_rot
    h_rel_v = np.einsum('nij,nj->ni', rotation_matrices, v_rel_rot)
    h_v_o = h_rel_v + h_v_E

    states_heliocentric[:3, :] = h_r_o.T
    states_heliocentric[3:, :] = h_v_o.T

    return states_heliocentric


def sun_earth_corotating_to_helio_eclip_single(state_corotating, earth_state):
    """
    Converts a batch of position and velocity state vectors from the Earth-centered
    Sun-Earth co-rotating frame (SECR) to heliocentric ECLIPJ2000, accounting for the Coriolis term.

    Parameters:
    - states_corotating: (1, N) array of states in the SECR frame, where M is the number of objects
      and N is the number of timesteps. The first 3 rows represent positions, and the last 3 rows represent velocities.
    - earth_states: (1, N) array of Earth state vectors in the heliocentric ECLIPJ2000 frame at each timestep.
      The first 3 rows represent Earth's position, and the last 3 rows represent Earth's velocity.

    Returns:
    - states_heliocentric: (1, N) array of transformed states in heliocentric ECLIPJ2000 frame.
      The first 3 rows represent positions, and the last 3 rows represent velocities.
    """

    # Extract Earth's position and velocity from heliocentric ECLIPJ2000
    h_r_E = earth_state[:3] # ( 3)
    h_v_E = earth_state[3:]  # (3)
    E_r_o_prime = state_corotating[:3]
    E_v_o_prime = state_corotating[3:]

    # Compute Earth's orbital angle (angle in the ecliptic plane)
    angle = np.arctan2(-h_r_E[1], -h_r_E[0])  # Shape: (N,)

    # Compute cosines and sines of rotation angles
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Construct rotation matrices (shape: Nx3x3), Z axis is fixed along ecliptic north
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, 0] = cos_angle
    rotation_matrix[0, 1] = -sin_angle
    rotation_matrix[1, 0] = sin_angle
    rotation_matrix[1, 1] = cos_angle
    rotation_matrix[2, 2] = 1  # Z remains unchanged (ecliptic north)

    state_heliocentric = np.zeros_like(state_corotating)

    h_rel_o = rotation_matrix @ E_r_o_prime  # now co-rotating frame
    h_r_o = h_rel_o + h_r_E

    # Angular velocity vector assuming uniform circular motion in ecliptic plane
    h_omega_mag = np.linalg.norm(np.cross(h_r_E, h_v_E)) / (np.linalg.norm(h_r_E) ** 2)  # (N,)
    h_omega = np.zeros(3,)
    h_omega[2] = h_omega_mag  # Only z-component for ecliptic plane rotation

    E_omega = rotation_matrix.T @ h_omega
    v_rot = np.cross(E_omega, E_r_o_prime)  # correct: using rotated position
    v_rel_rot = E_v_o_prime + v_rot
    h_rel_v = rotation_matrix @ v_rel_rot
    h_v_o = h_rel_v + h_v_E

    state_heliocentric[:3] = h_r_o
    state_heliocentric[3:] = h_v_o

    return state_heliocentric


def helio_eclip_to_geo_eme_batch(states_helio_eclip, earth_helio_eclip):

    # convert from helio eclip to geo eclip
    states_geo_eclip = states_helio_eclip - earth_helio_eclip
    return ecliptic_to_eme_batch(states_geo_eclip)



def helio_eclip_from_geo_eme(eme_vectors, earth_helio_state):
    ###########################
    # convert geo eme to geo elcip
    ###########################

    # Obliquity of the ecliptic at J2000 (in degrees)
    epsilon = 23.439281  # Mean obliquity of the ecliptic at J2000 epoch

    # Convert epsilon to radians
    epsilon_rad = np.radians(epsilon)

    # Rotation matrix for transformation about the x-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(epsilon_rad), np.sin(epsilon_rad)],
        [0, -np.sin(epsilon_rad), np.cos(epsilon_rad)]
    ])

    # Apply the rotation to each vector using matrix multiplication
    ecliptic_positions = np.dot(eme_vectors[:3], rotation_matrix.T)
    ecliptic_velocities = np.dot(eme_vectors[3:], rotation_matrix.T)

    ##############################
    # convert geo eclip to helio
    ##########################
    helio_eclip_position = ecliptic_positions + earth_helio_state[:3]
    helio_eclip_velocities = ecliptic_velocities + earth_helio_state[3:]

    return np.hstack((helio_eclip_position, helio_eclip_velocities))


def ecliptic_to_eme_single_posvel(state_vectors_ecliptic):
    """
    Transforms a batch of full state vectors from Ecliptic J2000 to EME J2000.

    Parameters:
    - state_vectors_ecliptic (numpy array): 6xN array representing N state vectors
      in Ecliptic J2000 (rows: [x, y, z, vx, vy, vz]).

    Returns:
    - numpy array: 6xN array representing N state vectors in EME J2000.
    """

    # Obliquity of the ecliptic at J2000 (in degrees)
    epsilon = 23.439281
    epsilon_rad = np.radians(epsilon)

    # Rotation matrix about the x-axis (−epsilon for Ecliptic to EME)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(-epsilon_rad), np.sin(-epsilon_rad)],
        [0, -np.sin(-epsilon_rad), np.cos(-epsilon_rad)]
    ])

    # Separate position and velocity (each 3xN)
    pos = state_vectors_ecliptic[0:3]
    vel = state_vectors_ecliptic[3:6]

    # Apply rotation
    pos_eme = R @ pos
    vel_eme = R @ vel

    # Stack back into 6×N
    state_vectors_eme = np.stack((pos_eme, vel_eme)).reshape(-1)
    return state_vectors_eme
