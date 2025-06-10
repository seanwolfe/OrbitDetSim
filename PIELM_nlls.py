import torch
import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
import spiceypy as spice
import numpy as np
import yaml
import argparse

from sympy.printing.pretty.pretty_symbology import line_width

import utilities as util
from typing import Callable, List, Literal, Tuple, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
from torch.autograd.functional import jacobian
import n_body_integrator as nbody



def solve(true, epochs_nd_norm, observations, obs_incdices, spacecraft_position, collocation_points_jdtdb, hidden_dim, normalization_constant, config):

    # Argument parser to get the config file path
    parser = argparse.ArgumentParser(description="Run the spacecraft simulation")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # === Dummy inputs for illustration ===
    # Dimensions
    q = 3  # output dimension (x, y, z)
    H_size = hidden_dim  # hidden layer size

    # Fake precomputed hidden layer activations and derivatives
    def compute_hidden_activations(z, W, b, activation=torch.tanh):
        """
        :param activation:
        :param z: shape (d, 1)
        :param W: shape (H, 1)
        :param b: shape (H,)
        :return: H, H_prime, H_double_prime
        """
        z_proj = W @ z.T + b[:, None]  # (H, d)
        H = activation(z_proj)
        H_prime = (1 - H ** 2) * W  # (H, d)
        H_double_prime = -2 * H * (1 - H ** 2) * (W ** 2)  # (H, d)
        return H.T, H_prime.T, H_double_prime.T  # shapes: (d, H)
    weights = 2 * torch.rand(H_size, 1) - 1
    bias = 2 * torch.rand(H_size) - 1
    H_matrix, H_dot, H_ddot = compute_hidden_activations(epochs_nd_norm, weights, bias)

    # === PIELM Time normalization constants ===
    # === Constants and scales ===
    L = config['normalization_ratio'] * config['EARTH_RADIUS_KM'] # Length scale in km
    mu_E = config['EARTH_MASS_PARAMETER']
    T = np.sqrt(L ** 3 / mu_E)


    c = normalization_constant  # normalization constant from z-domain
    lambda_phys = 1000 # physics weight (can be tuned)

    # === Residual Function for Least Squares ===
    def residual_function(beta_flat, return_debug=False):
        beta_tensor = beta_flat.view(q, H_size)

        Y_predicted = H_matrix @ beta_tensor.T  # (N, q)
        def ra_dec_observation_residual(Y_pred, Y_obs, spacecraft_pos):
            """
            Computes the MSE between observed and predicted [sin(RA), cos(RA), sin(DEC)].

            :param Y_pred: (N, 3) geocentric non-dim predicted positions [x, y, z] in units of 3 Earth Hill Radii
            :param Y_obs: (N, 3) observations as [sin(RA), cos(RA), sin(DEC)]
            :param spacecraft_pos: (N, 3) spacecraft geo positions (KM) - used to get observer-to-target vector
            :return: residual between predicted and observed angular vectors
            """
            Y_obs = torch.tensor(Y_obs, dtype=torch.float32)

            # Convert predicted positions to km (geocentric)
            Y_pred_km = Y_pred * L  # (N, 3)

            # Get observer positions in km
            spacecraft_pos_km = torch.tensor(spacecraft_pos, dtype=torch.float32)

            # Compute observer-to-target vector
            obs_to_target = Y_pred_km - spacecraft_pos_km  # (N, 3)

            # Convert to RA/DEC angular representation
            x, y, z = obs_to_target[:, 0], obs_to_target[:, 1], obs_to_target[:, 2]
            r = torch.norm(obs_to_target, dim=1) # Avoid division by 0
            r_xy = torch.norm(obs_to_target[:, :2], dim=1)

            sin_ra = y / r_xy
            cos_ra = x / r_xy
            sin_dec = z / r

            Y_pred_ang = torch.stack([sin_ra, cos_ra, sin_dec], dim=1)

            # Compute MSE between predicted and observed [sin RA, cos RA, sin DEC]
            obs_res = (Y_pred_ang - Y_obs).reshape(-1)

            return obs_res
        obs_residual = ra_dec_observation_residual(Y_predicted[obs_incdices], observations, spacecraft_position) # (N * q)

        Y_ddot_predicted = c ** 2 * (H_ddot @ beta_tensor.T)  # (N, q)
        def nbody_physics_residual(Y_pred, Y_ddot_pred, epochs, configuration):
            """
            Y_pred: (N, 3) — nondimensional predicted geocentric positions of asteroid
            Y_ddot_pred: (N, 3) — nondimensional predicted accelerations
            epochs: (N,) — JDTDB times
            configuration: dict with physical constants and body masses
            """

            def get_nbody_positions(epoch, config):
                """
                :param epoch: numpy array of shape (N,) — JDTDB times
                :param config: dictionary containing masses
                :return: positions (N, n, 3), masses (n,)
                """
                bodies = [10, 1, 2, 399, 4, 5, 6, 7, 8, 301]  # SPICE IDs: SUN, MERCURY, ..., MOON

                names = ['SUN', 'MERCURY', 'VENUS', 'EARTH', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'MOON']
                masses = np.array([config[f'{name}_MASS'] for name in names])

                N, n = len(epoch), len(bodies)
                positions = np.zeros((N, n, 3))

                # Convert epochs from JDTDB to ET
                epoch_ets = [spice.unitim(epoch_i, 'JDTDB', 'ET') for epoch_i in epoch]  # (N,)

                for i, et in enumerate(epoch_ets):
                    for j, body in enumerate(bodies):
                        state, _ = spice.spkgeo(targ=body, et=et, ref='J2000', obs=399)  # observer is Earth
                        positions[i, j, :] = state[:3]  # km

                return torch.tensor(positions, dtype=torch.float32), torch.tensor(masses, dtype=torch.float32)

            positions, masses = get_nbody_positions(epochs, configuration)

            # Nondimensionalize positions (geocentric)
            positions_nd = positions / L  # Now unitless

            G_km = (configuration['GRAVITATIONAL_CONSTANT'] / configuration['KM_TO_M'] ** 3)
            mus = G_km * masses
            mus_nd = mus / mu_E

            N, n, _ = positions_nd.shape
            Y_expanded = Y_pred[:, None, :]  # (N, 1, 3)

            r_vecs = positions_nd - Y_expanded  # (N, n, 3)
            r_norms = torch.norm(r_vecs, dim=-1, keepdim=True)  # (N, n, 1)

            accel_terms = mus_nd[None, :, None] * r_vecs / (r_norms ** 3)  # (N, n, 3)

            total_accel = accel_terms.sum(dim=1)  # (N, 3)

            return (Y_ddot_pred - total_accel).reshape(-1)
        physics_residual = nbody_physics_residual(Y_predicted, Y_ddot_predicted, collocation_points_jdtdb, config)


        print("Obs res:", torch.norm(obs_residual).item(), "Phys res:", torch.norm(physics_residual).item())
        print(Y_predicted[obs_incdices][0].cpu().detach() * L)

        if return_debug:
            return {
                "obs_residual": obs_residual,
                "physics_residual": physics_residual,
                "Y_predicted": Y_predicted.detach(),
                "Y_dot_predicted": (c * (H_dot @ beta_tensor.T)).detach()
            }

        return torch.cat([obs_residual, lambda_phys * physics_residual])  # total residual vector
        # return physics_residual  # total residual vector

    # === Autograd version of residual and Jacobian ===
    def residual_np(beta_flat_np):
        beta_flat = torch.tensor(beta_flat_np, dtype=torch.float32, requires_grad=True)
        res = residual_function(beta_flat)
        return res.detach().numpy()

    def jacobian_np(beta_flat_np):
        beta_flat = torch.tensor(beta_flat_np, dtype=torch.float32, requires_grad=True)

        def wrapped(beta):
            return residual_function(beta)

        J = jacobian(wrapped, beta_flat)  # shape: (n_residuals, n_params)
        return J.detach().numpy()

    loss_history = []
    state_history = []
    def optimization_callback(beta_flat_np):
        with torch.no_grad():
            beta_flat = torch.tensor(beta_flat_np, dtype=torch.float32)
            debug = residual_function(beta_flat, return_debug=True)

            obs_loss = torch.norm(debug['obs_residual']).item()
            phys_loss = torch.norm(debug['physics_residual']).item()
            total_loss = obs_loss + lambda_phys * phys_loss

            loss_history.append((total_loss, obs_loss, lambda_phys * phys_loss))

            # Save predicted trajectory (in km)
            Y_predicted_km = debug['Y_predicted'] * L
            Y_dot_predicted_km = debug['Y_dot_predicted'] * L / T
            state_history.append((Y_predicted_km.cpu().numpy(), Y_dot_predicted_km))

    # === Initial Guess ===
    beta0 = np.random.rand(q * H_size)

    # === Solve with SciPy ===
    res = least_squares(
        fun=residual_np,
        x0=beta0,
        jac=jacobian_np,
        verbose=2,
        method='trf',  # or 'trf', depending on structure
        xtol=5e-16,
    )


    # === Final output weights ===
    beta_opt = torch.tensor(res.x, dtype=torch.float32).view(q, H_size)
    final_pos = H_matrix @ beta_opt.T
    final_pos_km = final_pos * L
    final_pos_obs_km = final_pos_km[obs_incdices]

    final_vel = c * H_dot @ beta_opt.T
    final_vel_kms = final_vel / T * L
    final_vel_obs_kms = final_vel_kms[obs_incdices]

    # print(beta_opt)
    print(final_pos_obs_km)
    print(final_vel_obs_kms)

    # Get observer positions in km
    spacecraft_pos_kms = torch.tensor(spacecraft_position, dtype=torch.float32)

    # Compute observer-to-target vector
    obs_to_target = final_pos_obs_km - spacecraft_pos_kms  # (N, 3)

    # Convert to RA/DEC angular representation
    x, y, z = obs_to_target[:, 0], obs_to_target[:, 1], obs_to_target[:, 2]
    r = torch.norm(obs_to_target, dim=1) # Avoid division by 0
    r_xy = torch.norm(obs_to_target[:, :2], dim=1)

    sin_ra = y / r_xy
    cos_ra = x / r_xy
    sin_dec = z / r

    Y_pred_ang = torch.stack([sin_ra, cos_ra, sin_dec], dim=1)
    print(Y_pred_ang)

    # calc epochs
    num_frames = config['number_of_frames']
    asteroid_epoch = collocation_points_jdtdb[obs_incdices][0]
    step = config['time_between_frames'] / config['SECONDS_PER_DAY']
    total_observation_window = num_frames * step  # epoch is in jd

    # Function to get state vectors (position, velocity) in km & km/s
    epoch_et = spice.unitim(asteroid_epoch, 'JDTDB', 'ET')  # initial epoch
    def get_state(body, reference=10):
        state, _ = spice.spkgeo(body, epoch_et, "ECLIPJ2000", reference)
        return np.array(state)
    earth_state = get_state(399)


    asteroid_ini_pos_geo = final_pos_obs_km[0].cpu().detach().numpy()
    asteroid_ini_vel_geo = final_vel_obs_kms[0].cpu().detach().numpy()
    asteroid_state_geo = np.concatenate([asteroid_ini_pos_geo, asteroid_ini_vel_geo])
    asteroid_state_helio = util.eme_to_ecliptic_batch(asteroid_state_geo) + earth_state

    # integrate s/c traj
    asteroid_integrated_states, asteroid_earth_states = nbody.integrate_n_body(asteroid_state_helio,
                                                                               asteroid_epoch,
                                                                               total_observation_window *
                                                                               config['SECONDS_PER_DAY'],
                                                                               config['time_between_frames'],
                                                                               type="ASTEROID")  # integrator takes seconds

    asteroid_int_geo = (asteroid_integrated_states - asteroid_earth_states)
    asteroid_eme = util.ecliptic_to_eme_batch(asteroid_int_geo)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(*final_pos_obs_km.cpu().detach().numpy().T, label='Geo eme Pos')
    ax.plot(*spacecraft_position.T, label='Spacecraft Pos')
    ax.plot(*true.T, label='True')
    # ax.plot(*asteroid_int_geo, label='Integrated', linestyle='--')

    ax.set_xlabel('X [KM]')
    ax.set_ylabel('Y [KM]')
    ax.set_zlabel('Z [KM]')
    ax.legend()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')

    ax2.plot(*final_pos_obs_km.cpu().detach().numpy().T, label='EME Pos')
    ax2.plot(*asteroid_eme[:3, :], label='Integrated', linestyle='--', linewidth=3)

    ax2.set_xlabel('X [KM]')
    ax2.set_ylabel('Y [KM]')
    ax2.set_zlabel('Z [KM]')
    ax2.legend()
    plt.show()


    print(loss_history)


    return


def sample_time_points(
    method: Literal["lhs", "uniform", "gaussian"],
    observation_epochs: np.ndarray,
    delta: float,
    num_points: int,
    mean: float = None,
    std: float = None,
    layer_ratios: List[Tuple[float, float]] = None,  # Only used for lhs and random_uniform
        config=None,
    seed: Union[int, None] = None
) -> np.ndarray:
    """
    Generate time samples using specified strategy, including observation epochs.

    Parameters:
        method: Sampling method to use ("lhs", "random_uniform", or "gaussian").
        observation_epochs: Array of observation times.
        delta: Time to extend before and after observation window.
        num_points: Total number of time samples to draw (including observation_epochs).
        layer_ratios: List of (start_ratio, end_ratio) defining sub-regions in [0, 1] (only for LHS and uniform).
        config: Additional parameters:
            - expansion: float, factor to expand domain beyond obs epochs
            - mean: float (for Gaussian)
            - std: float (for Gaussian)
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray of sampled time points, including observation_epochs.
    """
    if config is None:
        config = {}
    rng = np.random.default_rng(config["seed"])
    t0, tN = observation_epochs[0], observation_epochs[-1]
    domain_start = t0 - delta
    domain_end = tN + delta
    layer_bounds = [(t0 - delta, t0), (t0, tN), (tN, tN + delta)]

    # Number of additional points to sample
    n_obs = len(observation_epochs)
    n_sample = num_points - n_obs
    if n_sample < 0:
        raise ValueError("num_points must be greater than or equal to number of observation_epochs.")

    # Sample additional time points
    if method == "gaussian":
        samples = []
        while len(samples) < n_sample:
            x = rng.normal(loc=mean, scale=std)
            if domain_start <= x <= domain_end:
                samples.append(x)
        additional_samples = np.array(samples)

    else:
        if layer_ratios is None:
            raise ValueError("layer_ratios must be provided for LHS and random_uniform methods.")

        # Convert layer_ratios to relative weights
        ratios = [end - start for start, end in layer_ratios]
        total_ratio = sum(ratios)
        normalized_ratios = [r / total_ratio for r in ratios]

        # Distribute `n_sample` as proportionally as possible across layers
        raw_counts = np.array([r * n_sample for r in normalized_ratios])
        base_counts = np.floor(raw_counts).astype(int)

        # Distribute remaining samples to best approximate the target ratios
        remainder = n_sample - np.sum(base_counts)
        if remainder > 0:
            fractional_parts = raw_counts - base_counts
            top_indices = np.argsort(-fractional_parts)[:remainder]
            for idx in top_indices:
                base_counts[idx] += 1

        points_per_layer = base_counts

        all_samples = []
        for i, (layer_start, layer_end) in enumerate(layer_bounds):
            n = points_per_layer[i]
            if n > 0:
                if method == "lhs":
                    strata = np.linspace(layer_start, layer_end, n + 1)
                    samples = strata[:-1] + rng.uniform(0, 1, size=n) * (strata[1:] - strata[:-1])
                elif method == "uniform":
                    if n == 1:
                        samples = np.array([(layer_start + layer_end) / 2])
                    else:
                        samples = np.linspace(layer_start, layer_end, n, endpoint=False) + (layer_end - layer_start) / (
                                    2 * n)

            else:
                raise ValueError(f"Unsupported method: {method}")

            all_samples.append(samples)

        additional_samples = np.concatenate(all_samples) if all_samples else np.array([])

    # Combine with observation epochs and sort
    combined = np.concatenate([observation_epochs, additional_samples])
    return np.sort(combined)


def epoch_normalization(epoch, z_range, configuration):
    """
    Normalize JDTDB epochs to a specified z_range after non-dimensionalizing.

    Parameters:
        epoch (np.ndarray): Array of JDTDB times (Julian Dates).
        z_range (Tuple[float, float]): Target range (z0, zf) to map the nondimensionalized epochs into.
        configuration (dict): Must contain:
            - 'EARTH_HILL_RADIUS_KM'
            - 'EARTH_MASS'
            - 'GRAVITATIONAL_CONSTANT'
            - 'KM_TO_M'

    Returns:
        normalized_epoch (np.ndarray): Epochs mapped to z_range.
        normalization_constant (float): (zf - z0) / (t_ndim_f - t_ndim_0)
    """

    # === Constants and scales ===
    L = configuration['normalization_ratio'] * configuration['EARTH_RADIUS_KM']  # Length scale in km
    mu_E = configuration['EARTH_MASS_PARAMETER']
    T = np.sqrt(L ** 3 / mu_E)


    # === Convert JDTDB to seconds since first epoch ===
    SECONDS_PER_DAY = 86400.0
    t_seconds = (epoch - epoch[0]) * SECONDS_PER_DAY

    # === Nondimensionalize ===
    t_nondim = t_seconds / T
    t0, tf = t_nondim[0], t_nondim[-1]

    # === Normalize to z_range ===
    z0, zf = z_range
    scale = (zf - z0) / (tf - t0)
    normalized_epoch = z0 + scale * (t_nondim - t0)

    return normalized_epoch, scale









