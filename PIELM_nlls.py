import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
import spiceypy as spice
import numpy as np
import yaml
import argparse
import utilities as util
from typing import Callable, List, Literal, Tuple, Union
import matplotlib.pyplot as plt


# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')

class ELM(nn.Module):
    def __init__(self, hidden_dim, q=3, activation=torch.tanh, c_normalization=1.0):
        """
        :param input_dim: Dimensionality of each input vector (e.g., 1 for time).
        :param hidden_dim: Number of hidden neurons (N_star).
        :param q: Number of output dimensions (e.g., 3 for 3D position).
        :param activation: Activation function, e.g., tanh.
        """
        super().__init__()
        input_dim= 1
        self.input_weights = nn.Parameter(torch.randn(hidden_dim, input_dim), requires_grad=False)  # (H x 1)
        self.bias = nn.Parameter(torch.randn(hidden_dim), requires_grad=False)  # (H,)
        self.activation = activation
        self.output_weights = nn.Parameter(torch.randn(q, hidden_dim))  # (q x H)
        self.c_normalization = c_normalization

    def forward(self, z):
        # z should be of shape (d, input_dim), typically (d, 1)
        # Transpose z to (input_dim, d) so matmul (H x 1) @ (1 x d) => (H x d)
        H = self.activation(self.input_weights @ z.T + self.bias[:, None])  # (H x d)
        Y = self.output_weights @ H  # (q x H) @ (H x d) => (q x d)
        return Y.T  # Return shape (d x q)

    def forward_with_derivatives(self, z):
        """
        z: shape (d, 1)
        Returns: y, y', y'' each of shape (d, q)
        """
        W = self.input_weights  # (H x 1)
        b = self.bias[:, None]  # (H x 1)
        z = W @ z.T + b  # (H x d)
        H = torch.tanh(z)  # (H x d)

        H_prime = (1 - H ** 2) * W  # (H x d)
        H_double_prime = -2 * H * (1 - H ** 2) * (W ** 2)  # (H x d)

        Y = self.output_weights @ H  # (q x N)
        Y_dot = self.c_normalization * self.output_weights @ H_prime  # (q x d)
        Y_dot_dot = self.c_normalization ** 2 * self.output_weights @ H_double_prime  # (q x d)

        return Y.T, Y_dot.T, Y_dot_dot.T


def nbody_physics_loss(Y_pred, Y_ddot_pred, epochs, configuration):
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
                state, _ = spice.spkgeo(targ=body, et=et, ref='ECLIPJ2000', obs=399)  # observer is Earth
                positions[i, j, :] = state[:3]  # km

        return torch.tensor(positions, dtype=torch.float32), torch.tensor(masses, dtype=torch.float32)

    positions, masses = get_nbody_positions(epochs, configuration)

    # === Constants and scales ===
    RH_km = configuration['EARTH_HILL_RADIUS_KM']
    L = 3 * RH_km  # Length scale in km
    M = configuration['EARTH_MASS']  # Mass scale in kg
    T = np.sqrt(L ** 3 / (configuration['GRAVITATIONAL_CONSTANT'] / configuration['KM_TO_M'] ** 3 * M))

    # Y_pred /= L
    # Y_ddot_pred /= (L / T ** 2)

    # Nondimensionalize positions (geocentric)
    positions_nd = positions / L  # Now unitless

    # Convert masses to nondimensional (mass / M)
    masses_nd = masses / M  # (n,)

    # The graviational constant is non-dimensionlized
    G_nd = (configuration['GRAVITATIONAL_CONSTANT'] / configuration['KM_TO_M'] ** 3) *  T ** 2 * M / L ** 3

    N, n, _ = positions_nd.shape
    Y_expanded = Y_pred[:, None, :]  # (N, 1, 3)

    r_vecs = positions_nd - Y_expanded  # (N, n, 3)
    r_norms = torch.norm(r_vecs, dim=-1, keepdim=True)  # (N, n, 1)

    accel_terms = G_nd * masses_nd[None, :, None] * r_vecs / (r_norms ** 3 + 1e-9)  # (N, n, 3)  # in km

    total_accel = accel_terms.sum(dim=1)  # (N, 3)

    return torch.mean((Y_ddot_pred - total_accel) ** 2)


def ra_dec_observation_loss(Y_pred, Y_obs, spacecraft_pos, epochs, configuration):
    """
    Computes the MSE between observed and predicted [sin(RA), cos(RA), sin(DEC)].

    :param Y_pred: (N, 3) geocentric non-dim predicted positions [x, y, z] in units of 3 Earth Hill Radii
    :param Y_obs: (N, 3) observations as [sin(RA), cos(RA), sin(DEC)]
    :param spacecraft_pos: (N, 3) spacecraft heliocentric positions (AU) - used to get observer-to-target vector
    :param epochs: (N,) JDTDB times
    :param configuration: dict with keys:
        - 'EHILL_KM'
        - 'AU_KM'
    :return: scalar MSE loss between predicted and observed angular vectors
    """
    Y_obs = torch.tensor(Y_obs, dtype=torch.float32)

    EHILL_KM = configuration['EARTH_HILL_RADIUS_KM']
    AU_KM = configuration['AU_TO_M'] / configuration['KM_TO_M']

    # Convert predicted positions to km (geocentric)
    Y_pred_km = Y_pred * (3 * EHILL_KM)  # (N, 3)

    # Get observer positions in km (AU -> km)
    spacecraft_pos_km = torch.tensor(spacecraft_pos, dtype=torch.float32) * AU_KM

    # Get Earth's position at each epoch in heliocentric km
    earth_positions = torch.zeros_like(Y_pred_km)
    for i, jd in enumerate(epochs):
        et = spice.unitim(jd, 'JDTDB', 'ET')
        state, _ = spice.spkgeo(targ=399, et=et, ref='ECLIPJ2000', obs=10)  # Earth wrt Sun
        earth_positions[i, :] = torch.tensor(state[:3], dtype=torch.float32)


    # Compute observer position in geocentric km
    observer_pos_geo = spacecraft_pos_km - earth_positions  # (N, 3)

    # Compute observer-to-target vector
    obs_to_target = Y_pred_km - observer_pos_geo  # (N, 3)

    # Convert to RA/DEC angular representation
    x, y, z = obs_to_target[:, 0], obs_to_target[:, 1], obs_to_target[:, 2]
    r = torch.norm(obs_to_target, dim=1) + 1e-12  # Avoid division by 0
    dec = torch.asin(z / r)
    ra = torch.atan2(y, x)

    sin_ra = torch.sin(ra)
    cos_ra = torch.cos(ra)
    sin_dec = torch.sin(dec)

    Y_pred_ang = torch.stack([sin_ra, cos_ra, sin_dec], dim=1)

    # Compute MSE between predicted and observed [sin RA, cos RA, sin DEC]
    loss = torch.mean((Y_pred_ang - Y_obs) ** 2)

    return loss


def sample_time_points(
    method: Literal["lhs", "uniform", "gaussian"],
    observation_epochs: np.ndarray,
    delta: float,
    num_points: int,
    mean: float = None,
    std: float = None,
    layer_ratios: List[Tuple[float, float]] = None,  # Only used for lhs and random_uniform
    config: dict = {},
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
    RH_km = configuration['EARTH_HILL_RADIUS_KM']
    L = 3 * RH_km  # km
    M = configuration['EARTH_MASS']  # kg
    G = configuration['GRAVITATIONAL_CONSTANT']  # m^3 / kg / s^2
    KM_TO_M = configuration['KM_TO_M']

    # Convert G to km^3 / kg / s^2
    G_km3 = G / KM_TO_M**3

    # Time scale (in seconds)
    T_scale = np.sqrt(L**3 / (G_km3 * M))

    # === Convert JDTDB to seconds since first epoch ===
    SECONDS_PER_DAY = 86400.0
    t_seconds = (epoch - epoch[0]) * SECONDS_PER_DAY

    # === Nondimensionalize ===
    t_nondim = t_seconds / T_scale
    t0, tf = t_nondim[0], t_nondim[-1]

    # === Normalize to z_range ===
    z0, zf = z_range
    scale = (zf - z0) / (tf - t0)
    normalized_epoch = z0 + scale * (t_nondim - t0)

    return normalized_epoch, scale


def train(model, z_data, y_obs, y_obs_index, spacecraft_pos, obs_epochs_jdtdb, colloc_epochs_jdtdb, configuration, epochs=100000, lr=1e-2, lambda_phys=0.000100):
    optimizer = torch.optim.Adam([model.output_weights], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass with derivatives
        Y_pred, Y_dot_pred, Y_ddot_pred = model.forward_with_derivatives(z_data)

        Y_pred_obs = Y_pred[y_obs_index]

        # Compute losses
        data_loss = ra_dec_observation_loss(Y_pred_obs, y_obs, spacecraft_pos, obs_epochs_jdtdb, configuration)
        phys_loss = nbody_physics_loss(Y_pred, Y_ddot_pred, colloc_epochs_jdtdb, configuration)

        # Total loss
        loss = data_loss + lambda_phys * phys_loss
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Data Loss = {data_loss.item():.4e}, Physics Loss = {phys_loss.item():.4e}")

    L = 3 * configuration['EARTH_HILL_RADIUS_KM'] * configuration['KM_TO_M'] / configuration['AU_TO_M']  # au
    print(Y_pred[y_obs_index] * L)
    # print(Y_dot_pred)

# Argument parser to get the config file path
parser = argparse.ArgumentParser(description="Run the spacecraft simulation")
parser.add_argument('--config', type=str, required=True, help="Path to the config file")
args = parser.parse_args()

# Load the config file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# elm class test
# elm = ELM(hidden_dim=50, q=3)
# z = torch.linspace(0, 1, 100).unsqueeze(1)  # (100 x 1)
# y_pred = elm(z)  # (100 x 3)

# physics loss test
# traj = pd.read_csv('asteroid_trajectory_jdtdb.csv')
# y_pred = traj.loc[:, ['x', 'y', 'z']].values
# y_dot_dot_pred = traj.loc[:, ['ax', 'ay', 'az']].values
# epochss = traj.loc[:, 'jdtdb'].values
# loss = nbody_physics_loss(y_pred, y_dot_dot_pred, epochss, config)
# print(loss)

# observation loss test
# read data
# file_path = 'minimoon-NESC000000B3_sc-1_index-10432_spacecraft_2_runs_2_run_2_part_1.csv'
# iod_data = util.read_IOD_data(file_path, config)
#
# yp_helio = iod_data.loc[:, ["HELIO_X(AU)", "HELIO_Y(AU)", "HELIO_Z(AU)"]].values
# e = iod_data.loc[:, 'EPOCH(JDTDB)']
#
# EHILL_KM = config['EARTH_HILL_RADIUS_KM']
# AU_KM = config['AU_TO_M'] / config['KM_TO_M']

# Get observer positions in km (AU -> km)
# yp_helio_km = torch.tensor(yp_helio, dtype=torch.float32) * AU_KM

# Get Earth's position at each epoch in heliocentric km
# earth_positions = np.zeros_like(yp_helio_km)
# for i, jd in enumerate(e):
#     et = spice.unitim(jd, 'JDTDB', 'ET')
#     state, _ = spice.spkgeo(targ=399, et=et, ref='ECLIPJ2000', obs=10)  # Earth wrt Sun
#     earth_positions[i, :] = state[:3]
# earth_positions_km = torch.tensor(earth_positions, dtype=torch.float32)

# Compute observer position in geocentric km
# yp_geo_km = yp_helio_km - earth_positions_km  # (N, 3)

# Convert predicted positions to km (geocentric)
# yp = yp_geo_km / (3 * EHILL_KM)  # (N, 3)

# sc = iod_data.loc[:, ["SC_HELIO_X(AU)", "SC_HELIO_Y(AU)", "SC_HELIO_Z(AU)"]].values
# yo = iod_data.loc[:, ['SIN_RA', 'COS_RA', 'SIN_DEC']].values
# print(ra_dec_observation_loss(yp, yo, sc, e, config))

##########
# collocation points test
##########
# obs_e = np.array([2454965.50836787, 2454966.50836787, 2454967.50836787, 2454968.50836787, 2454969.50836787, 2454970.50836787,
#           2454971.50836787, 2454972.50836787, 2454973.50836787, 2454974.50836787])
# delta = 10
# num_points = 20
# layer_ratios = [(0., 1/5), (1/5, 4/5), (4/5, 1.)]
# mean = obs_e[0] + (obs_e[-1] - obs_e[0]) / 2
# std = (obs_e[-1] - obs_e[0]) * 4
# nbins = num_points

# gaussian
# colloc_points = sample_time_points("gaussian", obs_e, delta, num_points, mean, std, layer_ratios=layer_ratios, config=config)
# print(colloc_points.shape)
# print(colloc_points[0])
# print(colloc_points[-1])
# for obs in obs_e:
#     if obs in colloc_points:
#         print("True")
#     else:
#         print("False")
# plt.figure()
# plt.hist(colloc_points)
# plt.hist(obs_e)
# plt.show()


# lhs
# colloc_points = sample_time_points("lhs", obs_e, delta, num_points, layer_ratios=layer_ratios, config=config)
# print(colloc_points.shape)
# print(colloc_points[0])
# print(colloc_points[-1])
# print(len(colloc_points[colloc_points < obs_e[0]]))
# print(len(colloc_points[colloc_points > obs_e[-1]]))
# for obs in obs_e:
#     if obs in colloc_points:
#         print("True")
#     else:
#         print("False")
# plt.figure()
# plt.hist(colloc_points, edgecolor='black', bins=nbins)
# plt.hist(obs_e, edgecolor='black')
# plt.show()

# uniform
# colloc_points = sample_time_points("uniform", obs_e, delta, num_points, layer_ratios=layer_ratios, config=config)
# print(colloc_points.shape)
# print(colloc_points[0])
# print(colloc_points[-1])
# print(len(colloc_points[colloc_points < obs_e[0]]))
# print(len(colloc_points[colloc_points > obs_e[-1]]))
# for obs in obs_e:
#     if obs in colloc_points:
#         print("True")
#     else:
#         print("False")
# plt.figure()
# plt.hist(colloc_points, edgecolor='black', bins=nbins)
# plt.hist(obs_e, edgecolor='black')
# plt.show()

############
# time input non-dim and normalizatoin test, with colloc points
###########
# epochs_nd_norm, c = epoch_normalization(colloc_points, (0, 1), config)
#
# print(epochs_nd_norm)
# print(c)


###############
# full-test
##############