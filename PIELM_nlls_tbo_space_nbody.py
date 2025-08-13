from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting.static import StaticOrbitPlotter
from poliastro.twobody.propagation import propagate
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import EarthLocation, AltAz, ITRS, GCRS, SkyCoord, Angle
from typing import Callable, List, Literal, Tuple, Union
import torch
from torch.autograd.functional import jacobian
from scipy.optimize import least_squares
import pandas as pd
import spiceypy as spice
import n_body_integrator as nbody
from utilities import eme_to_ecliptic_batch, ecliptic_to_eme_batch
import utilities as util


# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')

####
# generate data
###
def generate_data(config, parameters):


    all_files = util.get_all_files(config['IOD_folder_path'], config['save_format'])
    run_number = parameters['RUN_NUMBER']
    file_path = all_files[run_number]
    print(file_path)
    iod_data = util.read_IOD_data_geo(file_path, config)

    sin_ra_meas = torch.tensor(iod_data['SIN_RA_PHYS'].values, dtype=torch.float32)
    cos_ra_meas = torch.tensor(iod_data['COS_RA_PHYS'].values, dtype=torch.float32)
    sin_dec_meas = torch.tensor(iod_data['SIN_DEC_PHYS'].values, dtype=torch.float32)
    observatory_positions_gcrs = torch.tensor(iod_data.loc[:, ["SC_GEO_X(KM)_PHYS", "SC_GEO_Y(KM)_PHYS", "SC_GEO_Z(KM)_PHYS"]].values,
                                              dtype=torch.float32)
    observation_epochs = [Time(jd, format='jd', scale='tdb') for jd in iod_data['EPOCH(JDTDB)'].values]

    object_positions = iod_data.loc[:, ["GEO_X(KM)", "GEO_Y(KM)", "GEO_Z(KM)"]].values
    object_velocities = iod_data.loc[:, ["GEO_VX(KM/S)", "GEO_VY(KM/S)", "GEO_VZ(KM/S)"]].values

    return [sin_ra_meas, cos_ra_meas, sin_dec_meas], observatory_positions_gcrs, observation_epochs, object_positions, object_velocities


####
# PIELM
####

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
    L =  configuration['EARTH_RADIUS_KM']  * configuration['normalization_ratio'] # Length scale in km
    mu_E = configuration['EARTH_MASS_PARAMETER']
    T = np.sqrt(L ** 3 / mu_E)

    epoch = Time(epoch)
    t_seconds = (epoch - epoch[0]).sec

    # === Nondimensionalize ===
    t_nondim = t_seconds / T
    t0, tf = t_nondim[0], t_nondim[-1]

    # === Normalize to z_range ===
    z0, zf = z_range
    scale = (zf - z0) / (tf - t0)
    normalized_epoch = z0 + scale * (t_nondim - t0)

    return normalized_epoch, scale



def run(data, config, parameters):

    # get the collocation points
    colloc_points = sample_time_points(parameters['SAMPLING_METHOD'], data[2], parameters['TIME_DELTA'],
                                       parameters['TOTAL_POINTS'], layer_ratios=parameters['LAYER_RATIOS'], config=config)

    # normalize epochs (inputs)
    epochs_nd_norm, c = epoch_normalization(colloc_points, parameters['INPUT_RANGE'], config)
    epochs_nd_norm_reshaped_tensor = torch.tensor(epochs_nd_norm, dtype=torch.float32).unsqueeze(1)  # as a 2D tensor

    # get the indices where observations are
    obs_mask = np.isin(colloc_points, data[2])
    obs_indices = np.where(obs_mask)[0]

    return solve(epochs_nd_norm_reshaped_tensor, data[0], obs_indices, data[1], colloc_points, c, config, parameters)


def solve(epochs_nd_norm_reshaped_tensor, y_obs, obs_indices, observer_positions, colloc_epochs, c, configuration, parameters):

    L = configuration['EARTH_RADIUS_KM'] * configuration['normalization_ratio'] # Length scale in km
    mu_E = configuration['EARTH_MASS_PARAMETER']
    T = np.sqrt(L ** 3 / mu_E)
    lambda_phys = parameters['PHYSICS_WEIGHT']
    q = 3
    # === Dummy inputs for illustration ===
    # Dimensions
    H_size = parameters['HIDDEN_DIMENSION']  # hidden layer size
    # Fake precomputed hidden layer activations and derivatives
    W = 2 * torch.rand(H_size, 1) - 1
    b = 2 * torch.rand(H_size) - 1

    data_losses = []
    physics_losses = []
    positions = []
    velocities = []
    total_its = [0]

    def compute_hidden_activations():
        """
        :param activation:
        :param z: shape (d, 1)
        :param W: shape (H, 1)
        :param b: shape (H,)
        :return: H, H_prime, H_double_prime
        """
        z_proj = W @ epochs_nd_norm_reshaped_tensor.T + b[:, None]  # (H, d)
        H = np.tanh(z_proj)
        H_prime = (1 - H ** 2) * W  # (H, d)
        H_double_prime = -2 * H * (1 - H ** 2) * (W ** 2)  # (H, d)
        return H.T, H_prime.T, H_double_prime.T  # shapes: (d, H)
    H_matrix, H_dot, H_ddot = compute_hidden_activations()

    # === Residual Function for Least Squares ===
    def residual_function(beta_flat):
        beta_tensor = beta_flat.view(q, H_size)
        Y_predicted = H_matrix @ beta_tensor.T  # (N, q)
        Y_dot_predicted = c * H_dot @ beta_tensor.T

        Y_pred_obs = Y_predicted[obs_indices]
        Y_pred_obs_km = Y_pred_obs * L
        Y_dot_pred_obs = Y_dot_predicted[obs_indices] * L / T

        positions.append(Y_pred_obs_km.detach().cpu().detach())
        velocities.append(Y_dot_pred_obs.detach().cpu().detach())
        total_its[0] += 1

        def ra_dec_observation_residual():
            """
            Computes the MSE between observed and predicted [sin(RA), cos(RA), sin(DEC)].

            :param Y_pred: (N, 3) geocentric non-dim predicted positions [x, y, z] in units of 3 Earth Hill Radii
            :param Y_obs: (N, 3) observations as [sin(RA), cos(RA), sin(DEC)]
            :param spacecraft_pos: (N, 3) spacecraft geo positions (KM) - used to get observer-to-target vector
            :return: residual between predicted and observed angular vectors
            """

            def generate_ra_dec_from_gcrs():
                """
                Inputs:
                    Y_pred_gcrs_km: (N, 3) predicted satellite positions (GCRS, km)
                    observation_times: (N,) in seconds (e.g., since t0) — torch.Tensor
                    observatory_choices: list of dicts per time with 'lat', 'lon', 'elev' (in meters)

                Returns:
                    ra_pred: (N,) RA in degrees
                    dec_pred: (N,) DEC in degrees
                """

                topocentric_los_vec = Y_pred_obs_km - observer_positions  # (N, 3)

                # Normalize
                los_unit_topo = topocentric_los_vec / torch.norm(topocentric_los_vec, dim=1, keepdim=True)

                # Convert to RA/DEC
                x_topo, y_topo, z_topo = los_unit_topo[:, 0], los_unit_topo[:, 1], los_unit_topo[:, 2]
                ra_topo = torch.arctan2(y_topo, x_topo)
                dec_topo = torch.arcsin(z_topo)
                # Normalize RA to [0, 360)
                ra_topo = ra_topo % (2 * torch.pi)
                return ra_topo, dec_topo

            ra_predicted, dec_predicted = generate_ra_dec_from_gcrs()
            sin_ra_predicted, cos_ra_predicted, sin_dec_predicted = (torch.sin(ra_predicted),
                                                                     torch.cos(ra_predicted),
                                                                     torch.sin(dec_predicted))

            Y_pred_ang = torch.stack([sin_ra_predicted, cos_ra_predicted, sin_dec_predicted], dim=1)
            Y_obs = torch.stack(y_obs, dim=1)

            # Compute MSE between predicted and observed [sin RA, cos RA, sin DEC]
            obs_res = (Y_pred_ang - Y_obs).reshape(-1)

            return obs_res

        obs_residual = ra_dec_observation_residual()  # (N * q)

        Y_ddot_predicted = c ** 2 * (H_ddot @ beta_tensor.T)  # (N, q)

        def nbody_physics_residual(Y_preds, Y_ddot_preds, epochs, configuration):
            """
            Y_pred: (N, 3) — nondimensional predicted geocentric positions of asteroid
            Y_ddot_pred: (N, 3) — nondimensional predicted accelerations
            epochs: (N,) — JDTDB times
            configuration: dict with physical constants and body masses
            """

            Y_pred = Y_preds
            Y_ddot_pred = Y_ddot_preds

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

            epochs_val = [t.tdb.jd for t in epochs]
            positions, masses = get_nbody_positions(epochs_val, configuration)

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
        physics_residual = nbody_physics_residual(Y_predicted, Y_ddot_predicted, colloc_epochs, configuration)


        data_losses.append((torch.mean(obs_residual ** 2)).item())
        physics_losses.append(lambda_phys * (torch.mean(physics_residual ** 2)).item())

        print("Obs res:", torch.norm(obs_residual).item(), "Phys res:", torch.norm(physics_residual).item())
        print(Y_predicted[obs_indices][0].cpu().detach() * L)

        return torch.cat([obs_residual, lambda_phys * physics_residual])  # total residual vector

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

    beta0 = np.random.rand(q * H_size)

    # === Solve with SciPy ===
    res2 = least_squares(
        fun=residual_np,
        x0=beta0,
        jac=jacobian_np,
        verbose=2,
        method='lm',  # or 'trf', depending on structure
        xtol=parameters['X_TOLERANCE']
    )

    epochss = np.arange(total_its[0])
    data = {"TRAINING_EPOCH": epochss, "DATA_LOSS": data_losses, "PHYSICS_LOSS": physics_losses}

    return pd.DataFrame(data), positions, velocities
