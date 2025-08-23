from astropy import units as u
from astropy.time import Time, TimeDelta
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
from scipy.optimize import basinhopping
import pandas as pd
import spiceypy as spice
import n_body_integrator as nbody
from utilities import eme_to_ecliptic_batch, ecliptic_to_eme_batch
import utilities as util
import os

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
    observer_positions = torch.tensor(
        iod_data.loc[:, ["SC_GEO_X(KM)_PHYS", "SC_GEO_Y(KM)_PHYS", "SC_GEO_Z(KM)_PHYS"]].values,
        dtype=torch.float32)
    observer_velocities = torch.tensor(
        iod_data.loc[:, ["SC_GEO_VX(KM/S)_PHYS", "SC_GEO_VY(KM/S)_PHYS", "SC_GEO_VZ(KM/S)_PHYS"]].values,
        dtype=torch.float32)
    observation_epochs = [Time(jd, format='jd', scale='tdb') for jd in iod_data['EPOCH(JDTDB)'].values]

    positions = iod_data.loc[:, ["GEO_X(KM)", "GEO_Y(KM)", "GEO_Z(KM)"]].values
    velocities = iod_data.loc[:, ["GEO_VX(KM/S)", "GEO_VY(KM/S)", "GEO_VZ(KM/S)"]].values

    # Reconstruct RA and DEC in radians
    ra_meas = torch.atan2(sin_ra_meas, cos_ra_meas)
    ra = ra_meas % (2 * np.pi)  # Ensure RA in [0, 2π)
    dec = torch.asin(sin_dec_meas)  # DEC in [-π/2, π/2]

    if config['ADD_NOISE'] == 1:
        def add_noise(ra_rad, dec_rad, config):
            """
            Adds Gaussian noise to RA and DEC in radians using PyTorch.

            Args:
                ra_rad (torch.Tensor): Right Ascension in radians.
                dec_rad (torch.Tensor): Declination in radians.
                config (dict): Must contain:
                    - 'sigma_ra': float, noise stddev in mas
                    - 'sigma_dec': float, noise stddev in mas
                    - 'sigma_pointing': float, pointing error in mas
                    - 'MAS_TO_DEGREE': float, conversion factor (1e3 * 3600 = 3.6e6)

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Noisy RA and DEC in radians.
            """
            # Total noise in mas
            sigma_ra_mas = torch.sqrt(torch.tensor(config['sigma_ra'] ** 2 + config['sigma_pointing'] ** 2))
            sigma_dec_mas = torch.sqrt(torch.tensor(config['sigma_dec'] ** 2 + config['sigma_pointing'] ** 2))

            # Convert to degrees
            sigma_ra_deg = sigma_ra_mas / config['MAS_TO_DEGREE']
            sigma_dec_deg = sigma_dec_mas / config['MAS_TO_DEGREE']

            # Convert to radians
            sigma_ra_rad = torch.deg2rad(sigma_ra_deg)
            sigma_dec_rad = torch.deg2rad(sigma_dec_deg)

            # Generate noise
            ra_noise = torch.normal(mean=0.0, std=sigma_ra_rad, size=ra_rad.shape)
            dec_noise = torch.normal(mean=0.0, std=sigma_dec_rad, size=dec_rad.shape)

            # Apply noise
            ra_noisy = (ra_rad + ra_noise) % (2 * torch.pi)
            dec_noisy = torch.clamp(dec_rad + dec_noise, min=-torch.pi / 2, max=torch.pi / 2)

            return ra_noisy, dec_noisy, sigma_ra_deg, sigma_dec_deg
        ra_m, dec_m, sigma_ra_deg, sigma_dec_deg = add_noise(ra, dec, config)

    else:
        ra_m, dec_m = ra.clone(), dec.clone()
        sigma_ra_deg = 0.
        sigma_dec_deg = 0.

    sin_ra_meas, cos_ra_meas, sin_dec_meas, cos_dec_meas = torch.sin(ra_m), torch.cos(ra_m), torch.sin(dec_m), torch.cos(dec_m)

    return ([sin_ra_meas, cos_ra_meas, sin_dec_meas, cos_dec_meas], torch.tensor(observer_positions, dtype=torch.float32),
            observation_epochs, positions, velocities, ra, dec, sigma_ra_deg, sigma_dec_deg,  file_path)


####
# PIELM
####
def sample_time_points(
        method: Literal["lhs", "uniform", "gaussian"],
        observation_epochs: np.ndarray,
        delta: float,
        num_points: int,
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

    mean = t0 + (tN - t0) / 2
    std = (tN - t0) / config['gaussian_std_scale']

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
    L =  configuration['AU_TO_M'] / configuration['KM_TO_M']# Length scale in km
    G_km = (configuration['GRAVITATIONAL_CONSTANT'] / configuration['KM_TO_M'] ** 3)
    sys_mass = configuration['SUN_MASS'] + configuration['MOON_MASS'] + configuration['EARTH_MASS']
    T = np.sqrt(L ** 3 / (G_km * sys_mass))

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
    L = configuration['AU_TO_M'] / configuration['KM_TO_M']  # Length scale in km
    G_km = (configuration['GRAVITATIONAL_CONSTANT'] / configuration['KM_TO_M'] ** 3)
    sys_mass = configuration['SUN_MASS'] + configuration['MOON_MASS'] + configuration['EARTH_MASS']
    T = np.sqrt(L ** 3 / (G_km * sys_mass))
    lambda_phys = parameters['PHYSICS_WEIGHT']
    q = 3
    # === Dummy inputs for illustration ===
    # Dimensions
    H_size = parameters['HIDDEN_DIMENSION']  # hidden layer size
    # Fake precomputed hidden layer activations and derivatives
    W = (2 * torch.rand(H_size, 1) - 1) * configuration['WEIGHT_SCALE_FACTOR']
    b = (2 * torch.rand(H_size) - 1) * configuration['WEIGHT_SCALE_FACTOR']

    data_losses = []
    physics_losses = []
    range_losses = []
    positions = []
    velocities = []
    total_its = [0]
    global_its = [0]
    epsilon = [0]
    global_positions = []
    nlls_start = [0]
    first = [0]

    def compute_hidden_activations(input_z):
        """
        :param activation:
        :param z: shape (d, 1)
        :param W: shape (H, 1)
        :param b: shape (H,)
        :return: H, H_prime, H_double_prime
        """
        z_proj = W @ input_z.T + b[:, None]  # (H, d)
        H = torch.tanh(z_proj)
        H_prime = (1 - H ** 2) * W  # (H, d)
        H_double_prime = -2 * H * (1 - H ** 2) * (W ** 2)  # (H, d)

        return H.T, H_prime.T, H_double_prime.T  # shapes: (d, H)

    H_matrix, H_dot, H_ddot = compute_hidden_activations(epochs_nd_norm_reshaped_tensor)

    # === Residual Function for Least Squares ===
    def loss_function(beta_flat):
        beta_tensor = beta_flat.view(q, H_size)

        Y_pred = H_matrix @ beta_tensor.T  # (N, q)
        Y_pred_km = Y_pred * L
        Y_dot_pred = c * H_dot @ beta_tensor.T
        Y_dot_pred_kms = Y_dot_pred * L / T

        Y_pred_obs = Y_pred[obs_indices]
        Y_pred_obs_km = Y_pred_obs * L
        Y_dot_pred_obs = Y_dot_pred[obs_indices]
        Y_dot_pred_obs_kms  = Y_dot_pred_obs * L / T
        positions.append(Y_pred_obs_km.detach().cpu().numpy())  # undo scaling and move to original CR3BP frame
        velocities.append(Y_dot_pred_obs_kms.detach().cpu().numpy())

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

                topocentric_los_vec = Y_pred_obs - observer_positions / L

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
            sin_ra_predicted, cos_ra_predicted, sin_dec_predicted, cos_dec_predicted = (torch.sin(ra_predicted),
                                                                     torch.cos(ra_predicted),
                                                                     torch.sin(dec_predicted), torch.cos(dec_predicted))

            Y_pred_ang = torch.stack([sin_ra_predicted, cos_ra_predicted, sin_dec_predicted, cos_dec_predicted], dim=1)
            Y_obs = torch.stack(y_obs, dim=1)

            # Compute MSE between predicted and observed [sin RA, cos RA, sin DEC]
            obs_res = torch.mean((Y_pred_ang - Y_obs) ** 2)

            return obs_res

        obs_residual = ra_dec_observation_residual()  # (N * q)

        Y_ddot_pred = c ** 2 * (H_ddot @ beta_tensor.T)  # (N, q)

        def nbody_physics_residual(Y_preds, Y_ddot_preds, epochs, configuration):
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

            epochs_val = [t.tdb.jd for t in epochs]
            positions, masses = get_nbody_positions(epochs_val, configuration)

            # Nondimensionalize positions (geocentric)
            positions_nd = positions / L  # Now unitless

            G_km = (configuration['GRAVITATIONAL_CONSTANT'] / configuration['KM_TO_M'] ** 3)
            mus = G_km * masses
            mus_nd = mus / (G_km * sys_mass)

            N, n, _ = positions_nd.shape
            Y_expanded = Y_preds[:, None, :]  # (N, 1, 3)

            r_vecs = positions_nd - Y_expanded  # (N, n, 3)
            r_norms = torch.norm(r_vecs, dim=-1, keepdim=True)  # (N, n, 1)

            accel_terms = mus_nd[None, :, None] * r_vecs / (r_norms ** 3)  # (N, n, 3)

            total_accel = accel_terms.sum(dim=1)  # (N, 3)

            return torch.mean((Y_ddot_preds - total_accel) ** 2)

        physics_residual = nbody_physics_residual(Y_pred, Y_ddot_pred, colloc_epochs, configuration)


        def distance_penalty(Y_predicted):
            return torch.mean(torch.sum(Y_predicted ** 2, dim=1))

        lambda_dis = parameters['LAMBDA_DIST']
        weighted_dist_res = lambda_dis * distance_penalty(Y_pred)

        data_losses.append(obs_residual.item())
        physics_losses.append(lambda_phys * physics_residual.item())
        range_losses.append(weighted_dist_res.item())

        # print("Obs res:", obs_residual.item(), "Phys res:", physics_residual.item())
        # print(Y_pred[obs_indices][0].cpu().detach())

        return obs_residual + lambda_phys * physics_residual + weighted_dist_res

    def func(beta_flat):
        total_its[0] += 1
        # print(total_its[0])
        beta_tensor_np = beta_flat.reshape(q, H_size)
        beta_tensor = torch.tensor(beta_tensor_np, dtype=torch.float32, requires_grad=True)
        loss = loss_function(beta_tensor)
        loss.backward()
        grad = beta_tensor.grad.detach().numpy().astype(np.float64).reshape(-1)

        return loss.item(), grad

    # === Initial Guess ===
    # Instantiate your MyStep
    mystep = MyStep(
        q=3,
        H_size=parameters['HIDDEN_DIMENSION'],
        observations=y_obs,
        H=H_matrix,
        cH_dot=c * H_dot,
        rho_range=(parameters['MIN_RHO'], parameters['MAX_RHO']),
        rho_dot_range=(parameters['MIN_RHO_DOT'], parameters['MAX_RHO_DOT']),
        observer_positions=observer_positions,
        observer_velocities=torch.zeros_like(observer_positions),
        obs_indices=obs_indices,
        obs_epochs=colloc_epochs[obs_indices],
        delta_rho=parameters['DELTA_RHO'],
        delta_rho_dot=parameters['DELTA_RHO_DOT'],
        delta_rho_step=parameters['DELTA_RHO_STEP']
    )

    beta0 = mystep.take_initial_step(n_init=parameters['INITIAL_TRAJECTORIES'])

    # Basin hopping configuration
    options = {"ftol": parameters['F_TOLERANCE'], "gtol": parameters['G_TOLERANCE'],
               "maxfun": parameters['MAX_FUNCTION_EVAL'],
               "maxiter": parameters['MAX_ITERATiONS'], "disp": False}
    minimizer_kwargs = {"method": "L-BFGS-B", "jac": True, "options": options}  # Local optimizer

    def callback(x, f, accept):
        # print(f"Global iteration {global_its[0]} complete: accepted={accept}, epsilon={epsilon[0]}")
        global_its[0] += 1

    # Run basin hopping
    res = basinhopping(func, beta0, minimizer_kwargs=minimizer_kwargs, niter=parameters['NUMBER_OF_ITERATIONS'],
                       stepsize=parameters['STEPSIZE'], T=parameters['TEMPERATURE'], callback=callback, take_step=mystep, disp=True)

    L = configuration['AU_TO_M'] / configuration['KM_TO_M']  # Length scale in km
    G_km = (configuration['GRAVITATIONAL_CONSTANT'] / configuration['KM_TO_M'] ** 3)
    sys_mass = configuration['SUN_MASS'] + configuration['MOON_MASS'] + configuration['EARTH_MASS']
    T = np.sqrt(L ** 3 / (G_km * sys_mass))

    # === Residual Function for Least Squares ===
    def residual_function(beta_flat):
        beta_tensor = beta_flat.view(q, H_size)

        Y_pred = H_matrix @ beta_tensor.T  # (N, q)
        Y_pred_km = Y_pred * L
        Y_dot_pred = c * H_dot @ beta_tensor.T
        Y_dot_pred_kms = Y_dot_pred * L / T

        Y_pred_obs = Y_pred[obs_indices]
        Y_pred_obs_km = Y_pred_obs * L
        Y_dot_pred_obs = Y_dot_pred[obs_indices]
        Y_dot_pred_obs_kms = Y_dot_pred_obs * L / T
        positions.append(Y_pred_obs_km.detach().cpu().numpy())  # undo scaling and move to original cr3bp frame
        velocities.append(Y_dot_pred_obs_kms.detach().cpu().numpy())


        if first[0] == 0:
            nlls_start[0] = total_its[0]
            first[0] = 1
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

                topocentric_los_vec = Y_pred_obs - observer_positions / L # (N, 3), predict observer centered at origin

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
            sin_ra_predicted, cos_ra_predicted, sin_dec_predicted, cos_dec_predicted = (torch.sin(ra_predicted),
                                                                                        torch.cos(ra_predicted),
                                                                                        torch.sin(dec_predicted),
                                                                                        torch.cos(dec_predicted))

            Y_pred_ang = torch.stack([sin_ra_predicted, cos_ra_predicted, sin_dec_predicted, cos_dec_predicted], dim=1)
            Y_obs = torch.stack(y_obs, dim=1)

            # Compute MSE between predicted and observed [sin RA, cos RA, sin DEC]
            obs_res = (Y_pred_ang - Y_obs).reshape(-1)

            return obs_res


        obs_residual = ra_dec_observation_residual()  # (N * q)


        Y_ddot_pred = c ** 2 * (H_ddot @ beta_tensor.T)  # (N, q)

        def nbody_physics_residual(Y_preds, Y_ddot_preds, epochs, configuration):
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

            epochs_val = [t.tdb.jd for t in epochs]
            positions, masses = get_nbody_positions(epochs_val, configuration)

            # Nondimensionalize positions (geocentric)
            positions_nd = positions / L  # Now unitless

            G_km = (configuration['GRAVITATIONAL_CONSTANT'] / configuration['KM_TO_M'] ** 3)
            mus = G_km * masses
            mus_nd = mus / (G_km * sys_mass)

            N, n, _ = positions_nd.shape
            Y_expanded = Y_preds[:, None, :]  # (N, 1, 3)

            r_vecs = positions_nd - Y_expanded  # (N, n, 3)
            r_norms = torch.norm(r_vecs, dim=-1, keepdim=True)  # (N, n, 1)

            accel_terms = mus_nd[None, :, None] * r_vecs / (r_norms ** 3)  # (N, n, 3)

            total_accel = accel_terms.sum(dim=1)  # (N, 3)

            return (Y_ddot_preds - total_accel).reshape(-1)


        physics_residual = nbody_physics_residual(Y_pred, Y_ddot_pred, colloc_epochs, configuration)


        def distance_penalty(Y_predicted):
            return torch.sum(Y_predicted ** 2, dim=1).reshape(-1)

        lambda_dis = parameters['LAMBDA_DIST']
        dist_res = distance_penalty(Y_pred)

        data_losses.append((torch.mean(obs_residual ** 2)).item())
        physics_losses.append(lambda_phys * (torch.mean(physics_residual ** 2)).item())
        range_losses.append(lambda_dis * torch.mean(dist_res).item())

        # return torch.cat([obs_residual, lambda_phys * physics_residual, lambda_dis * dist_res])

        return torch.cat([obs_residual / np.sqrt(len(obs_residual)), lambda_phys * physics_residual / np.sqrt(len(physics_residual)),
                          lambda_dis * dist_res / np.sqrt(len(dist_res))])

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

    def bounded_around(beta0, percent=0.01, min_step=1e-8):
        beta0 = np.asarray(beta0)
        delta = np.maximum(np.abs(beta0) * percent, min_step)

        lower = beta0 - delta
        upper = beta0 + delta

        # ensure lower < upper elementwise
        lower_bound = np.minimum(lower, upper)
        upper_bound = np.maximum(lower, upper)

        # if beta0[i] == 0 → [-min_step, +min_step]
        return (lower_bound, upper_bound)


    beta00 = res.x
    bounds = bounded_around(beta00, percent=0.05)
    # Solve least-squares
    res2 = least_squares(
        fun=residual_np,
        x0=beta00,
        jac=jacobian_np,
        verbose=2,
        method='trf',
        xtol=parameters['X_TOLERANCE'],
        ftol=parameters['F_TOLERANCE'],
        max_nfev=parameters['MAX_NFEV'],
        bounds=bounds
    )

    beta_tensor_bh = np.asarray(res.x).reshape(q, H_size)
    Y_pred_bh = H_matrix @ beta_tensor_bh.T  # (N, q)
    Y_pred_bh_km = Y_pred_bh * L
    Y_dot_pred_bh = c * H_dot @ beta_tensor_bh.T
    Y_dot_pred_bh_kms = Y_dot_pred_bh * L / T

    Y_pred_obs_bh = Y_pred_bh[obs_indices]
    Y_pred_obs_bh_km = Y_pred_obs_bh * L
    Y_dot_pred_obs_bh = Y_dot_pred_bh[obs_indices]
    Y_dot_pred_obs_bh_kms = Y_dot_pred_obs_bh * L / T
    positions[-2] = Y_pred_obs_bh_km.detach().cpu().numpy()
    velocities[-2] = Y_dot_pred_obs_bh_kms.detach().cpu().numpy()

    beta_tensor_nlls = np.asarray(res2.x).reshape(q, H_size)
    Y_pred_nlls = H_matrix @ beta_tensor_nlls.T  # (N, q)
    Y_dot_pred_nlls = c * H_dot @ beta_tensor_nlls.T

    Y_pred_obs_nlls = Y_pred_nlls[obs_indices]
    Y_dot_pred_obs_nlls = Y_dot_pred_nlls[obs_indices]
    positions[-1] = ((Y_pred_obs_nlls * L).detach().cpu().numpy())
    velocities[-1] = (Y_dot_pred_obs_nlls * L / T).detach().cpu().numpy()
    # print(res.x)

    # for error measurement purposes
    initial_colloc_epoch = colloc_epochs[0]
    final_colloc_epoch = colloc_epochs[-1]

    initial_obs_epoch = colloc_epochs[obs_indices][0]
    final_obs_epoch = colloc_epochs[obs_indices][-1]

    num_points = 1000
    test_epochs = np.linspace(initial_obs_epoch, final_obs_epoch, num_points)

    # Combine, ensuring the start and end colloc epochs are at the edges
    total_test_epochs = np.concatenate((
        [initial_colloc_epoch],
        test_epochs,
        [final_colloc_epoch]
    ))

    # normalize epochs (inputs)
    test_epochs_nd_norm, c_test = epoch_normalization(total_test_epochs, parameters['INPUT_RANGE'], configuration)
    tests_epochs_nd_norm_reshaped_tensor = torch.tensor(test_epochs_nd_norm, dtype=torch.float32).unsqueeze(1)

    H_test, H_dot_test, H_ddot_test = compute_hidden_activations(tests_epochs_nd_norm_reshaped_tensor)

    test_Y_pred_nlls = H_test @ beta_tensor_nlls.T  # (N, q)
    test_Y_dot_pred_nlls = c_test * H_dot_test @ beta_tensor_nlls.T

    final_positions_all_nlls = (test_Y_pred_nlls * L).detach().cpu().numpy()
    final_velocities_all_nlls = (test_Y_dot_pred_nlls * L / T).detach().cpu().numpy()

    final_positions = [final_positions_all_nlls, final_positions_all_nlls]
    final_velocities = [final_velocities_all_nlls, final_velocities_all_nlls]

    epochss = np.arange(total_its[0])
    data = {"TRAINING_EPOCH": epochss, "DATA_LOSS": data_losses, "PHYSICS_LOSS": physics_losses, "RANGE_LOSS": range_losses}

    return pd.DataFrame(data), positions, velocities, nlls_start[0], final_positions, final_velocities, end - start
