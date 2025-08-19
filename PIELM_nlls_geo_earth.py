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
import n_body_integrator as nbody
import time

####
# generate data
###
def generate_data(config, parameters):
    # generate s/c trajectory - emej2000 (km and km/s)
    # initial orbit parameters: a = 42164, all other elements 0
    # create arc which is half orbit period
    # 1. Define GEO orbital elements
    a = 42164 * u.km       # GEO semi-major axis
    ecc = 0.0 * u.one      # Circular orbit
    inc = 0.0 * u.deg      # Equatorial orbit
    raan = 0.0 * u.deg
    argp = 0.0 * u.deg
    nu = 90.0 * u.deg       # Start at perigee

    # Random perturbation scales (adjust as needed)
    perturb_scale = {
        "a": parameters['A_PERT'] * u.km,  # ±10 km
        "ecc": parameters['ECC_PERT'] * u.one,  # ±0.001
        "inc": parameters['INC_PERT'] * u.deg,  # ±0.1°
        "raan": parameters['RAAN_PERT'] * u.deg,  # ±1°
        "argp": parameters['ARGPER_PERT'] * u.deg,
        "nu": parameters['ANOM_PERT'] * u.deg
    }

    # Generate random perturbations
    rng = np.random.default_rng()
    a_pert = a + rng.uniform(-1, 1) * perturb_scale["a"]
    ecc_pert = ecc + rng.uniform(0, 1) * perturb_scale["ecc"]
    inc_pert = inc + rng.uniform(0, 1) * perturb_scale["inc"]
    raan_pert = raan + rng.uniform(-1, 1) * perturb_scale["raan"]
    argp_pert = argp + rng.uniform(-1, 1) * perturb_scale["argp"]
    nu_pert = nu + rng.uniform(-1, 1) * perturb_scale["nu"]

    # 2. Create orbit
    epoch = Time.now()
    geo_orbit = Orbit.from_classical(Earth, a_pert, ecc_pert, inc_pert, raan_pert, argp_pert, nu_pert, epoch)

    # 3. Time array: propagate for half a period
    T_half = geo_orbit.period * parameters['OBSERVATION_TIME_FRACTION']
    num_obs = parameters['NUMBER_OF_OBSERVATIONS'] # number of points
    observation_epochs = [epoch + (T_half * i / (num_obs - 1)) for i in range(num_obs)]


    # 4. Extract position and velocity at each time step
    object_positions = []
    object_velocities = []

    for t in observation_epochs:
        propagated = geo_orbit.propagate(t - epoch)
        object_positions.append(propagated.r.to_value(u.km))
        object_velocities.append(propagated.v.to_value(u.km / u.s))

    object_positions = np.array(object_positions)
    object_velocities = np.array(object_velocities)


    # obsrevatories (lat deg, lon deg, el m)
    # Catalina (32.417, -110.733, 2791)
    # Haleakala (20.708, -156.257, 3052)
    # Tenerife (28.474, -16.308, 2390)
    # Cerro Tololo (-30.169, -70.806, 2207)
    # Asiago (45.866, 11.5264, 1045)
    # South African Large Telescope (-32.376 20.810678, 1798)
    obs_locations = [{"name": "Catalina", "lat": 32.417, "lon": -110.733, "elev": 2791},
                     {"name": "Haleakala", "lat": 20.708, "lon": -156.257, "elev": 3052},
                     {"name": "Tenerife", "lat": 28.474, "lon": -16.308, "elev": 2390},
                     {"name": "Cerro Tololo", "lat": -30.169, "lon": -70.806, "elev": 2207},
                     {"name": "Asiago", "lat": 45.866, "lon": 11.5264, "elev": 1045},
                     {"name": "SALT", "lat": -32.376, "lon": 20.810678, "elev": 1798}]

    def generate_ra_dec_measurements():
        """
        Generate topocentric RA/DEC observations from the observatory with highest elevation.

        Parameters:
            times: array of datetime or astropy.Time
            sat_positions_eci: Nx3 array in km (GCRS)
            observatories: list of dicts with 'name', 'lat', 'lon', 'elev'

        Returns:
            ra_list, dec_list, used_obs
        """
        ra_list, dec_list, used_obs, used_obs_gcrs_position = [], [], [], []

        times = Time(observation_epochs)
        for t, sat_pos in zip(times, object_positions):
            max_el = -np.inf
            best_obs = None
            best_obs_pos = None
            best_ra = None
            best_dec = None

            for obs in obs_locations:
                location = EarthLocation(
                    lat=obs['lat'] * u.deg,
                    lon=obs['lon'] * u.deg,
                    height=obs['elev'] * u.m
                )

                gcrs_sat = SkyCoord(sat_pos[0] * u.km, sat_pos[1] * u.km, sat_pos[2] * u.km,
                                    frame='gcrs', obstime=t, representation_type='cartesian')

                altaz = gcrs_sat.transform_to(AltAz(obstime=t, location=location))
                if altaz.alt.deg > max_el:
                    max_el = altaz.alt.deg

                    obs_gcrs = location.get_gcrs_posvel(t)[0].xyz.to(u.km)  # position only

                    # Line-of-sight vector in GCRS
                    sat_vec = gcrs_sat.cartesian.xyz.to(u.km)  # Convert SkyCoord to Quantity
                    los_vec = sat_vec - obs_gcrs  # Now both are Quantity arrays (3,)

                    # Normalize
                    los_unit = los_vec / np.linalg.norm(los_vec)

                    # Convert to RA/DEC
                    x, y, z = los_unit

                    ra = np.arctan2(y, x)
                    dec = np.arcsin(z)

                    # Normalize RA to [0, 360)
                    ra = ra.value % (2 * torch.pi)

                    best_ra = ra
                    best_dec = dec.value
                    best_obs = obs
                    best_obs_pos = obs_gcrs

            ra_list.append(best_ra)  # IRCS
            dec_list.append(best_dec)
            used_obs.append(best_obs)
            used_obs_gcrs_position.append(best_obs_pos.value)
        return (torch.tensor(np.array(ra_list), dtype=torch.float32),
                torch.tensor(np.array(dec_list), dtype=torch.float32),
                used_obs, torch.tensor(np.array(used_obs_gcrs_position), dtype=torch.float32))
    ra, dec, observatory_choices, observatory_positions_gcrs = generate_ra_dec_measurements()
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

    sin_ra_meas, cos_ra_meas, sin_dec_meas, cos_dec_meas = torch.sin(ra_m), torch.cos(ra_m), torch.sin(
        dec_m), torch.cos(dec_m)
    file = "NA"

    return ([sin_ra_meas, cos_ra_meas, sin_dec_meas, cos_dec_meas], observatory_positions_gcrs, observation_epochs,
            object_positions, object_velocities, ra, dec, sigma_ra_deg, sigma_dec_deg, file)


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
    L = configuration['normalization_ratio'] * configuration['EARTH_RADIUS_KM']  # Length scale in km
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

    data_df, positions, velocities, nlls_start, final_positions, final_velocities, comp_time = solve(epochs_nd_norm_reshaped_tensor, data[0], obs_indices, data[1], colloc_points, c, config, parameters)

    # Extract initial position/velocity
    ini_pos = data[3][0, :]  # km
    ini_vel = data[4][0, :]  # km/s

    # Set up epochs
    num_points = 1000
    epochs = np.linspace(data[2][0], data[2][-1], num_points)  # still astropy Time objects

    # Initial epoch is just the first time
    epoch0 = data[2][0]  # already Time

    # Uniform time step in seconds
    timestep_sec = (epochs[1] - epochs[0]).sec

    # Call integrator
    tpositions, tvelocities, propagated_epochs = nbody.two_body_integrator(
        r0_km=ini_pos,
        v0_kms=ini_vel,
        epoch=epoch0,
        timestep_sec=timestep_sec,
        num_frames=num_points
    )

    return data_df, positions, velocities, nlls_start, final_positions, final_velocities, tpositions, tvelocities, propagated_epochs, comp_time


def solve(epochs_nd_norm_reshaped_tensor, y_obs, obs_indices, observer_positions, colloc_epochs, c, configuration,
          parameters):

    L = configuration['normalization_ratio'] * configuration['EARTH_RADIUS_KM']  # Length scale in km
    mu_E = configuration['EARTH_MASS_PARAMETER']
    T = np.sqrt(L ** 3 / mu_E)
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
    def residual_function(beta_flat):
        beta_tensor = beta_flat.view(q, H_size)

        Y_pred = H_matrix @ beta_tensor.T  # (N, q)
        Y_dot_pred = c * H_dot @ beta_tensor.T

        Y_pred_obs = Y_pred[obs_indices]
        Y_pred_obs_km = Y_pred_obs * L
        Y_dot_pred_obs = Y_dot_pred[obs_indices] * L / T

        positions.append(Y_pred_obs_km.detach().cpu().numpy())
        velocities.append(Y_dot_pred_obs.detach().cpu().numpy())
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

        def twobody_physics_residual(Y_predicted, Y_ddot_predicted):
            """
            Y_pred: (N, 3) — nondimensional predicted positions of the asteroid (in geocentric or heliocentric frame)
            Y_ddot_pred: (N, 3) — nondimensional predicted accelerations
            configuration: dict with 'MU' and 'L' (characteristic length), and optionally 'MU_UNIT' for nondimensionalization
            """

            r_norms = torch.norm(Y_predicted, dim=-1, keepdim=True)  # (N, 1)
            true_accel = - Y_predicted / (r_norms ** 3)  # (N, 3) - non - dimensional

            residual = Y_ddot_predicted - true_accel  # (N, 3)
            return residual.reshape(-1)

        physics_residual = twobody_physics_residual(Y_pred, Y_ddot_pred)

        data_losses.append((torch.mean(obs_residual ** 2)).item())
        physics_losses.append(lambda_phys * (torch.mean(physics_residual ** 2)).item())

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

    start = time.time()

    # === Solve with SciPy ===
    res2 = least_squares(
        fun=residual_np,
        x0=beta0,
        jac=jacobian_np,
        verbose=2,
        method='trf',  # or 'trf', depending on structure
        xtol=parameters['X_TOLERANCE'],
        ftol=parameters['F_TOLERANCE'],
        max_nfev=parameters['MAX_NFEV'],
    )

    end = time.time()

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
    data = {"TRAINING_EPOCH": epochss, "DATA_LOSS": data_losses, "PHYSICS_LOSS": physics_losses}

    return pd.DataFrame(data), positions, velocities, nlls_start[0], final_positions, final_velocities, end - start
