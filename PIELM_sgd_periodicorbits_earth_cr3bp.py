import numpy as np
from typing import List, Literal, Tuple, Union
import torch
import pandas as pd
import os
import torch.nn as nn
import n_body_integrator as nbody
from scipy.integrate import odeint
import time


####
# generate data
###
def generate_data(config, parameters):


    # get orbit type
    orbit_type = parameters['ORBIT_TYPE']
    print(orbit_type)

    # get orbit number
    orbit_number = parameters['RUN_NUMBER']

    # get orbit file and trajectory data
    folder = [f for f in os.listdir() if os.path.isdir(f) and orbit_type in f][0]
    folder_path = os.path.join(os.getcwd(), folder)
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    files_sorted = sorted(file_paths, key=lambda x: float(os.path.basename(x)))
    file = files_sorted[orbit_number]
    print(file)
    data = pd.read_csv(file)

    # 3. Time array: propagate for half a period
    # T_half = data['halfT'].iloc[0]
    num_obs = parameters['NUMBER_OF_OBSERVATIONS']  # number of points
    mu = config['SYSTEM_MASS_PARAMETER']

    def sample_equally_spaced(df, n):
        if n >= len(df) / 2:
            return df.copy()
        indices = np.linspace(0 * len(df) / 4, 1 * len(df) / 8 - 1, n, dtype=int)
        return df.iloc[indices]

    trajectory = sample_equally_spaced(data, num_obs)
    positions = trajectory.loc[:, ['x', 'y', 'z']].values
    velocities = trajectory.loc[:, ['vx', 'vy', 'vz']].values
    observation_epochs = trajectory['time'].values
    observer_position = np.array([1. - mu, 0, 0])
    observer_positions = np.tile(observer_position, (velocities.shape[0], 1))
    # positions -= observer_positions
    # observer_positions = np.tile(np.array([0., 0., 0.]), (velocities.shape[0], 1))

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

        los_vec = positions - observer_positions  # Now both are Quantity arrays (3,)

        # Normalize
        los_unit = los_vec / np.linalg.norm(los_vec, axis=1, keepdims=True)


        # Convert to RA/DEC
        x, y, z = los_unit.T

        ra = np.arctan2(y, x)
        dec = np.arcsin(z)

        # Normalize RA to [0, 360)
        ra = ra % (2 * torch.pi)

        return torch.tensor(np.array(ra), dtype=torch.float32), torch.tensor(np.array(dec), dtype=torch.float32)

    ra, dec = generate_ra_dec_measurements()
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
            observation_epochs, positions, velocities, ra, dec, sigma_ra_deg, sigma_dec_deg, file)



####
# PIELM
####
class ELM(nn.Module):
    def __init__(self, hidden_dim, q=3, activation=torch.tanh, c_normalization=1.0, initial_nd_positions=None, z_s=None, config=None):
        """
        :param input_dim: Dimensionality of each input vector (e.g., 1 for time).
        :param hidden_dim: Number of hidden neurons (N_star).
        :param q: Number of output dimensions (e.g., 3 for 3D position).
        :param activation: Activation function, e.g., tanh.
        """
        super().__init__()
        input_dim= 1
        self.input_weights = nn.Parameter(config['WEIGHT_SCALE_FACTOR'] * (2 * torch.rand(hidden_dim, input_dim) - 1), requires_grad=False)  # (H x 1)
        self.bias = nn.Parameter(config['WEIGHT_SCALE_FACTOR'] * (2 * torch.rand(hidden_dim) - 1), requires_grad=False)  # (H,)
        self.activation = activation
        if initial_nd_positions is None:
            self.output_weights = nn.Parameter(torch.rand(q, hidden_dim))  # (q x H) random initializtion
        else:
            H = torch.tanh(self.input_weights @ z_s.T + self.bias[:, None])
            H_inv = torch.linalg.pinv(H)
            self.output_weights = nn.Parameter(initial_nd_positions @ H_inv )
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


    # === Nondimensionalize ===
    t_nondim = epoch
    t0, tf = t_nondim[0], t_nondim[-1]

    # === Normalize to z_range ===
    z0, zf = z_range
    scale = (zf - z0) / (tf - t0)
    normalized_epoch = z0 + scale * (t_nondim - t0)

    return normalized_epoch, scale


def run(data, config, parameters):
    # get the collocation points
    colloc_points = sample_time_points(parameters['SAMPLING_METHOD'], data[2], parameters['TIME_DELTA'].value,
                                       parameters['TOTAL_POINTS'], layer_ratios=parameters['LAYER_RATIOS'],
                                       config=config)

    # normalize epochs (inputs)
    epochs_nd_norm, c = epoch_normalization(colloc_points, parameters['INPUT_RANGE'], config)
    epochs_nd_norm_reshaped_tensor = torch.tensor(epochs_nd_norm, dtype=torch.float32).unsqueeze(1)  # as a 2D tensor

    # get the indices where observations are
    obs_mask = np.isin(colloc_points, data[2])
    obs_indices = np.where(obs_mask)[0]

    elm = ELM(parameters['HIDDEN_DIMENSION'], c_normalization=c, config=parameters)
    start = time.time()
    data_df, positions, velocities, nlls_start, final_positions, final_velocities = train(elm,
        epochs_nd_norm_reshaped_tensor, data[0], obs_indices, data[1], colloc_points, config, parameters)
    end = time.time()

    # true positions
    mu = config['SYSTEM_MASS_PARAMETER']
    asteroid_ini_pos = data[3][0, :]
    asteroid_ini_vel = data[4][0, :]
    ini_state = np.concatenate([asteroid_ini_pos, asteroid_ini_vel])
    phi_0 = np.eye(6)  # initial Phi (state transition matrix)
    state = np.hstack((np.array(ini_state), phi_0.ravel()))
    num_points = 1000
    epochs = np.linspace(data[2][0], data[2][-1], num_points)
    res = odeint(nbody.cr3bp, state, epochs, args=(mu,))
    asteroid_cr3bp_position = np.array(res[:, :3])
    asteroid_cr3bp_velocity = np.array(res[:, 3:])

    return data_df, positions, velocities, nlls_start, final_positions, final_velocities, asteroid_cr3bp_position, asteroid_cr3bp_velocity, epochs, end - start


def train(model, epochs_nd_norm_reshaped_tensor, y_obs, obs_indices, observer_positions, colloc_epochs, configuration, parameters):
    epochs = parameters['NUMBER_OF_EPOCHS']
    lr = parameters['LEARNING_RATE']
    lambda_phys = parameters['PHYSICS_WEIGHT']

    optimizer = torch.optim.Adam([model.output_weights], lr=lr)

    data_losses = []
    physics_losses = []
    positions = []
    velocities = []
    total_its = [0]
    global_its = [0]
    epsilon = [0]
    global_positions = []
    nlls_start = [0]
    first = [0]

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass with derivatives
        Y_pred, Y_dot_pred, Y_ddot_pred = model.forward_with_derivatives(epochs_nd_norm_reshaped_tensor)
        Y_pred_obs = Y_pred[obs_indices]
        Y_dot_pred_obs = Y_dot_pred[obs_indices]

        positions.append(
            (Y_pred_obs + observer_positions).detach().cpu().numpy())  # undo scaling and move to original cr3bp frame
        velocities.append(Y_dot_pred_obs.detach().cpu().numpy())
        # Compute losses
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

                topocentric_los_vec = Y_pred_obs  # (N, 3), predict observer centered at origin

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
            obs_res = torch.mean((Y_pred_ang - Y_obs) ** 2)

            return obs_res
        data_loss = ra_dec_observation_residual()

        def cr3bp_physics_residual(Y_predicted, Y_dot_predicted, Y_ddot_predicted, configuration):
            """
            Computes the physics residual based on CR3BP (Circular Restricted Three-Body Problem) dynamics.

            Parameters
            ----------
            Y_predicted : torch.Tensor of shape (N, 3)
                Predicted non-dimensional asteroid positions (in the synodic frame).
            Y_ddot_predicted : torch.Tensor of shape (N, 3)
                Predicted non-dimensional asteroid accelerations (second time derivatives).
            configuration : dict
                Dictionary containing:
                    - 'MU': float, mass parameter (μ = m_secondary / (m_primary + m_secondary)).
                    - 'L': float, characteristic length (used for non-dimensionalization).
                    - Optionally 'MU_UNIT': float, gravitational parameter in dimensional units (ignored here if non-dimensional).

            Returns
            -------
            torch.Tensor of shape (N * 3,)
                Flattened residuals between the predicted accelerations and the CR3BP model accelerations.
            """
            mu = configuration['SYSTEM_MASS_PARAMETER']
            observer_position = torch.tensor([1. - mu, 0., 0.], dtype=Y_predicted.dtype)
            observer_positions = observer_position.unsqueeze(0).repeat(Y_predicted.shape[0], 1)

            Y_predict = Y_predicted + observer_positions  # move back to original
            x = Y_predict[:, 0]
            y = Y_predict[:, 1]
            z = Y_predict[:, 2]
            vx = Y_dot_predicted[:, 0]
            vy = Y_dot_predicted[:, 1]
            vz = Y_dot_predicted[:, 2]

            # alpha = 10.0  # slope control
            # epsilon_0 = np.power(10, -float(2))
            # N = parameters['NUMBER_OF_ITERATIONS']
            # i = global_its[0]
            # fraction = (N - 1 - i) / (N - 1)
            # epsilon[0] = epsilon_0 * fraction ** alpha
            epsilon[0] = 0  # 1e-2 # np.power(10, -float(4))
            # epsilon[0] = epsilon_0  * (1 - i / (N - 1))
            r_2 = torch.sqrt((mu + x - 1) ** 2 + y ** 2 + z ** 2) + epsilon[0]
            r_1 = torch.sqrt((mu + x) ** 2 + y ** 2 + z ** 2)
            dUdx = -(mu * (mu + x - 1)) / torch.pow(r_2, 3) - \
                   ((1 - mu) * (mu + x)) / torch.pow(r_1, 3) + x
            dUdy = - (mu * y) / torch.pow(r_2, 3) - \
                   ((1 - mu) * y) / torch.pow(r_1, 3) + y
            dUdz = - (mu * z) / torch.pow(r_2, 3) - \
                   ((1 - mu) * z) / torch.pow(r_1, 3)

            dxdt = vx  # derivative of position is velocity
            dydt = vy
            dzdt = vz
            ax = dUdx + 2 * dydt  # derivative of velocity is acceleration
            ay = dUdy - 2 * dxdt
            az = dUdz

            true_accel = torch.stack((ax, ay, az), dim=1)  # (N, 3)

            residual = torch.mean((Y_ddot_predicted - true_accel) ** 2)

            return residual

        phys_loss = cr3bp_physics_residual(Y_pred, Y_dot_pred, Y_ddot_pred, configuration)

        data_losses.append(data_loss.item())
        physics_losses.append(lambda_phys * phys_loss.item())

        # Total loss
        loss = data_loss + lambda_phys * phys_loss
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Data Loss = {data_loss.item():.4e}, Physics Loss = {phys_loss.item():.4e}")


    Y_pred_nlls, Y_dot_pred_nlls, Y_ddot_pred_nlls = model.forward_with_derivatives(epochs_nd_norm_reshaped_tensor)

    Y_pred_obs_nlls = Y_pred_nlls[obs_indices]
    Y_dot_pred_obs_nlls = Y_dot_pred_nlls[obs_indices]
    positions[-1] = ((Y_pred_obs_nlls + observer_positions).detach().cpu().numpy())
    velocities[-1] = Y_dot_pred_obs_nlls.detach().cpu().numpy()
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

    Y_test, Y_dot_test, Y_ddot_test = model.forward_with_derivatives(tests_epochs_nd_norm_reshaped_tensor)

    mu = configuration['SYSTEM_MASS_PARAMETER']
    observer_position_final = torch.tensor([1. - mu, 0., 0.], dtype=Y_test.dtype)
    observer_positions_final = observer_position_final.unsqueeze(0).repeat(Y_test.shape[0], 1)

    final_positions_all_nlls = (Y_test + observer_positions_final).detach().cpu().numpy()
    final_velocities_all_nlls = (Y_dot_test).detach().cpu().numpy()

    final_positions = [final_positions_all_nlls, final_positions_all_nlls]
    final_velocities = [final_velocities_all_nlls, final_velocities_all_nlls]

    epochss = np.arange(epochs)

    data = {"TRAINING_EPOCH": epochss, "DATA_LOSS": data_losses, "PHYSICS_LOSS": physics_losses}

    return pd.DataFrame(data), positions, velocities, nlls_start[0], final_positions, final_velocities
