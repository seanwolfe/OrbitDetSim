from astropy.time import Time
import numpy as np
from typing import List, Literal, Tuple, Union
import torch
from scipy.optimize import basinhopping
import pandas as pd
import spiceypy as spice
import n_body_integrator as nbody
import utilities as util
import time
import os
import random

# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')


def _set_reproducibility_seed(config):
    """Best-effort deterministic seeding for one IOD/hyperparameter task.

    The hyperparameter runner sets TASK_SEED and seed in a per-task copy of
    the config. This helper makes NumPy/Python/Torch randomness deterministic
    for noise injection, collocation sampling, ELM weights, and basin-hopping
    perturbations.
    """
    seed = config.get("TASK_SEED", config.get("seed", None))
    if seed is None:
        return
    try:
        seed = int(seed) % (2**32 - 1)
    except Exception:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


####
# generate data
###
def generate_data(config, parameters, *, master_row=None, saved_as=None):
    _set_reproducibility_seed(config)
    """
    Build inputs for the IOD solver using the per-row file(s) recorded in MASTER.
    One run per MASTER row.

    Pass either:
      - saved_as: the string from MASTER['IOD_DATA_SAVED_AS'] (e.g. 'a.parquet;a.csv'), or
      - master_row: a dict/Series with key 'IOD_DATA_SAVED_AS'

    Returns (same structure as before):
      ([sin_ra_meas, cos_ra_meas, sin_dec_meas, cos_dec_meas],
       observer_positions,
       observation_epochs,
       positions,
       velocities,
       ra, dec,
       sigma_ra_deg, sigma_dec_deg,
       file_path,            # <- now chosen from MASTER
       observer_velocities)
    """

    def _iod_dir(config):
        root = os.path.abspath(os.path.join(config['top_dir'], config['IOD_folder_path']))
        return root

    # ----- resolve which per-row file to use -----
    if saved_as is None:
        if master_row is not None:
            saved_as = master_row.get("IOD_DATA_SAVED_AS", "")
        else:
            saved_as = parameters.get("MASTER_SAVED_AS", "")  # optional fallback

    saved_as = str(saved_as or "")
    if not saved_as.strip():
        raise FileNotFoundError("MASTER row has empty IOD_DATA_SAVED_AS; cannot load per-row IOD data.")

    # Multiple entries allowed, separated by ';' (we stored 'parquet;csv' in that order when both exist)
    names = [s.strip() for s in saved_as.split(";") if s.strip()]
    if not names:
        raise FileNotFoundError("Parsed IOD_DATA_SAVED_AS produced no filenames.")

    iod_dir = _iod_dir(config)  # points to .../IOD_folder_path/spacecraft_{num_sc}
    # Prefer parquet if present; else take the first
    def pick_full_path(names_list):
        # exact join first
        parquet = [n for n in names_list if n.lower().endswith(".parquet")]
        ordered = (parquet + [n for n in names_list if n.lower().endswith(".csv")]) or names_list
        for n in ordered:
            p = os.path.join(iod_dir, n)
            if os.path.exists(p):
                return p
        # last resort: recursive search (robust to subdirs)
        for n in ordered:
            for root, _, files in os.walk(iod_dir):
                if n in files:
                    return os.path.join(root, n)
        return None

    file_path = pick_full_path(names)
    if file_path is None:
        raise FileNotFoundError(f"Could not locate any of {names} under {iod_dir}")

    # ----- load the per-row IOD data -----
    iod_data = util.read_IOD_data_geo(file_path, config)

    # measurements
    sin_ra_meas  = torch.tensor(iod_data['SIN_RA_PHYS'].values, dtype=torch.float32)
    cos_ra_meas  = torch.tensor(iod_data['COS_RA_PHYS'].values, dtype=torch.float32)
    sin_dec_meas = torch.tensor(iod_data['SIN_DEC_PHYS'].values, dtype=torch.float32)

    observer_positions = torch.tensor(
        iod_data.loc[:, ["SC_GEO_X(KM)_PHYS", "SC_GEO_Y(KM)_PHYS", "SC_GEO_Z(KM)_PHYS"]].values,
        dtype=torch.float32
    )
    observer_velocities = torch.tensor(
        iod_data.loc[:, ["SC_GEO_VX(KM/S)_PHYS", "SC_GEO_VY(KM/S)_PHYS", "SC_GEO_VZ(KM/S)_PHYS"]].values,
        dtype=torch.float32
    )
    observation_epochs = [Time(jd, format='jd', scale='tdb') for jd in iod_data['EPOCH(JDTDB)'].values]

    # truth/reference states (if present)
    positions  = iod_data.loc[:, ["GEO_X(KM)",  "GEO_Y(KM)",  "GEO_Z(KM)"]].values
    velocities = iod_data.loc[:, ["GEO_VX(KM/S)","GEO_VY(KM/S)","GEO_VZ(KM/S)"]].values

    # reconstruct angles
    ra_meas  = torch.atan2(sin_ra_meas, cos_ra_meas)
    ra       = ra_meas % (2 * np.pi)      # [0, 2π)
    dec      = torch.asin(sin_dec_meas)   # [-π/2, π/2]

    # optional noise
    if int(config.get('ADD_NOISE', 0)) == 1:
        def add_noise(ra_rad, dec_rad, cfg):
            # total (mas)
            sigma_ra_mas  = torch.sqrt(torch.tensor(cfg['sigma_ra']**2  + cfg['sigma_pointing']**2))
            sigma_dec_mas = torch.sqrt(torch.tensor(cfg['sigma_dec']**2 + cfg['sigma_pointing']**2))
            # deg
            sigma_ra_deg  = sigma_ra_mas  / cfg['MAS_TO_DEGREE']
            sigma_dec_deg = sigma_dec_mas / cfg['MAS_TO_DEGREE']
            # rad
            sigma_ra_rad  = torch.deg2rad(sigma_ra_deg)
            sigma_dec_rad = torch.deg2rad(sigma_dec_deg)
            # noise
            ra_noise  = torch.normal(mean=0.0, std=sigma_ra_rad,  size=ra_rad.shape)
            dec_noise = torch.normal(mean=0.0, std=sigma_dec_rad, size=dec_rad.shape)
            # apply
            ra_noisy  = (ra_rad + ra_noise) % (2 * torch.pi)
            dec_noisy = torch.clamp(dec_rad + dec_noise, min=-torch.pi/2, max=torch.pi/2)
            return ra_noisy, dec_noisy, sigma_ra_deg, sigma_dec_deg

        ra_m, dec_m, sigma_ra_deg, sigma_dec_deg = add_noise(ra, dec, config)
    else:
        ra_m, dec_m = ra.clone(), dec.clone()
        sigma_ra_deg = 0.0
        sigma_dec_deg = 0.0

    sin_ra_meas = torch.sin(ra_m)
    cos_ra_meas = torch.cos(ra_m)
    sin_dec_meas = torch.sin(dec_m)
    cos_dec_meas = torch.cos(dec_m)

    return (
        [sin_ra_meas, cos_ra_meas, sin_dec_meas, cos_dec_meas],
        observer_positions,
        observation_epochs,
        positions,
        velocities,
        ra, dec,
        sigma_ra_deg, sigma_dec_deg,
        file_path,                  # used by callers as "FILE_USED"
        observer_velocities
    )



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


    mean = t0.value + (tN.value - t0.value) / 2
    std = (tN.value - t0.value) / config['gaussian_std_scale']


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
            if domain_start.value <= x <= domain_end.value:
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
    if method == 'gaussian':
        combined = np.concatenate([[observation.value for observation in observation_epochs], additional_samples])
        time_combined = Time(combined, format='jd', scale='tdb')
        return np.sort(time_combined)
    else:
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
    _set_reproducibility_seed(config)
    # get the collocation points
    colloc_points = sample_time_points(parameters['SAMPLING_METHOD'], data[2], parameters['TIME_DELTA'],
                                       parameters['TOTAL_POINTS'], layer_ratios=parameters['LAYER_RATIOS'], config=config)

    # normalize epochs (inputs)
    epochs_nd_norm, c = epoch_normalization(colloc_points, parameters['INPUT_RANGE'], config)
    epochs_nd_norm_reshaped_tensor = torch.tensor(epochs_nd_norm, dtype=torch.float32).unsqueeze(1)  # as a 2D tensor

    # get the indices where observations are
    obs_mask = np.isin(colloc_points, data[2])
    obs_indices = np.where(obs_mask)[0]
    print("before solve")

    data_df, positions, velocities, nlls_start, final_positions, final_velocities, comp_time, best_bh = solve(epochs_nd_norm_reshaped_tensor, data[0], obs_indices, data[1], colloc_points, c, config, parameters, data[-1])

    print("after solve")

    # Extract initial position/velocity
    ini_pos = data[3][0, :]  # km
    ini_vel = data[4][0, :]  # km/s

    # Set up epochs
    num_points = config['error']['num_points']
    epochs = np.linspace(data[2][0], data[2][-1], num_points)  # still astropy Time objects

    # Initial epoch is just the first time
    epoch0 = data[2][0]  # already Time

    # Uniform time step in seconds
    timestep_sec = (epochs[1] - epochs[0]).sec

    # calc epochs
    total_observation_window = num_points * timestep_sec  # epoch is in jd

    # Convert astropy Time -> JD (TDB scale)
    jd_tdb = epoch0.tdb.jd  # this is a float

    # Convert JD_TDB -> ET (seconds past J2000)
    epoch_et = spice.unitim(jd_tdb, 'JDTDB', 'ET')

    def get_state(body, reference=10):
        state, _ = spice.spkgeo(body, epoch_et, "ECLIPJ2000", reference)
        return np.array(state)

    earth_state = get_state(399)

    asteroid_state_geo = np.concatenate([ini_pos, ini_vel])
    asteroid_state_helio = util.geo_eme_to_geo_eclip_generic(asteroid_state_geo) + earth_state

    # integrate s/c traj
    asteroid_integrated_states, asteroid_earth_states = nbody.integrate_n_body(asteroid_state_helio,
                                                                               jd_tdb,
                                                                               total_observation_window,
                                                                               timestep_sec,
                                                                               type="ASTEROID")  # integrator takes seconds

    asteroid_int_geo = (asteroid_integrated_states - asteroid_earth_states)
    asteroid_eme = util.geo_eclip_to_geo_eme_generic(asteroid_int_geo, layout="time")

    return data_df, positions, velocities, nlls_start, final_positions, final_velocities, asteroid_eme[:3, :].T, asteroid_eme[3:, :].T, epochs, comp_time, best_bh


def solve(epochs_nd_norm_reshaped_tensor, y_obs, obs_indices, observer_positions, colloc_epochs, c, configuration, parameters, observer_velocity):
    viz_flag = bool(configuration.get('visualization_flag', 0))

    L = configuration['AU_TO_M'] / configuration['KM_TO_M']  # Length scale in km
    G_km = (configuration['GRAVITATIONAL_CONSTANT'] / configuration['KM_TO_M'] ** 3)
    sys_mass = configuration['SUN_MASS'] + configuration['MOON_MASS'] + configuration['EARTH_MASS']
    T = np.sqrt(L ** 3 / (G_km * sys_mass))
    lambda_phys = parameters['PHYSICS_WEIGHT']
    lambda_dist = parameters['LAMBDA_DIST']
    q = 3
    # === Dummy inputs for illustration ===
    # Dimensions
    H_size = parameters['HIDDEN_DIMENSION']  # hidden layer size
    # Fake precomputed hidden layer activations and derivatives
    W = (2 * torch.rand(H_size, 1) - 1) * parameters['WEIGHT_SCALE_FACTOR']
    b = (2 * torch.rand(H_size) - 1) * parameters['WEIGHT_SCALE_FACTOR']

    data_losses = []
    physics_losses = []
    range_losses = []
    positions = []
    velocities = []
    total_its = [0]
    global_its = [0]
    best = {'fun': np.inf, 'x': None, 'iteration': None}
    epsilon = [0]
    global_positions = []
    nlls_start = [0]
    first = [0]

    class MyStep:
        """
         Structured basin-hopping step proposal constrained by the admissible region.

         This class implements a physics-informed proposal mechanism for basin-hopping
         in orbit determination problems. It perturbs the current solution `beta` by sampling
         line-of-sight range (`rho`) and range-rate (`rho_dot`) uniformly within user-specified bounds
         at each observation epoch. The resulting state vectors remain consistent with the
         observed right ascension (`alpha`) and declination (`delta`) and their best-fit angular rates.

         The step is constructed in the admissible region defined by the current observations and
         projected back into the parameter space via the pseudoinverse of the design matrix.

         Attributes:
             q (int): Dimension of position/velocity vectors (typically 3 for Cartesian space).
             H_size (int): Size of the basis vector `beta`.
             observations (torch.Tensor): Observations at each epoch, shape (N, 4), containing:
                 [sin(alpha), cos(alpha), sin(delta), cos(delta)] per row.
             pos_step (float): Default positional step size scale (not currently used directly).
             vel_step (float): Default velocity step size scale (not currently used directly).
             H (torch.Tensor): Design matrix for position at each epoch, shape (N, H_size).
             cH_dot (torch.Tensor): Design matrix for velocity at each epoch, shape (N, H_size).
             rho_range (tuple): (min_rho, max_rho), uniform sampling range for line-of-sight distance.
             rho_dot_range (tuple): (min_rho_dot, max_rho_dot), uniform sampling range for line-of-sight rate.
             observer_positions (torch.Tensor): Observer position vectors at each epoch, shape (N, 3).
             observer_velocities (torch.Tensor): Observer velocity vectors at each epoch, shape (N, 3).
             obs_indices (torch.Tensor or list): Indices of observations in `H`/`cH_dot`.
             obs_epochs (torch.Tensor): Times of each observation epoch, shape (N,).

         Methods:
             __call__(x: torch.Tensor) -> torch.Tensor:
                 Delegates to `take_step(x)`.

             take_step(beta: torch.Tensor) -> torch.Tensor:
                 Propose a new `beta` vector by perturbing the current state in the admissible region.

                 Args:
                     beta (torch.Tensor): Current parameter vector, shape (q * H_size,).

                 Returns:
                     beta_new (torch.Tensor): Proposed parameter vector after admissible perturbation,
                         shape (q * H_size,).
         """

        def __init__(self, q=3, H_size=10, observations=None, H=None, cH_dot=None,
                     rho_range=None, rho_dot_range=None, observer_positions=None, observer_velocities=None,
                     obs_indices=None, obs_epochs=None, delta_rho=None, delta_rho_dot=None):
            """
               Initialize a structured basin-hopping stepper constrained by the admissible region.

               This constructor sets up the parameters, observation data, and matrices needed to
               propose admissible steps in the solution space of an orbit determination problem.

               Args:
                   q (int, optional): Dimensionality of position/velocity vectors (default: 3).
                   H_size (int, optional): Basis vector size in the parameter space (default: 10).
                   observations (torch.Tensor, optional): Observation matrix of shape (N,4), with
                       columns [sin(alpha), cos(alpha), sin(delta), cos(delta)] for each epoch.
                   pos_step (float, optional): Position step size scale (not directly used here).
                   vel_step (float, optional): Velocity step size scale (not directly used here).
                   H (torch.Tensor, optional): Design matrix mapping beta to position (shape: [N, H_size]).
                   cH_dot (torch.Tensor, optional): Design matrix mapping beta to velocity (shape: [N, H_size]).
                   rho_range (tuple, optional): Tuple (rho_min, rho_max) defining uniform sampling range
                       for line-of-sight distance perturbations.
                   rho_dot_range (tuple, optional): Tuple (rho_dot_min, rho_dot_max) defining uniform sampling range
                       for line-of-sight rate perturbations.
                   observer_positions (torch.Tensor, optional): Observer positions at each epoch (shape: [N,3]).
                   observer_velocities (torch.Tensor, optional): Observer velocities at each epoch (shape: [N,3]).
                   obs_indices (torch.Tensor or list, optional): Indices of epochs to extract from H/cH_dot.
                   obs_epochs (torch.Tensor, optional): Times of observations (shape: [N]).
               """

            self.obs = observations
            self.q = q
            self.H_size = H_size
            self.H = H
            self.cH_dot = cH_dot
            self.rho_range = rho_range
            self.rho_dot_range = rho_dot_range
            self.observer_positions = observer_positions
            self.observer_velocities = observer_velocities
            self.obs_indices = obs_indices
            self.obs_epochs = obs_epochs
            self.delta_rho_step = delta_rho
            self.delta_rho_dot_step = delta_rho_dot

        def __call__(self, x):
            """
                Callable interface for proposing a new admissible step.

                This method allows instances of the class to be called like a function,
                and simply delegates to `take_step()`.

                Args:
                    x (torch.Tensor): Current parameter vector `beta`, shape (q * H_size,).

                Returns:
                    torch.Tensor: Proposed new parameter vector `beta_new`, shape (q * H_size,).
                """
            return self.take_step(x)

        def take_step(self, beta):
            """
            Propose a new admissible step in the solution space.

            This method perturbs the current parameter vector `beta` by sampling
            line-of-sight distance (`rho`) and rate (`rho_dot`) uniformly at each epoch
            within the specified admissible ranges. It computes the resulting position and
            velocity vectors consistent with the observed right ascension (`alpha`) and declination (`delta`),
            plus their best-fit angular rates, and projects the perturbed state back into
            the parameter space via the pseudoinverse of the design matrix.

            Args:
                beta (torch.Tensor): Current parameter vector, shape (q * H_size,).

            Returns:
                torch.Tensor: Proposed parameter vector after admissible perturbation,
                    shape (q * H_size,).
            """

            # transform beta to current r and v
            beta_tensor = beta.reshape(self.q, self.H_size)
            r = (self.H @ beta_tensor.T)[self.obs_indices]  # (N_obs, q)
            v = (self.cH_dot @ beta_tensor.T)[self.obs_indices]  # (N_obs, q)

            # unpack obs
            sin_ra = self.obs[0]
            cos_ra = self.obs[1]
            sin_dec = self.obs[2]
            cos_dec = self.obs[3]

            alpha = torch.atan2(sin_ra, cos_ra)

            def torch_unwrap(p, discont=np.pi, dim=-1):
                diff = torch.diff(p, dim=dim)
                diff_mod = (diff + np.pi) % (2 * np.pi) - np.pi
                diff_mod = torch.where((diff_mod == -np.pi) & (diff > 0), np.pi, diff_mod)
                p0 = torch.index_select(p, dim, torch.tensor([0], device=p.device))
                return torch.cat([p0, p0 + torch.cumsum(diff_mod, dim=dim)], dim=dim)

            alpha = torch_unwrap(alpha, dim=0)
            delta = torch.atan2(sin_dec, cos_dec)

            # compute basis vectors, shape (N,3)
            l = torch.stack([
                cos_ra * cos_dec,
                sin_ra * cos_dec,
                sin_dec
            ], dim=-1)

            l_alpha = torch.stack([
                -sin_ra * cos_dec,
                cos_ra * cos_dec,
                torch.zeros_like(cos_dec)
            ], dim=-1)

            l_delta = torch.stack([
                -cos_ra * sin_dec,
                -sin_ra * sin_dec,
                cos_dec
            ], dim=-1)

            def fit_angular_rates(t, alpha, delta):
                # Convert Time object to seconds (or days, or any consistent unit)
                t = Time(t)  # convert list/array of Time objects into a Time array
                t_sec = (t - t[0]).to_value('s')  # TimeDelta → float seconds

                t0 = t_sec.mean()
                dt = t_sec - t0

                alpha_mean = alpha.mean()
                delta_mean = delta.mean()

                dalpha = ((alpha - alpha_mean) * dt).sum() / (dt * dt).sum()
                ddelta = ((delta - delta_mean) * dt).sum() / (dt * dt).sum()

                return dalpha, ddelta

            alpha_dot, delta_dot = fit_angular_rates(self.obs_epochs, alpha, delta)

            # observer positions & velocities (N,3)
            L = configuration['AU_TO_M'] / configuration['KM_TO_M']  # Length scale in km
            G_km = (configuration['GRAVITATIONAL_CONSTANT'] / configuration['KM_TO_M'] ** 3)
            sys_mass = configuration['SUN_MASS'] + configuration['MOON_MASS'] + configuration['EARTH_MASS']
            T = np.sqrt(L ** 3 / (G_km * sys_mass))
            observer_pos = self.observer_positions / L
            observer_vel = self.observer_velocities / L * T

            # LOS relative to observer
            r_rel = r - observer_pos  # (N,3)

            # LOS projection per epoch (N,)
            rho_base = torch.sum(r_rel * l, dim=1)  # (N,)

            # Sample perturbation in range space per epoch
            delta_rho = (torch.rand(rho_base.shape, device=r.device) * 2 - 1.0) * self.delta_rho_step  # (N,)
            rho_min, rho_max = self.rho_range
            rho = torch.clamp(rho_base + delta_rho, min=rho_min, max=rho_max)  # (N,)

            # Perturbation vector in r-space (N,3)
            delta_r = (rho - rho_base)[:, None] * l  # (N,3)

            # Perturbed candidate positions (geocentric)
            r_obs = r + delta_r  # (N,3)

            # For velocities
            v_rel = v - observer_vel  # (N,3)
            rho_dot_base = torch.sum(v_rel * l, dim=1)  # (N,)

            # Sample rho_dot per epoch
            delta_rho_dot = (torch.rand(rho_dot_base.shape, device=v.device) * 2 - 1.0) * self.delta_rho_dot_step
            rho_dot = torch.clamp(rho_dot_base + delta_rho_dot, min=self.rho_dot_range[0],
                                  max=self.rho_dot_range[1])  # (N,)


            # Full velocity reconstruction:
            v_obs = (observer_vel +
                    rho_dot[:, None] * l
                    + rho[:, None] * (alpha_dot * l_alpha + delta_dot * l_delta)
            )

            all_H = torch.cat([self.H, self.cH_dot], dim=1)  # (N,2H_size)
            H_obs = all_H[self.obs_indices]  # (N_obs, 2H_size)

            H_pos = H_obs[:, :self.H_size]  # (N_obs, H_size)
            H_vel = H_obs[:, self.H_size:]  # (N_obs, H_size)

            # stack and solve
            H_stacked = torch.cat([H_pos, H_vel], dim=0)  # (2N_obs, H_size)
            rv_stacked_tensor = torch.cat([r_obs, v_obs], dim=0).float()  # (2N_obs, 3)

            beta_best = torch.linalg.pinv(H_stacked) @ rv_stacked_tensor  # (H_size, 3)

            return beta_best.view(-1)

        def take_initial_step(self, n_init=1):
            """
            Sample n_init initial trajectories and pick the one
            minimizing Jacobi constant variation.
            """

            # unpack obs
            sin_ra = self.obs[0]
            cos_ra = self.obs[1]
            sin_dec = self.obs[2]
            cos_dec = self.obs[3]

            alpha = torch.atan2(sin_ra, cos_ra)

            def torch_unwrap(p, discont=np.pi, dim=-1):
                diff = torch.diff(p, dim=dim)
                diff_mod = (diff + np.pi) % (2 * np.pi) - np.pi
                diff_mod = torch.where((diff_mod == -np.pi) & (diff > 0), np.pi, diff_mod)
                p0 = torch.index_select(p, dim, torch.tensor([0], device=p.device))
                return torch.cat([p0, p0 + torch.cumsum(diff_mod, dim=dim)], dim=dim)

            alpha = torch_unwrap(alpha, dim=0)
            delta = torch.atan2(sin_dec, cos_dec)

            # basis vectors (N,3)
            l = torch.stack([
                cos_ra * cos_dec,
                sin_ra * cos_dec,
                sin_dec
            ], dim=-1)

            l_alpha = torch.stack([
                -sin_ra * cos_dec,
                cos_ra * cos_dec,
                torch.zeros_like(cos_dec)
            ], dim=-1)

            l_delta = torch.stack([
                -cos_ra * sin_dec,
                -sin_ra * sin_dec,
                cos_dec
            ], dim=-1)

            def fit_angular_rates(t, alpha, delta):
                # Convert Time object to seconds (or days, or any consistent unit)
                t = Time(t)  # convert list/array of Time objects into a Time array
                t_sec = (t - t[0]).to_value('s')  # TimeDelta → float seconds

                t0 = t_sec.mean()
                dt = t_sec - t0

                alpha_mean = alpha.mean()
                delta_mean = delta.mean()

                dalpha = ((alpha - alpha_mean) * dt).sum() / (dt * dt).sum()
                ddelta = ((delta - delta_mean) * dt).sum() / (dt * dt).sum()

                return dalpha, ddelta

            alpha_dot, delta_dot = fit_angular_rates(self.obs_epochs, alpha, delta)

            # observer pos & vel: (N,3)
            L = configuration['AU_TO_M'] / configuration['KM_TO_M']  # Length scale in km
            G_km = (configuration['GRAVITATIONAL_CONSTANT'] / configuration['KM_TO_M'] ** 3)
            sys_mass = configuration['SUN_MASS'] + configuration['MOON_MASS'] + configuration['EARTH_MASS']
            T = np.sqrt(L ** 3 / (G_km * sys_mass))
            observer_pos = self.observer_positions / L
            observer_vel = self.observer_velocities * T / L

            # sample n_init central candidates: (n_init,1)
            rho_0_central = (
                    torch.rand(n_init, 1) * (self.rho_range[1] - self.rho_range[0]) + self.rho_range[0]
            )  # (n_init,1)

            rho_dot_0_central = (
                    torch.rand(n_init, 1) * (self.rho_dot_range[1] - self.rho_dot_range[0]) + self.rho_dot_range[0]
            )  # (n_init,1)

            # expand to (n_init,N,1) with perturbations
            N = l.shape[0]

            rho_0 = rho_0_central[:, None, :]

            rho_dot_0 = rho_dot_0_central[:, None, :]

            # expand basis vectors to (1,N,3)
            l = l[None, :, :]
            l_alpha = l_alpha[None, :, :]
            l_delta = l_delta[None, :, :]
            observer_pos = observer_pos[None, :, :]
            observer_vel = observer_vel[None, :, :]

            # compute r_0 and v_0: (n_init,N,3)
            r_0 = observer_pos + rho_0 * l
            v_0 = (
                    observer_vel +
                    rho_dot_0 * l +
                    rho_0 * alpha_dot * l_alpha +
                    rho_0 * delta_dot * l_delta
            )

            # compute Jacobi for each (n_init,N)
            def compute_jacobi(r_0, v_0):
                mu = configuration['SYSTEM_MASS_PARAMETER']

                r_bary = r_0
                v_bary = v_0
                x = r_bary[:, :, 0]
                y = r_bary[:, :, 1]
                z = r_bary[:, :, 2]
                vx = v_bary[:, :, 0]
                vy = v_bary[:, :, 1]
                vz = v_bary[:, :, 2]
                r_2 = torch.sqrt((mu + x - 1) ** 2 + y ** 2 + z ** 2)
                r_1 = torch.sqrt((mu + x) ** 2 + y ** 2 + z ** 2)
                omega = 0.5 * (x ** 2 + y ** 2) + (1 - mu) / r_1 + mu / r_2
                energy = 2 * omega - (vx ** 2 + vy ** 2 + vz ** 2)
                return energy

            jacobi = compute_jacobi(r_0, v_0)

            # variance of Jacobi over timesteps, per candidate
            jacobi_var = torch.var(jacobi, dim=1)

            # pick candidate with smallest variance
            best_idx = torch.argmin(jacobi_var)
            r_best = r_0[best_idx]  # (N,3)
            v_best = v_0[best_idx]  # (N,3)

            r_obs = r_best
            v_obs = v_best

            all_H = torch.cat([self.H, self.cH_dot], dim=1)  # (N,2H_size)
            H_obs = all_H[self.obs_indices]  # (N_obs, 2H_size)

            H_pos = H_obs[:, :self.H_size]  # (N_obs, H_size)
            H_vel = H_obs[:, self.H_size:]  # (N_obs, H_size)

            # stack and solve
            H_stacked = torch.cat([H_pos, H_vel], dim=0)  # (2N_obs, H_size)
            rv_stacked = torch.cat([r_obs, v_obs], dim=0)  # (2N_obs, 3)

            beta_best = torch.linalg.pinv(H_stacked) @ rv_stacked  # (H_size, 3)

            global_positions.append(r_best)

            return beta_best.view(-1)


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

    def precompute_body_states(epochs, config, L, sys_mass):
        """
        Precompute other-body positions and nondimensional masses.

        Parameters
        ----------
        epochs : list[astropy.time.Time]
            Epochs to query (N,).
        config : dict
            Dict containing physical constants and body masses.
        L : float
            Length scale for nondimensionalization.
        sys_mass : float
            Total system mass.

        Returns
        -------
        positions_nd : torch.Tensor
            (N, n, 3) nondimensional positions.
        mus_nd : torch.Tensor
            (n,) nondimensional body gravitational parameters.
        """
        bodies = [10, 1, 2, 399, 4, 5, 6, 7, 8, 301]
        names = ['SUN', 'MERCURY', 'VENUS', 'EARTH', 'MARS',
                 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'MOON']
        masses = np.array([config[f'{name}_MASS'] for name in names])

        N, n = len(epochs), len(bodies)
        positions = np.zeros((N, n, 3))

        # Convert epochs to ET
        epoch_ets = [spice.unitim(t.tdb.jd, 'JDTDB', 'ET') for t in epochs]

        for i, et in enumerate(epoch_ets):
            for j, body in enumerate(bodies):
                state, _ = spice.spkgeo(targ=body, et=et, ref='J2000', obs=399)
                positions[i, j, :] = state[:3]  # km

        # Nondimensionalize positions
        positions_nd = torch.tensor(positions / L, dtype=torch.float32)

        # Nondimensionalize μ
        G_km = (config['GRAVITATIONAL_CONSTANT'] / config['KM_TO_M'] ** 3)
        mus = G_km * masses
        mus_nd = torch.tensor(mus / (G_km * sys_mass), dtype=torch.float32)

        return positions_nd, mus_nd
    non_dim_positions, non_dim_grav_params = precompute_body_states(colloc_epochs, configuration, L, sys_mass)


    # === Residual Function for Least Squares ===
    def loss_function(beta_flat):
        beta_tensor = beta_flat.view(q, H_size)

        Y_pred = H_matrix @ beta_tensor.T  # (N, q)
        Y_dot_pred = c * H_dot @ beta_tensor.T

        Y_pred_obs = Y_pred[obs_indices]
        Y_pred_obs_km = Y_pred_obs * L
        Y_dot_pred_obs = Y_dot_pred[obs_indices] * L / T

        if viz_flag:
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
            obs_res = torch.mean((Y_pred_ang - Y_obs) ** 2)

            return obs_res

        obs_residual = ra_dec_observation_residual()  # (N * q)

        Y_ddot_pred = c ** 2 * (H_ddot @ beta_tensor.T)  # (N, q)

        def nbody_physics_residual(Y_preds, Y_ddot_preds, positions_nd, mus_nd):
            """
            Compute residual between predicted and n-body accelerations.

            Parameters
            ----------
            Y_preds : torch.Tensor
                (N, 3) — nondimensional predicted positions of asteroid (geocentric).
            Y_ddot_preds : torch.Tensor
                (N, 3) — nondimensional predicted accelerations of asteroid.
            positions_nd : torch.Tensor
                (N, n, 3) — nondimensional positions of other bodies (geocentric).
            mus_nd : torch.Tensor
                (n,) — nondimensional gravitational parameters of other bodies.
            L : float
                Length scale (e.g. Earth–Sun distance in km).
            sys_mass : float
                Total system mass (for nondimensionalization).
            """

            N, n, _ = positions_nd.shape
            Y_expanded = Y_preds[:, None, :]  # (N, 1, 3)

            r_vecs = positions_nd - Y_expanded  # (N, n, 3)
            r_norms = torch.norm(r_vecs, dim=-1, keepdim=True)  # (N, n, 1)

            accel_terms = mus_nd[None, :, None] * r_vecs / (r_norms ** 3)  # (N, n, 3)
            total_accel = accel_terms.sum(dim=1)  # (N, 3)

            return torch.mean((Y_ddot_preds - total_accel) ** 2)

        physics_residual = nbody_physics_residual(Y_pred, Y_ddot_pred, non_dim_positions, non_dim_grav_params)


        def distance_penalty(Y_predicted):
            return torch.mean(torch.sum(Y_predicted ** 2, dim=1))

        weighted_dist_res = lambda_dist * distance_penalty(Y_pred)

        if viz_flag:
            data_losses.append(obs_residual.item())
            physics_losses.append(lambda_phys * physics_residual.item())
            range_losses.append(weighted_dist_res.item())

        return obs_residual + lambda_phys * physics_residual + weighted_dist_res

    def func(beta_flat):
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
        observer_velocities=observer_velocity,
        obs_indices=obs_indices,
        obs_epochs=colloc_epochs[obs_indices],
        delta_rho=parameters['DELTA_RHO'],
        delta_rho_dot=parameters['DELTA_RHO_DOT'],
    )

    beta0 = mystep.take_initial_step()

    # Basin hopping configuration
    options = {"ftol": parameters['F_TOLERANCE'], "gtol": parameters['G_TOLERANCE'],
               "maxfun": parameters['MAX_FUNCTION_EVAL'],
               "maxiter": parameters['MAX_ITERATiONS'], "disp": False}
    minimizer_kwargs = {"method": "L-BFGS-B", "jac": True, "options": options}  # Local optimizer

    def callback(x, f, accept):
        # increment global iteration
        global_its[0] += 1

        # track best solution
        if f < best['fun']:
            best['fun'] = f
            best['x'] = x
            best['iteration'] = global_its[0]

    start = time.time()
    # Run basin hopping
    res2 = basinhopping(func, beta0, minimizer_kwargs=minimizer_kwargs, niter=parameters['NUMBER_OF_ITERATIONS'],
                        T=parameters['TEMPERATURE'], callback=callback, take_step=mystep, disp=False)
    end = time.time()

    beta_tensor_nlls = np.asarray(res2.x).reshape(q, H_size)
    Y_pred_nlls = H_matrix @ beta_tensor_nlls.T  # (N, q)
    Y_dot_pred_nlls = c * H_dot @ beta_tensor_nlls.T

    Y_pred_obs_nlls = Y_pred_nlls[obs_indices]
    Y_dot_pred_obs_nlls = Y_dot_pred_nlls[obs_indices]
    if viz_flag:
        positions[-1] = ((Y_pred_obs_nlls * L).detach().cpu().numpy())
        velocities[-1] = (Y_dot_pred_obs_nlls * L / T).detach().cpu().numpy()
    # print(res.x)

    # for error measurement purposes
    initial_colloc_epoch = colloc_epochs[0]
    final_colloc_epoch = colloc_epochs[-1]

    initial_obs_epoch = colloc_epochs[obs_indices][0]
    final_obs_epoch = colloc_epochs[obs_indices][-1]

    num_points = configuration['error']['num_points']
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

    if viz_flag:
        epochss = np.arange(total_its[0])
        data = {"TRAINING_EPOCH": epochss, "DATA_LOSS": data_losses, "PHYSICS_LOSS": physics_losses, "RANGE_LOSS": range_losses}
        return pd.DataFrame(data), positions, velocities, nlls_start[0], final_positions, final_velocities, end - start, best['iteration']
    else:
        return [], [], [], nlls_start[0], final_positions, final_velocities, end - start, best['iteration']
