import spiceypy as spice
from scipy.integrate import solve_ivp
import yaml
import argparse
import numpy as np

# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')


class NBodyPropagator:
    """
    General N-body propagator for one or many independent 6D states (km, km/s)
    in Earth-centered EME/J2000.

    Conventions:
      - Epoch inputs: JDTDB (float)
      - Internal time: ET seconds (SPICE)
      - Frame: "J2000" (SPICE inertial ≈ EME/J2000)
      - ORIGIN: Earth-centered => origin NAIF ID 399

    Dynamics:
      - Propagates the object(s) under point-mass gravity from selected bodies.
      - Perturbing body states are pulled from SPICE at integration time.

    Outputs:
      - If t1 is scalar -> returns (6,) for propagate(), and (Ns,6) for propagate_multiple_objects()
      - If t1 is array-like of length K -> returns (K,6) for propagate(), and (K,Ns,6) for propagate_multiple_objects()
        (time-major)
    """

    def __init__(
        self,
        *,
        spice,                   # spiceypy module (kernels already furnished)
        config: dict,            # masses + G + KM_TO_M
        bodies=('10', '1', '2', '399', '4', '5', '6', '7', '8', '301'),
        frame="J2000",
        origin="399",              # Earth-centered
        eps=1e-12,
        rtol=1e-10,
        atol=1e-12,
        method="RK45",
    ):
        self.spice = spice
        self.config = config
        self.bodies = tuple(bodies)
        self.frame = str(frame)
        self.origin = str(origin)
        self.eps = float(eps)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.method = str(method)

        self.KM_TO_M = float(config.get("KM_TO_M", 1000.0))
        self.G = float(config["GRAVITATIONAL_CONSTANT"])  # m^3 kg^-1 s^-2

        self.mass_map = {
            "10": float(config["SUN_MASS"]),
            "1": float(config["MERCURY_MASS"]),
            "2": float(config["VENUS_MASS"]),
            "399": float(config["EARTH_MASS"]),
            "4": float(config["MARS_MASS"]),
            "5": float(config["JUPITER_MASS"]),
            "6": float(config["SATURN_MASS"]),
            "7": float(config["URANUS_MASS"]),
            "8": float(config["NEPTUNE_MASS"]),
            "301": float(config["MOON_MASS"]),
        }

        # Ensure bodies are strings that match the dict keys
        self.bodies = [str(b) for b in self.bodies]

        self.masses = np.array([self.mass_map[b] for b in self.bodies], dtype=float)

        # μ = G*M (m^3/s^2), keyed by body id (string)
        self.mu_by_id = {b: self.G * self.mass_map[b] for b in self.mass_map}

    # ----------------------------
    # Time conversion (JDTDB -> ET)
    # ----------------------------
    def jdtdb_to_et(self, jdtdb: float) -> float:
        return float(self.spice.unitim(float(jdtdb), "JDTDB", "ET"))

    # ----------------------------
    # Acceleration model
    # ----------------------------
    def _body_positions_m(self, et: float) -> np.ndarray:
        """
        Positions of perturbing bodies relative to Earth (origin=399), in meters.
        Shape: (Nb,3)
        """
        Nb = len(self.bodies)
        out = np.zeros((Nb, 3), dtype=float)
        for i, bid in enumerate(self.bodies):
            r_km, _ = self.spice.spkpos(bid, et, self.frame, "NONE", self.origin)
            out[i] = np.asarray(r_km, dtype=float) * self.KM_TO_M
        return out

    def _accel_geo_m_s2(self, et: float, r_obj_m: np.ndarray) -> np.ndarray:
        """
        Acceleration of the object in an Earth-centered inertial frame (origin=399),
        including Earth central gravity + differential 3rd-body terms.
        """
        r = np.asarray(r_obj_m, dtype=float).reshape(3, )
        rmag = np.linalg.norm(r)

        a = np.zeros(3, dtype=float)

        # --- Earth central gravity ---
        mu_earth = self.mu_by_id['399']  # [m^3/s^2]
        if rmag > self.eps:
            a += -mu_earth * r / (rmag ** 3)

        # --- Third-body differential terms (Sun, Moon, planets, etc.) ---
        for bid in self.bodies:
            if bid == '399':
                continue

            # r_j : body position relative to Earth, meters
            rj_km, _ = self.spice.spkpos(bid, et, self.frame, "NONE", "399")
            rj = np.asarray(rj_km, dtype=float) * self.KM_TO_M

            mu = self.mu_by_id[bid]  # [m^3/s^2]

            # term1: attraction of object toward body
            rel = rj - r
            relmag = np.linalg.norm(rel)
            if relmag > self.eps:
                a += mu * rel / (relmag ** 3)

            # term2: subtract attraction of Earth toward body (indirect term)
            rjmag = np.linalg.norm(rj)
            if rjmag > self.eps:
                a += -mu * rj / (rjmag ** 3)

        return a

    # ----------------------------
    # Single-object propagation
    # ----------------------------
    def propagate(self, x0_km: np.ndarray, t0_jdtdb: float, t1_jdtdb):
        """
        Propagate one 6D state (km, km/s) from t0 to t1 in Earth-centered J2000.

        If t1_jdtdb is scalar -> returns x1 (6,)
        If t1_jdtdb is array-like (K,) -> returns X (K,6) evaluated at each epoch
        """
        x0 = np.asarray(x0_km, dtype=float).reshape(6,)
        t1_arr = np.asarray(t1_jdtdb, dtype=float).ravel()
        scalar = (t1_arr.size == 1)

        et0 = self.jdtdb_to_et(t0_jdtdb)
        et1s = np.array([self.jdtdb_to_et(t) for t in t1_arr], dtype=float)

        if np.any(et1s < et0 - 1e-12):
            raise ValueError("t1 epochs must be >= t0 (non-decreasing).")
        if et1s.size > 1 and np.any(np.diff(et1s) < -1e-12):
            raise ValueError("t1 epoch series must be monotonic non-decreasing.")

        # initial in meters
        y0 = np.hstack([x0[:3] * self.KM_TO_M, x0[3:] * self.KM_TO_M])

        def dyn(et, y):
            r = y[:3]
            v = y[3:]
            a = self._accel_geo_m_s2(et, r)
            return np.hstack([v, a])

        et_end = float(et1s[-1])
        if abs(et_end - et0) < 1e-15:
            if scalar:
                return x0.copy()
            return np.broadcast_to(x0, (et1s.size, 6)).copy()

        t_eval = None
        if et1s.size > 1:
            t_eval = et1s

        sol = solve_ivp(
            dyn,
            (et0, et_end),
            y0,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
        )

        if not sol.success:
            raise RuntimeError(f"Propagation failed: {sol.message}")

        if et1s.size == 1:
            y1 = sol.y[:, -1]
            x1 = np.hstack([y1[:3] / self.KM_TO_M, y1[3:] / self.KM_TO_M])
            return x1

        Y = sol.y.T  # (K,6) in meters/meters/s
        X = np.hstack([Y[:, :3] / self.KM_TO_M, Y[:, 3:] / self.KM_TO_M])
        return X

    # ----------------------------
    # Many-object propagation (general)
    # ----------------------------
    def propagate_multiple_objects(
        self,
        X0_km: np.ndarray,
        t0_jdtdb: float,
        t1_jdtdb,
    ) -> np.ndarray:
        """
        Propagate multiple independent objects.

        Inputs:
          - X0_km: (Ns,6) initial states in km, km/s
          - t0_jdtdb: scalar epoch (JDTDB)
          - t1_jdtdb: scalar epoch OR array-like of K epochs (JDTDB)

        Returns:
          - if t1_jdtdb is scalar:
                X1: (Ns,6)
          - if t1_jdtdb is array-like with K epochs:
                X:  (K, Ns, 6)   (time-major)
        """
        X0_km = np.asarray(X0_km, dtype=float)
        if X0_km.ndim != 2 or X0_km.shape[1] != 6:
            raise ValueError(f"X0_km must be (Ns,6), got {X0_km.shape}")

        t1_arr = np.asarray(t1_jdtdb, dtype=float).ravel()
        scalar = (t1_arr.size == 1)

        Ns = X0_km.shape[0]

        if scalar:
            t1s = float(t1_arr[0])
            X_out = np.zeros((Ns, 6), dtype=float)
            for i in range(Ns):
                X_out[i] = self.propagate(X0_km[i], t0_jdtdb, t1s)
            return X_out

        K = int(t1_arr.size)
        X_out = np.zeros((K, Ns, 6), dtype=float)  # (time, obj, state)
        for i in range(Ns):
            Xi = self.propagate(X0_km[i], t0_jdtdb, t1_arr)  # (K,6)
            if Xi.shape != (K, 6):
                raise RuntimeError(f"Expected propagate() to return (K,6), got {Xi.shape}")
            X_out[:, i, :] = Xi
        return X_out



def integrate_n_body(object_state, epoch, end_time, time_interval, type):
    # Argument parser to get the config file path
    parser = argparse.ArgumentParser(description="Run the spacecraft simulation")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    bodies = [10, 1, 2, 399, 4, 5, 6, 7, 8,
              301]  # ["SUN", "MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE", "MOON"]

    masses = {
        10: config['SUN_MASS'], 1: config['MERCURY_MASS'], 2: config['VENUS_MASS'],
        399: config['EARTH_MASS'], 4: config['MARS_MASS'], 5: config['JUPITER_MASS'], 6: config['SATURN_MASS'],
        7: config['URANUS_MASS'], 8: config['NEPTUNE_MASS'], 301: config['MOON_MASS'],
        "ASTEROID": config['asteroid_mass'], "SPACECRAFT": config['mass']  # Arbitrary mass
    }


    if type == "ASTEROID":
        epoch_et = spice.unitim(epoch, 'JDTDB', 'ET')  # initial epoch
        mass_array = np.array([masses[body] for body in bodies] + [masses["ASTEROID"]])
    elif type == "SPACECRAFT-ASTEROIDTIME":
        epoch_et = spice.unitim(epoch, 'JDTDB', 'ET')  # initial epoch
        mass_array = np.array([masses[body] for body in bodies] + [masses["SPACECRAFT"]])
    else:
        epoch_et = spice.str2et(epoch)
        mass_array = np.array([masses[body] for body in bodies] + [masses["SPACECRAFT"]])

    # Function to get state vectors (position, velocity) in km & km/s
    def get_state(body, reference=10):
        state, _ = spice.spkgeo(body, epoch_et, "ECLIPJ2000", reference)
        return np.array(state)

    # Get Sun & planets' initial states
    planet_states = {body: get_state(body) for body in bodies}

    # Combine all bodies into a state vector
    initial_states = np.vstack([planet_states[body] for body in bodies] + [object_state])
    initial_positions = initial_states[:, :3]
    initial_velocities = initial_states[:, 3:]

    # Convert km, km/s to meters, meters/s
    initial_positions *= config['KM_TO_M']
    initial_velocities *= config['KM_TO_M']

    # Flatten initial state vector (for integration)
    y0 = np.hstack([initial_positions.flatten(), initial_velocities.flatten()])

    # Define masses (in kg) for Sun, planets, and Moon
    G = config['GRAVITATIONAL_CONSTANT']  # m^3 kg^-1 s^-2

    start_time = 0
    t_span = (start_time, end_time)  # Start at t=0, end at t=900s
    t_eval = np.arange(start_time, end_time, time_interval)  # 30s intervals

    # Define N-body equations of motion
    def nbody_derivatives(t, y):
        n = len(mass_array)
        positions = y[:3 * n].reshape((n, 3))
        velocities = y[3 * n:].reshape((n, 3))
        accelerations = np.zeros((n, 3))

        min_r = np.inf
        min_pair = None

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    if r_mag < min_r:
                        min_r = r_mag
                        min_pair = (i, j)
                    accelerations[i] += G * mass_array[j] * r_vec / r_mag ** 3

        if min_r < 1e6:  # 1000 km, adjust threshold
            print("t=", t, "min_r(m)=", min_r, "pair=", min_pair)

        return np.hstack([velocities.flatten(), accelerations.flatten()])

    # Solve the N-body problem
    sol = solve_ivp(nbody_derivatives, t_span, y0, method="DOP853", t_eval=t_eval)

    # Extract asteroid's trajectory
    n_bodies = len(mass_array)
    object_idx = n_bodies - 1  # asteroid index

    n = n_bodies
    pos0 = 0
    vel0 = 3 * n

    earth_i = bodies.index(399)  # should be 3
    earth_positions = sol.y[pos0 + 3 * earth_i: pos0 + 3 * (earth_i + 1), :]
    earth_velocities = sol.y[vel0 + 3 * earth_i: vel0 + 3 * (earth_i + 1), :]
    object_positions = sol.y[3 * object_idx: 3 * (object_idx + 1), :]
    object_velocities = sol.y[2 * 3 * object_idx + 3: 2 * 3 * object_idx + 6, :]

    return np.vstack((object_positions, object_velocities)) / config['KM_TO_M'], np.vstack((earth_positions, earth_velocities)) / config['KM_TO_M']


import numpy as np
from astropy.time import TimeDelta
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

def two_body_integrator(r0_km, v0_kms, epoch, timestep_sec, num_frames):
    """
    Propagate a state under two-body dynamics.

    Parameters:
    ----------
    r0_km : array_like
        Initial position vector [x, y, z] in km.
    v0_kms : array_like
        Initial velocity vector [vx, vy, vz] in km/s.
    epoch : astropy.time.Time
        Initial epoch of the orbit.
    timestep_sec : float
        Time step between frames in seconds.
    num_frames : int
        Number of frames to propagate.

    Returns:
    -------
    positions : np.ndarray
        Array of propagated positions of shape (num_frames, 3) in km.
    velocities : np.ndarray
        Array of propagated velocities of shape (num_frames, 3) in km/s.
    epochs : list of astropy.time.Time
        List of epochs corresponding to each frame.
    """
    # Create initial orbit
    r0 = np.array(r0_km, dtype=np.float64) * u.km
    v0 = np.array(v0_kms, dtype=np.float64) * u.km / u.s
    orbit = Orbit.from_vectors(Earth, r0, v0, epoch)

    # Time steps
    epochs = [epoch + TimeDelta(i * timestep_sec, format='sec') for i in range(num_frames)]

    # Propagate and collect
    positions = []
    velocities = []

    for t in epochs:
        propagated = orbit.propagate(t - epoch)
        positions.append(propagated.r.to_value(u.km))
        velocities.append(propagated.v.to_value(u.km / u.s))

    return np.array(positions), np.array(velocities), epochs


# to be used with odeint
def cr3bp(state, time, mu=0.01215):
    # Define the dynamics of the system
    # state: current state vector
    # time: current time
    # return: derivative of the state vector

    x, y, z, vx, vy, vz = state[:6]  # position and velocity
    phi = np.reshape(state[6:], (6, 6))

    dUdx = -(mu * (mu + x - 1))/np.power(((mu + x - 1)**2 + y**2 + z**2), (3/2)) - \
           ((1 - mu) * (mu + x))/np.power(((mu + x)**2 + y**2 + z**2),(3/2)) + x
    dUdy = - (mu * y)/np.power(((mu + x - 1)**2 + y**2 + z**2), (3/2)) - \
           ((1 - mu) * y)/np.power(((mu + x)**2 + y**2 + z**2), (3/2)) + y
    dUdz = - (mu * z)/np.power(((mu + x - 1)**2 + y**2 + z**2), (3/2)) - \
           ((1 - mu) * z)/np.power(((mu + x)**2 + y**2 + z**2), (3/2))

    dxdt = vx  # derivative of position is velocity
    dydt = vy
    dzdt = vz
    dvxdt = dUdx + 2*dydt  # derivative of velocity is acceleration
    dvydt = dUdy - 2*dxdt
    dvzdt = dUdz

    dXdt = np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])

    def gen_F_matrix(x, y, z, mu):
        """

        :param x: current x
        :param y: current y
        :param z: current z
        :param mu: gravitional parameter
        :return: the F matrix from Howells method, to update the state transition matrix
        """

        F = np.zeros((6, 6))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 3:6] = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])

        # Second order partials
        U_xx = (mu - 1) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** 1.5000 - mu / (
                    (mu + x - 1) ** 2 + y ** 2 + z ** 2) ** 1.5000 + \
               (0.7500 * mu * (2 * x + 2 * mu - 2) ** 2) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** 2.5000 - \
               (0.7500 * (2 * x + 2 * mu) ** 2 * (mu - 1)) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** 2.5000 + 1
        U_yy = (mu - 1) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** 1.5000 - mu / (
                    (mu + x - 1) ** 2 + y ** 2 + z ** 2) ** 1.5000 + \
               (3 * mu * y ** 2) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** 2.5000 - \
               (3 * y ** 2 * (mu - 1)) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** 2.5000 + 1
        U_zz = (mu - 1) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** 1.5000 - mu / (
                    (mu + x - 1) ** 2 + y ** 2 + z ** 2) ** 1.5000 + \
               (3 * mu * z ** 2) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** 2.5000 - (3 * z ** 2 * (mu - 1)) / (
                           (mu + x) ** 2 + y ** 2 + z ** 2) ** 2.5000
        U_xy = (1.5000 * mu * y * (2 * x + 2 * mu - 2)) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** 2.5000 - \
               (1.5000 * y * (2 * x + 2 * mu) * (mu - 1)) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** 2.5000
        U_xz = (1.5000 * mu * z * (2 * x + 2 * mu - 2)) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** 2.5000 - \
               (1.5000 * z * (2 * x + 2 * mu) * (mu - 1)) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** 2.5000
        U_yz = (3 * mu * y * z) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** 2.5000 - \
               (3 * y * z * (mu - 1)) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** 2.5000
        U_yx = U_xy
        U_zx = U_xz
        U_zy = U_yz

        F[3:6, 0:3] = np.array([[U_xx, U_xy, U_xz], [U_yx, U_yy, U_zy], [U_zx, U_zy, U_zz]])

        return F
    F = gen_F_matrix(x, y, z, mu)

    dphidt = np.matmul(F, phi)

    return np.hstack((np.array(dXdt), dphidt.ravel()))
