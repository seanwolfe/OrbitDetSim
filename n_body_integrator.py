import spiceypy as spice
from scipy.integrate import solve_ivp
import yaml
import argparse

# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')


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
        "ASTEROID": config['asteroid_mass'], "SPACECRAFT": config['spacecraft_mass']  # Arbitrary mass
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

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    accelerations[i] += G * mass_array[j] * r_vec / r_mag ** 3

        return np.hstack([velocities.flatten(), accelerations.flatten()])

    # Solve the N-body problem
    sol = solve_ivp(nbody_derivatives, t_span, y0, method="RK45", t_eval=t_eval)

    # Extract asteroid's trajectory
    n_bodies = len(mass_array)
    object_idx = n_bodies - 1  # asteroid index

    earth_positions = sol.y[9:12, :]
    earth_velocities = sol.y[3 * (n_bodies + 3):3 * (n_bodies + 3) + 3, :]
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
