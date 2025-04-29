import spiceypy as spice
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import argparse

# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')


def integrate_n_body_multi(object_states, epoch, end_time, time_interval):

    # Argument parser to get the config file path
    parser = argparse.ArgumentParser(description="Run the spacecraft simulation")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    epoch_et = spice.unitim(epoch, 'JDTDB', 'ET')  # initial epoch

    # Function to get state vectors (position, velocity) in km & km/s
    def get_state(body, reference=10):
        state, _ = spice.spkgeo(body, epoch_et, "ECLIPJ2000", reference)
        return np.array(state)

    # Get Sun & planets' initial states
    bodies = [10, 1, 2, 399, 4, 5, 6, 7, 8,
              301]  # ["SUN", "MERCURY", "VENUS", "EARTH", "JUPITER", "SATURN", "URANUS", "NEPTUNE", "MOON"]
    planet_states = {body: get_state(body) for body in bodies}

    # Combine all bodies into a state vector
    initial_states = np.vstack([planet_states[body] for body in bodies] + object_states)
    initial_positions = initial_states[:, :3]
    initial_velocities = initial_states[:, 3:]

    # Convert km, km/s to meters, meters/s
    initial_positions *= config['KM_TO_M']
    initial_velocities *= config['KM_TO_M']

    # Flatten initial state vector (for integration)
    y0 = np.hstack([initial_positions.flatten(), initial_velocities.flatten()])

    # Define masses (in kg) for Sun, planets, and Moon
    G = config['GRAVITATIONAL_CONSTANT']  # m^3 kg^-1 s^-2
    masses = {
        10: config['SUN_MASS'], 1: config['MERCURY_MASS'], 2: config['VENUS_MASS'],
        399: config['EARTH_MASS'], 4: config['MARS_MASS'], 5: config['JUPITER_MASS'], 6: config['SATURN_MASS'],
        7: config['URANUS_MASS'], 8: config['NEPTUNE_MASS'], 301: config['MOON_MASS'],
        "ASTEROID": config['asteroid_mass'], "SPACECRAFT": config['spacecraft_mass']  # Arbitrary mass
    }
    mass_array = np.array([masses[body] for body in bodies] + [masses["ASTEROID"]] + [masses["SPACECRAFT"] for i in object_states[1:]])

    start_time = 0
    t_span = (start_time, end_time)  # Start at t=0, end at t=900s
    t_eval = np.linspace(start_time, end_time, time_interval)  # 30s intervals

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
    n_spacecraft = len(object_states) - 1
    asteroid_idx = n_bodies - n_spacecraft - 1  # asteroid index
    sc_idx = n_bodies - 1


    earth_positions = sol.y[9:12, :]
    earth_velocities = sol.y[3 * (n_bodies + 3), :]
    object_positions = sol.y[3 * asteroid_idx:3 * (sc_idx + 1), :]
    object_velocities = sol.y[2 * 3 * (asteroid_idx + 1): 2 * 3 * (sc_idx + 1), :]

    M = object_positions.shape[0] // 3  # Number of objects
    N = object_positions.shape[1]  # Number of time steps

    # Reshape to (M, 3, N), where 3 corresponds to x, y, z
    object_pos_reshaped = object_positions.reshape(M, 3, N)  # (M, 3, N)
    object_vel_reshaped = object_velocities.reshape(M, 3, N)  # (M, 3, N)

    return np.hstack((object_pos_reshaped, object_vel_reshaped)), np.vstack((earth_positions, earth_velocities))

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

    return np.vstack((object_positions, object_velocities)), np.vstack((earth_positions, earth_velocities))

