import spiceypy as spice
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import yaml



# Load YAML config file
with open("orbit_det_configuration.yaml", "r") as file:
    config = yaml.safe_load(file)

id = "NESC0000025V"
test_idx = 10000
file_path = config['minimoon_files_folder'] + id + '.csv'
orbit = pd.read_csv(file_path, sep=' ', header=0, names=config['minimoon_column_names'])

# Load SPICE kernels (Ensure you downloaded DE440 as mentioned before)
spice.furnsh("de430.bsp")
spice.furnsh('naif0012.tls')


# Add an asteroid (near Earth)
asteroid_state = orbit.loc[test_idx, ['Helio x', 'Helio y', 'Helio z', 'Helio vx', 'Helio vy', 'Helio vz']].values  # au and au/d
asteroid_state[:3] *= config['AU_TO_M'] / config['KM_TO_M']  # to match spice
asteroid_state[3:] *= (config['AU_TO_M'] / config['KM_TO_M'] / config['SECONDS_PER_DAY'])

epoch_et= spice.unitim(orbit.loc[test_idx, 'Julian Date'], 'JDTDB', 'ET')  # initial epoch

# Function to get state vectors (position, velocity) in km & km/s
def get_state(body, reference=10):
    state, _ = spice.spkgeo(body, epoch_et, "ECLIPJ2000", reference)
    return np.array(state)

moon_state = orbit.loc[test_idx, ["Moon x (Helio)",
                         "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)", "Moon vy (Helio)", "Moon vz (Helio)"]].values  # au and au/d
earth_state = orbit.loc[test_idx, ["Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)",
                         "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)"]].values  # au and au/d


# Get Sun & planets' initial states
bodies = [10, 1, 2, 399, 4, 5, 6, 7, 8, 301] # ["SUN", "MERCURY", "VENUS", "EARTH", "JUPITER", "SATURN", "URANUS", "NEPTUNE", "MOON"]
planet_states = {body: get_state(body) for body in bodies}


# Combine all bodies into a state vector
initial_states = np.vstack([planet_states[body] for body in bodies] + [asteroid_state])
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
    7: config['URANUS_MASS'], 8: config['NEPTUNE_MASS'], 301: config['MOON_MASS'], "ASTEROID": 1e5  # Arbitrary mass
}
mass_array = np.array([masses[body] for body in bodies] + [masses["ASTEROID"]])

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


start_time = 0
end_time = config['SECONDS_PER_DAY'] * 10
interval_time = 3600
t_span = (start_time, end_time)  # Start at t=0, end at t=900s
t_eval = np.linspace(start_time, end_time, interval_time)  # 30s intervals

# Solve the N-body problem
sol = solve_ivp(nbody_derivatives, t_span, y0, method="RK45", t_eval=t_eval)

# Extract asteroid's trajectory
n_bodies = len(mass_array)
asteroid_idx = n_bodies - 1  # Last body in the array is the asteroid
asteroid_x = sol.y[3 * asteroid_idx, :]  # m and m/s
asteroid_y = sol.y[3 * asteroid_idx + 1, :]
asteroid_z = sol.y[3 * asteroid_idx + 2, :]
asteroid_vx = sol.y[2 * 3 * asteroid_idx + 3, :]
asteroid_vy = sol.y[2 * 3 * asteroid_idx + 4, :]
asteroid_vz = sol.y[2 * 3 * asteroid_idx + 5, :]


####################################
# Comparison
####################################
# Plot results
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(asteroid_x / config['AU_TO_M'], asteroid_y / config['AU_TO_M'], asteroid_z / config['AU_TO_M'], 'r-',
        label="Integrated", linewidth=3, zorder=5)
ax.scatter(asteroid_x[0] / config['AU_TO_M'], asteroid_y[0] / config['AU_TO_M'], asteroid_z[0] / config['AU_TO_M'], 'g',
        label="Start", s=10, zorder=15)
ax.plot(orbit['Helio x'], orbit['Helio y'], orbit['Helio z'], 'b', label="Openorb", linewidth=1, zorder=10)
ax.set_xlabel("X Position (au)")
ax.set_ylabel("Y Position (au)")
ax.set_zlabel("Z Position (au)")
plt.legend()

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.plot(asteroid_vx / config['AU_TO_M'] * config['SECONDS_PER_DAY'],
        asteroid_vy / config['AU_TO_M'] * config['SECONDS_PER_DAY'],
        asteroid_vz / config['AU_TO_M'] * config['SECONDS_PER_DAY'], 'r-', label="Integrated", linewidth=3, zorder=5)
ax2.plot(orbit['Helio vx'], orbit['Helio vy'], orbit['Helio vz'], 'b', label="Openorb vel", linewidth=1, zorder=10)
ax2.set_xlabel("X Velocity (au/d)")
ax2.set_ylabel("Y Velocity (au/d)")
ax2.set_zlabel("Z Velocity (au/d)")
plt.legend()


plt.show()

# Unload SPICE kernels
spice.kclear()
