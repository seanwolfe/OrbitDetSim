import rebound
import numpy as np

def add_solar_system_km(sim):
    sun_mass = 1.9885e30  # kg
    planet_data = [
        ("Mercury",   3.3011e23,         5.79e7),
        ("Venus",     4.8675e24,        1.082e8),
        ("Earth",     5.9724e24,        1.496e8),
        ("Moon",      7.3477e22,        1.5e8),
        ("Mars",      6.4171e23,        2.279e8),
        ("Jupiter",   1.8982e27,        7.785e8),
        ("Saturn",    5.6834e26,        1.433e9),
        ("Uranus",    8.6810e25,        2.877e9),
        ("Neptune",   1.0241e26,        4.503e9),
    ]
    sim.add(m=sun_mass)
    for _, mass, a in planet_data:
        sim.add(m=mass, a=a)

def generate_asteroid_trajectory_with_jdtdb(start_jdtdb=2460000.5):
    sim = rebound.Simulation()
    sim.units = ('km', 's', 'kg')
    add_solar_system_km(sim)

    # Add an asteroid
    sim.add(m=0.0, x=1.6e8, y=3.0e7, z=0.0, vx=0.0, vy=30.0, vz=0.0)
    sim.move_to_com()

    dt = 3600  # seconds (1 hour)
    n_steps = 24 * 30  # 30 days
    sec_per_day = 86400.0
    trajectory = []

    for i in range(n_steps):
        sim.integrate(sim.t + dt)
        asteroid = sim.particles[-1]
        # REBOUND's `sim.t` is in seconds from start
        jdtdb = start_jdtdb + (sim.t / sec_per_day)
        row = [jdtdb, asteroid.x, asteroid.y, asteroid.z, asteroid.ax, asteroid.ay, asteroid.az]
        trajectory.append(row)

    return np.array(trajectory)

# Generate and save to CSV
trajectory_data = generate_asteroid_trajectory_with_jdtdb()
np.savetxt("asteroid_trajectory_jdtdb.csv", trajectory_data, delimiter=",",
           header="jdtdb,x,y,z,ax,ay,az", comments='')

print("Saved trajectory with JDTDB epochs to 'asteroid_trajectory_jdtdb.csv'.")
