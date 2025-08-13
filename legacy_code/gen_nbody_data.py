import rebound
import spiceypy as spice
import numpy as np
import pandas as pd

# Masses in kg (you can define this however you prefer — dictionary, class, etc.)
body_masses = {
    'SUN': 1.989e30,
    'MERCURY': 3.3e23,
    'VENUS': 4.87e24,
    'EARTH': 5.97e24,
    'MOON': 7.3e22,
    'MARS': 6.42e23,
    'JUPITER': 1.898e27,
    'SATURN': 5.68e26,
    'URANUS': 8.68e25,
    'NEPTUNE': 1.02e26,
}

def add_body_from_spice(sim, target_name, spice_id, et, center='EARTH', frame='ECLIPJ2000'):
    state, _ = spice.spkezr(str(spice_id), et, frame, 'NONE', center)
    pos = state[:3]  # km
    vel = state[3:]  # km/s
    mass = body_masses.get(target_name, 0.0)

    sim.add(m=mass, x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2])

def generate_asteroid_trajectory_real_epoch(start_utc="2023-01-01T00:00:00", duration_days=30):
    # Load kernels
    spice.furnsh("../naif0012.tls")
    spice.furnsh("../de430.bsp")
    filename = "asteroid_trajectory_jdtdb.csv"

    et0 = spice.utc2et(start_utc)
    sim = rebound.Simulation()
    sim.units = ('km', 's', 'kg')

    # Add Sun and planets
    planet_ids = {
        'EARTH': 399,
        'SUN': 10,
        'MERCURY': 1,
        'VENUS': 2,
        'MOON': 301,
        'MARS': 4,
        'JUPITER': 5,
        'SATURN': 6,
        'URANUS': 7,
        'NEPTUNE': 8
    }

    for name, spk_id in planet_ids.items():
        add_body_from_spice(sim, name, spk_id, et0)

    # Add your asteroid manually
    sim.add(m=0.0, x=1.6e6, y=3e4, z=0.0, vx=0.0, vy=5.0, vz=0.0)
    sim.move_to_com()

    # Integrate
    dt = 3600  # seconds
    n_steps = int(duration_days * 24)

    data = []
    for i in range(n_steps):
        sim.integrate(sim.t + dt)
        jdtdb = spice.unitim(et0 + sim.t, 'ET', 'JDTDB')
        earth = sim.particles[0]
        asteroid = sim.particles[-1]


        data.append({
            "jdtdb": jdtdb,
            "x": asteroid.x - earth.x,
            "y": asteroid.y - earth.y,
            "z": asteroid.z - earth.z,
            "ax": asteroid.ax,
            "ay": asteroid.ay,
            "az": asteroid.az
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, float_format="%.10e")

    print(f"Saved trajectory with JDTDB epochs to '{filename}'.")

# Usage
generate_asteroid_trajectory_real_epoch()
