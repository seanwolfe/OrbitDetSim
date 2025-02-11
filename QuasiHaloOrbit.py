import pandas as pd
import matplotlib.pyplot as plt
import spiceypy as sp
import numpy as np

# Define a function to parse the scientific notation
def parse_numbers(row):
    return [float(num.replace("D", "E")) for num in row.split()[1:]]


def lph_orbit_txt_csv(file_path="LPF_orbit.txt"):
    # Read file
    data = []

    with open(file_path, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) == 7:  # Ensure correct format
                timestamp = parts[0]
                numbers = parse_numbers(line)
                data.append([timestamp] + numbers)

    # Convert to Pandas DataFrame
    columns = ["Time", "GEO_X_(km)", "GEO_Y_(km)", "GEO_Z_(km)", "GEO_Vx_(km/s)", "GEO_Vy_(km/s)", "GEO_Vz_(km/s)"]
    df = pd.DataFrame(data, columns=columns)

    # Convert Time column to datetime format (optional)
    df["Time"] = pd.to_datetime(df["Time"])

    df.to_csv("LPF_orbit.csv", sep=',', header=True, index=False)

    return

def get_lpf_helio():
    lpf_orbit = pd.read_csv('LPF_orbit.csv', sep=',', header=0, names=["Time", "GEO_X_(km)", "GEO_Y_(km)",
                                                                       "GEO_Z_(km)", "GEO_Vx_(km/s)", "GEO_Vy_(km/s)",
                                                                       "GEO_Vz_(km/s)"])

    # Load the necessary SPICE kernels (make sure you specify the correct path)
    sp.furnsh('de430.bsp')  # Load ephemeris data (e.g., DE431)
    sp.furnsh('naif0012.tls')


    # Convert start time to ephemeris time
    et_times = [sp.str2et(row['Time']) for i, row in lpf_orbit.iterrows()]

    # Query ephemeris for all time points (batch mode)
    earth_vectors = np.vstack([sp.spkezr('EARTH', et, 'J2000', 'NONE', 'SUN')[0] for et in et_times])


    # Orbit's Helio position
    # Ensure indices match
    lpf_orbit = lpf_orbit.reset_index(drop=True)

    helio_df = pd.DataFrame(
    earth_vectors[:, :3] + lpf_orbit[['GEO_X_(km)', 'GEO_Y_(km)', 'GEO_Z_(km)']].values,
    columns=['HELIO_X_(km)', 'HELIO_Y_(km)', 'HELIO_Z_(km)'],
    index=lpf_orbit.index  # Maintain correct indexing
    )

    lpf_orbit[['HELIO_X_(km)', 'HELIO_Y_(km)', 'HELIO_Z_(km)']] = helio_df

    # Orbit's Helio velocity
    helio_df_vel =  pd.DataFrame(
        earth_vectors[:, 3:] + lpf_orbit[["GEO_Vx_(km/s)", "GEO_Vy_(km/s)", "GEO_Vz_(km/s)"]].values,
        columns=['HELIO_Vx_(km/s)', 'HELIO_Vy_(km/s)', 'HELIO_Vz_(km/s)'],
        index=lpf_orbit.index
    )

    lpf_orbit[['HELIO_Vx_(km/s)', 'HELIO_Vy_(km/s)', 'HELIO_Vz_(km/s)']] = helio_df_vel

    # Unload SPICE kernels
    sp.kclear()

    lpf_orbit.to_csv("LPF_orbit.csv", sep=',', header=True, index=False)

    return


import numpy as np


def eci_ecliptic_to_sunearth_synodic(sun_eph, obj_xyz):
    """
    Transforms coordinates from the ECI ecliptic plane to an Earth-centered Sun-Earth co-rotating (synodic) frame.

    :param sun_eph: (3, n) array, ephemeris x, y, z of the Sun in ECI ecliptic frame
    :param obj_xyz: (3, n) array, position of the object in the ECI ecliptic frame
    :return: (3, n) array, transformed x, y, z coordinates in the synodic frame
    """

    # Compute unit vectors pointing from Earth to the Sun
    u_s = -sun_eph / np.linalg.norm(sun_eph, axis=0, keepdims=True)

    # Compute rotation angles (theta) for each time step
    theta = np.arctan2(u_s[1, :], u_s[0, :])  # Shape: (n,)

    # Compute cos and sin of theta for batch matrix construction
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Create 2x2 rotation matrices in batch form (3x3 but with Z rotation only)
    Rz_theta = np.array([
        [cos_t, -sin_t, np.zeros_like(theta)],
        [sin_t, cos_t, np.zeros_like(theta)],
        [np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta)]
    ])  # Shape: (3, 3, n)

    # Apply the rotation (batch matrix multiplication)
    trans_xyz = np.einsum('ijk,jk->ik', Rz_theta, obj_xyz)  # Shape: (3, n)

    return trans_xyz


def get_emb_synodic():
    lpf_orbit = pd.read_csv('LPF_orbit.csv', sep=',', header=0,
                            names=["Time", "GEO_X_(km)", "GEO_Y_(km)", "GEO_Z_(km)", "GEO_Vx_(km/s)", "GEO_Vy_(km/s)",
                                   "GEO_Vz_(km/s)", 'HELIO_X_(km)', 'HELIO_Y_(km)', 'HELIO_Z_(km)', 'HELIO_Vx_(km/s)',
                                   'HELIO_Vy_(km/s)', 'HELIO_Vz_(km/s)'])

    # Load the necessary SPICE kernels (make sure you specify the correct path)
    sp.furnsh('de430.bsp')  # Load ephemeris data (e.g., DE431)
    sp.furnsh('naif0012.tls')

    # Convert start time to ephemeris time
    et_times = [sp.str2et(row['Time']) for i, row in lpf_orbit.iterrows()]

    # Query ephemeris for all time points (batch mode)
    moon_vectors = np.vstack([sp.spkezr('MOON', et, 'J2000', 'NONE', 'SUN')[0] for et in et_times])

    # Orbit's Helio velocity
    helio_df_moon = pd.DataFrame(
        moon_vectors,
        columns=['MOON_HELIO_X_(km)', 'MOON_HELIO_Y_(km)', 'MOON_HELIO_Z_(km)', 'MOON_HELIO_Vx_(km/s)',
                  'MOON_HELIO_Vy_(km/s)', 'MOON_HELIO_Vz_(km/s)'],
        index=lpf_orbit.index
    )

    lpf_orbit[['MOON_HELIO_X_(km)', 'MOON_HELIO_Y_(km)', 'MOON_HELIO_Z_(km)', 'MOON_HELIO_Vx_(km/s)',
                  'MOON_HELIO_Vy_(km/s)', 'MOON_HELIO_Vz_(km/s)']] = helio_df_moon

    # generate the x, y, z of the trajectory in the Sun-Earth/Moon synodic frame, centered at the earth-moon barycentre
    # calcualte the position of the earth-moon barycentr
    m_e = 5.97219e24  # mass of Earth
    m_m = 7.34767309e22  # mass of the Moon

    helio_earth = lpf_orbit[['HELIO_X_(km)', 'HELIO_Y_(km)', 'HELIO_Z_(km)']].values - lpf_orbit[["GEO_X_(km)",
                                                                                                  "GEO_Y_(km)",
                                                                                                  "GEO_Z_(km)"]].values
    barycentre = (m_e * helio_earth + m_m *
                  lpf_orbit[['MOON_HELIO_X_(km)', 'MOON_HELIO_Y_(km)', 'MOON_HELIO_Z_(km)']].values) / (m_m + m_e)

    # translate x, y, z to EMB
    emb_xyz = lpf_orbit[['HELIO_X_(km)', 'HELIO_Y_(km)', 'HELIO_Z_(km)']].values - barycentre

    # get synodic in emb frame
    emb_xyz_synodic = eci_ecliptic_to_sunearth_synodic(-barycentre.T, emb_xyz.T)
    good_dims = emb_xyz_synodic.T

    # Orbit's Helio velocity
    emb_synodic = pd.DataFrame(
        good_dims,
        columns=['EMB_SYNODIC_X_(km)', 'EMB_SYNODIC_Y_(km)', 'EMB_SYNODIC_Z_(km)'],
        index=lpf_orbit.index
    )

    lpf_orbit[['EMB_SYNODIC_X_(km)', 'EMB_SYNODIC_Y_(km)', 'EMB_SYNODIC_Z_(km)']] = emb_synodic
    lpf_orbit.to_csv("LPF_orbit.csv", sep=',', header=True, index=False)

    return


lpf_orbit = pd.read_csv('LPF_orbit.csv', sep=',', header=0,
                            names=["Time", "GEO_X_(km)", "GEO_Y_(km)", "GEO_Z_(km)", "GEO_Vx_(km/s)", "GEO_Vy_(km/s)",
                                   "GEO_Vz_(km/s)", 'HELIO_X_(km)', 'HELIO_Y_(km)', 'HELIO_Z_(km)', 'HELIO_Vx_(km/s)',
                                   'HELIO_Vy_(km/s)', 'HELIO_Vz_(km/s)', 'MOON_HELIO_X_(km)', 'MOON_HELIO_Y_(km)',
                                   'MOON_HELIO_Z_(km)', 'MOON_HELIO_Vx_(km/s)', 'MOON_HELIO_Vy_(km/s)',
                                   'MOON_HELIO_Vz_(km/s)', 'EMB_SYNODIC_X_(km)', 'EMB_SYNODIC_Y_(km)',
                                   'EMB_SYNODIC_Z_(km)'])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(lpf_orbit['EMB_SYNODIC_X_(km)'], lpf_orbit['EMB_SYNODIC_Y_(km)'], lpf_orbit['EMB_SYNODIC_Z_(km)'])
# plt.show()