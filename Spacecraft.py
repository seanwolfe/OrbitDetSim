
import numpy as np

class Spacecraft:


    def __init__(self, ini_pos, ini_pos_index, configs):
        self.ini_position = ini_pos  # initial position of the spacecraft in the quasi-halo orbit
        self.ini_pos_index = ini_pos_index  # initial position index in the quasi-halo orbit csv
        self.velocity = None
        self.position = None
        self.boresight = np.array([-1, 0, 0])
        self.pixel_scale = configs['pixel_scale']
        self.fov = configs['fov']
        self.number_of_pixels = configs['number_of_pixels']
        self.reaction_wheel_torque = configs['reaction_wheel_torque']
        self.reaction_wheel_momentum = configs['reaction_wheel_momentum']
        self.mass = configs['mass']
        self.length = configs['length']
        self.telescope_diameter = configs['telescope_diameter']
        self.sigma_ra = configs['sigma_ra']
        self.sigma_dec = configs['sigma_dec']
        self.sigma_pointing = configs['sigma_pointing']
        self.matched_trajectory = None  # this contains an array of the trajectory of the sc that has same length as the asteroid traj in question
        return


    def set_state(self, position, velocity):
        self.position = position
        self.velocity = velocity
        return


    def get_spacecraft_pos(self, index):
        return self.matched_trajectory[index, :]


    def get_attitude(self):
        raise NotImplementedError


    def is_occluded_batch(self, spacecraft_pos, asteroid_pos, earth_pos, moon_pos, configs):
        """Vectorized occlusion check for all time steps."""
        def check_occlusion(body_pos, body_radius):
            sc_to_ast = asteroid_pos - spacecraft_pos  # (N,3)
            sc_to_body = body_pos - spacecraft_pos  # (N,3)

            proj_length = np.einsum('ij,ij->i', sc_to_body, sc_to_ast) / np.linalg.norm(sc_to_ast, axis=1)
            proj_point = spacecraft_pos + (sc_to_ast / np.linalg.norm(sc_to_ast, axis=1)[:, None]) * proj_length[:, None]

            min_distance = np.linalg.norm(proj_point - body_pos, axis=1)
            occluded = (min_distance < body_radius) & (proj_length > 0) & (proj_length < np.linalg.norm(sc_to_ast, axis=1))
            return occluded

        return (check_occlusion(earth_pos, configs['EARTH_RADIUS_AU'] / configs['AU_TO_M'])
                | check_occlusion(moon_pos, configs['MOON_RADIUS_AU'] / configs['AU_TO_M']))

    def asteroid_in_fov_batch(self, asteroid_trajectory, spacecraft_position, earth_position, moon_position, configs):
        """
        Determine when the asteroid is in the field of view, considering occlusion.

        Parameters:
            asteroid_positions Nx3 array in AU
            spacecraft_position (Nx3 array in AU): Spacecraft position at each epoch.
            earth_position (Nx3 array in AU): Earth position at each epoch.
            moon_position (Nx3 array in AU): Moon position at each epoch.
            configs: confiugaration from yaml file

        Returns:
            in_fov (Nx1 array): Indices where the asteroid is visible, NaN if not visible.
        """
        positions = asteroid_trajectory
        fov_radians = np.radians(np.sqrt(self.fov))

        # Compute relative position vectors (N,3)
        rel_pos = positions - spacecraft_position
        rel_pos_norm = np.linalg.norm(rel_pos, axis=1)

        # Compute angle with boresight
        dot_product = np.dot(rel_pos, self.boresight)
        angles = np.arccos(dot_product / rel_pos_norm)

        # Check if inside FOV
        in_fov = angles < (fov_radians / 2)

        # Check occlusion
        occluded = self.is_occluded_batch(spacecraft_position, positions, earth_position, moon_position, configs)

        # Final visibility check
        visible_indices = np.where(in_fov & ~occluded)[0]

        # Create output array
        result = np.full_like(asteroid_trajectory[:, 0], np.nan, dtype=float)  # Initialize with NaN
        result[visible_indices] = visible_indices  # Assign index where visible

        return result

