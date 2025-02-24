





class Spacecraft:


    def __init__(self, ini_pos, ini_pos_index, configs):
        self.position = ini_pos  # initial position of the spacecraft in the quasi-halo orbit
        self.pos_index = ini_pos_index  # initial position index in the quasi-halo orbit csv
        self.velocity = None
        self.attitude = None
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
        return


    def set_state(self, position, velocity):
        self.position = position
        self.velocity = velocity
        return


    def get_attitude(self):
        raise NotImplementedError


    def field_of_view(self, asteroid_distance):
        raise NotImplementedError

