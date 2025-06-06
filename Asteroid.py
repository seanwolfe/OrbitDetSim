import pandas as pd
import numpy as np


class Asteroid:

    def __init__(self, id, ini_index, configs):
        self.id = id
        file_path = configs['minimoon_files_folder'] + id + '.csv'
        self.orbit = pd.read_csv(file_path, sep=' ', header=0, names=configs['minimoon_column_names'])
        temp_x = self.orbit['Synodic x']
        temp_y = self.orbit['Synodic y']
        moon_temp_x = self.orbit['Moon Synodic x']
        moon_temp_y = self.orbit['Moon Synodic y']
        self.orbit['Synodic x'] = temp_x
        self.orbit['Synodic y'] = temp_y
        self.orbit['Moon Synodic x'] = moon_temp_x
        self.orbit['Moon Synodic y'] = moon_temp_y
        self.start_index = ini_index  # point of min apparant magnitude in traj
        self.position = np.array([self.orbit['Synodic x'].iloc[ini_index], self.orbit['Synodic y'].iloc[ini_index],
                                  self.orbit['Synodic z'].iloc[ini_index]])
        self.velocity = None

        return

    def set_state(self, position, velocity):
        self.position = position
        self.velocity = velocity

        return


    def get_asteroid_pos(self, index):
        return self.orbit.loc[index, ['Synodic x', 'Synodic y', 'Synodic z']].values