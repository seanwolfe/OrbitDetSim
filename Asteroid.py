import pandas as pd
import numpy as np


class Asteroid:

    def __init__(self, id, ini_index, configs):
        self.id = id
        file_path = configs['minimoon_files_folder'] + id + '.csv'
        self.orbit = pd.read_csv(file_path, sep=' ', header=0, names=configs['minimoon_column_names'])
        self.orbit['Synodic x'] *= -1  # old files used positive x away from sun, this simulation positive x is towards sun
        self.orbit['Synodic y'] *= -1
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