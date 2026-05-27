import pandas as pd
import numpy as np
import utilities as util


class Asteroid:

    def __init__(self, id, ini_index, configs, current_state_eme=None, current_epoch=None):
        self.id = id
        file_path = configs['minimoon_files_folder'] + id + '.csv'
        self.orbit = util.read_csv_comma_or_space(file_path)
        temp_x = self.orbit['Synodic x']
        temp_y = self.orbit['Synodic y']
        # moon_temp_x = self.orbit['Moon Synodic x']
        # moon_temp_y = self.orbit['Moon Synodic y']
        self.orbit['Synodic x'] = temp_x
        self.orbit['Synodic y'] = temp_y
        # self.orbit['Moon Synodic x'] = moon_temp_x
        # self.orbit['Moon Synodic y'] = moon_temp_y
        self.start_index = int(ini_index)  # point of min apparant magnitude in traj
        self.position = np.array([self.orbit['Synodic x'].iloc[self.start_index], self.orbit['Synodic y'].iloc[self.start_index],
                                  self.orbit['Synodic z'].iloc[self.start_index]])
        self.velocity = None
        self.curr_state_eme = current_state_eme
        self.curr_epoch = current_epoch

        return

    def set_state(self, state):
        self.curr_state_eme = state

        return


    def get_asteroid_pos(self, index):
        return self.orbit.loc[index, ['Synodic x', 'Synodic y', 'Synodic z']].values
