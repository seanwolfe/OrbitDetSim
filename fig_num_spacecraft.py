import numpy as np
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import ast
import utilities as util
import yaml
import os
import argparse

def save_data_num_spacecraft(config):


    first_level_dirs = [os.path.join(config['spacecraft_folder'], name) for name in os.listdir(config['spacecraft_folder'])
                        if os.path.isdir(os.path.join(config['spacecraft_folder'], name))]

    num_cols = len(first_level_dirs)

    file_numbers = []
    for idx, directory in enumerate(first_level_dirs):
        all_files = util.get_all_files(directory)
        file_numbers.append(len(all_files))

    num_rows = max(file_numbers)
    percentages = np.zeros((num_rows, num_cols))

    for idx, directory in enumerate(first_level_dirs):
        all_files = util.get_all_files(directory)
        num_files = 0
        print(directory.split('/')[-1])

        for i, file_i in enumerate(all_files):
            print(file_i.split('/')[-1].split('.')[0])
            print(f"Files completed: {num_files}")
            num_files += 1


            run_data = util.read_master(file_i, config)

            # Create a mask to filter nonnegative values
            run_data["min_nonnegative"] = run_data["values"].apply(
                lambda x: min([y for y in x if y >= 0]) if np.any(np.array(x) >= 0) else np.nan)

            # Find the spacecraft with the minimum value for each object_id
            detected_pop = run_data[~np.isnan(run_data["min_nonnegative"])]
            missed_pop = run_data[np.isnan(run_data["min_nonnegative"])]

            total = run_data.index.get_level_values('object_id').nunique()
            unique_object_ids = detected_pop.index.get_level_values('object_id').nunique()
            missed_unique_object_ids = missed_pop.index.get_level_values('object_id').nunique()

            spacecraft_number_idx = int(directory.split('_')[1])- 1
            run_number_idx = int(file_i.split('_')[-1].split('.')[0]) - 1

            percentages[run_number_idx, spacecraft_number_idx] = unique_object_ids / total * 100

    spacecraft_numbers = [str(i + 1) + '_Spacecraft' for i in range(num_cols)]
    run_numbers =  np.arange(1, num_rows + 1).reshape(-1, 1)
    data = np.hstack((run_numbers, percentages))
    df = pd.DataFrame(data, columns=['Run_numbers'] + spacecraft_numbers)
    df['Run_numbers'] = df['Run_numbers'].astype(int)

    df.to_csv('fig_num_spacecraft_data.csv', sep=',', header=True, index=False)

    return




# Argument parser to get the config file path
parser = argparse.ArgumentParser(description="Run the spacecraft simulation")
parser.add_argument('--config', type=str, required=True, help="Path to the config file")
args = parser.parse_args()

# Load the config file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

save_data_num_spacecraft(config)
