import os
import yaml
from pathlib import Path

# Path to your base config file
base_config_path = 'to_sync/orbit_det_configuration.yaml'
output_dir = 'config_IODDATA'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load base config
with open(base_config_path, 'r') as f:
    base_config = yaml.safe_load(f)

# Generate 80 configs
for run_number in range(1, 81):
    config = base_config.copy()
    config['run_number'] = run_number

    # Save new config file
    filename = f"orbit_det_config_run_{run_number}.yaml"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as out_file:
        yaml.dump(config, out_file, sort_keys=False)

print(f"Generated 80 config files in '{output_dir}'")
