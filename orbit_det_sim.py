import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import yaml

# Load YAML config file
with open("orbit_det_configuration.yaml", "r") as file:
    config = yaml.safe_load(file)

# open minimoon_master (assuming master and minimoon files have l1 apparant magnitude over traj and min index in master)


# iterate over minimoons in parallel

    # declare asteroid

    # declare formation

    # determine end index

    # reintegrate from start time to end time at desired time interval

    # transform to sun-earth co-rotating (position and velocity)

    # calc ra and dec

    # save file
