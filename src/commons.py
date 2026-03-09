import os
from os.path import join

SAMPLE_SIZE = 500
# N_TEST_SAMPLES = 1000
# N_VAL_SAMPLES = 250
# N_RUNS=5
N_TEST_SAMPLES = 1000
N_VAL_SAMPLES = 250
N_RUNS=5


# One reviewer asked about the sensitivity to different discretizations of the classes.
# We have created this flag to allow for a different discretization globally.
# The default configuration we used in our paper now corresponds to "fixed".
# The new configuration we tested corresponds to "isodense"
# GLOBAL_DISCRETIZATION_SETUP = "fixed"
GLOBAL_DISCRETIZATION_SETUP = "isodense"


def get_full_path(base_dir, dataset_name, n_classes, sample_size=SAMPLE_SIZE):
    path = join(base_dir, f'samplesize{sample_size}', dataset_name, f'{n_classes}_classes')
    os.makedirs(path, exist_ok=True)
    return path
