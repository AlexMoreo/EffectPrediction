import os
from os.path import join

SAMPLE_SIZE = 500


def get_full_path(base_dir, dataset_name, n_classes, sample_size=SAMPLE_SIZE):
    path = join(base_dir, f'samplesize{sample_size}', dataset_name, f'{n_classes}_classes')
    os.makedirs(path, exist_ok=True)
    return path
