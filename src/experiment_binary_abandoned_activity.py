import os
import pickle
from os.path import join
from itertools import product

import pandas as pd
import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from data import load_dataset
from experiments import main
from methods import methods
import warnings

from utils import mmd_pairwise_rbf_blocks

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')


if __name__ == '__main__':
    dataset_dir = '../datasets'

    targets = ['global'] #, 'periods']
    n_classes = 2
    dataset_name = 'activity'
    for target in targets:
        results_dir = f'../results_abandoned_{target}'
        print(f'running {dataset_name=} {n_classes}')
        data = load_dataset(join(dataset_dir, f'{dataset_name}_dataset'), n_classes=n_classes,
                            filter_out_multiple_subreddits=True,
                            filter_abandoned_activity=False)
        os.makedirs(join(results_dir, f'{n_classes}_classes', dataset_name), exist_ok=True)
        main(data, n_classes, target, dataset_name, results_dir)