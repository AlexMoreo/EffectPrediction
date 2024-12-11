# from submodule.result_table.src.table import Table
from itertools import product
from glob import glob
from os.path import join
import pickle
from pathlib import Path
import pandas as pd

result_dir = '../results_filtered'
datasets = ['activity', 'diversity', 'toxicity']
# datasets = ['diversity']
n_classes_list = ['3_classes', '5_classes']

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


if __name__ == '__main__':
    final_df = []
    for dataset, n_classes in product(datasets, n_classes_list):
        # table = Table()

        path = join(result_dir, n_classes, dataset, '*.pkl')
        for method_file in glob(path):
            result_df = pickle.load(open(method_file, 'rb'))
            method_name = Path(method_file).name.replace('.pkl', '')
            result_df['method'] = method_name
            result_df['dataset'] = dataset
            result_df['n_classes'] = n_classes
            final_df.append(result_df)
            # print(method_file, method_name)
            # print(result_df)

    final_df = pd.concat(final_df, ignore_index=True)
    pv = pd.pivot_table(
        final_df,
        values='ae',
        index=['n_classes', 'dataset', 'method'],
        columns='period',
        aggfunc='mean',
    )
    mean_values = pv.mean(axis=1)
    pv['Mean'] = mean_values
    print(pv)
