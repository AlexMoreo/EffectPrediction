from itertools import product
from glob import glob
from os.path import join
import pickle
from pathlib import Path
import pandas as pd



pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def gen_tables_periods(datasets, n_classes_list, result_dir):
    final_df = []
    for dataset, n_classes in product(datasets, n_classes_list):

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
        values=['ae'],
        index=['n_classes', 'dataset', 'method'],
        columns='period',
        aggfunc='mean',
    )
    mean_values = pv.mean(axis=1)
    pv['Mean'] = mean_values
    print(pv)


def gen_tables_global(datasets, n_classes_list, result_dir, tables_pdf_path):
    final_df = []
    tables=[]
    for dataset, n_classes in product(datasets, n_classes_list):
        table = Table(f'{dataset}-{n_classes}classes-global')
        table.format.show_std=False

        path = join(result_dir, n_classes, dataset, '*.pkl')
        for method_file in glob(path):
            result_df = pickle.load(open(method_file, 'rb'))
            method_name = Path(method_file).name.replace('.pkl', '')
            if method_name=='bPCC-cv-sel': continue
            result_df['method'] = method_name
            result_df['dataset'] = dataset
            result_df['n_classes'] = n_classes
            final_df.append(result_df)
            subreddit_idx = result_df['subreddit_idx_test'].values
            nmd = result_df['nmd'].values
            for idx, val in zip(subreddit_idx, nmd):
                table.add(f'sub-{idx}', method_name, val)
            # print(method_file, method_name)
            # print(result_df)
        tables.append(table)

    final_df = pd.concat(final_df, ignore_index=True)
    pv = pd.pivot_table(
        final_df,
        values=['nmd'], #, 'rae'],
        index=['n_classes', 'dataset', 'method'],
        columns='subreddit_idx_test',
        aggfunc='mean',
    )
    mean_values = pv.mean(axis=1)
    pv['Mean'] = mean_values
    print(pv)

    Table.LatexPDF(tables_pdf_path, tables, resizebox=True, verbose=True)


if __name__ == '__main__':

    datasets = ['activity', 'diversity', 'toxicity']
    # n_classes_list = ['3_classes']
    n_classes_list = ['5_classes']

    # result_dir = '../results_periods'
    # gen_tables_periods(datasets, n_classes_list, result_dir)

    result_dir = '../results/results_global'
    gen_tables_global(datasets, n_classes_list, result_dir, '../tables/global.pdf')

    # result_dir = '../results_abandoned_global'
    # gen_tables_global(['activity'], ['2_classes'], result_dir, '../tables/abandoned_global.pdf')
