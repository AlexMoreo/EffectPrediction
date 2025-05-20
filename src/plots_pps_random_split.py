import os
import pathlib
from os.path import join
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout
from data import FEATURE_GROUP_PREFIXES
import numpy as np

from submodules.result_table.src.new_table import LatexTable

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def plot_trend(report_list, path_name, dataset, n_classes, plotsize, legend, title=''):
    df = pd.concat(report_list)

    print(df)

    df["method_features"] = df["method"] #+ " (" + df["features"] + ")"
    df["method_features"] = df["method_features"].str.replace(r"MLPE .*", "MLPE", regex=True)

    print(df)

    sns.set(style="whitegrid")

    plt.figure(figsize=plotsize)
    sns.lineplot(data=df, x="tr_size", y="nmd", hue="method_features", marker="o", palette="tab10")

    plt.xlabel("Training Size")
    plt.ylabel("NMD Error")
    plt.title(title)

    if legend:
        plt.legend(title="Method", loc='upper left', bbox_to_anchor=(1,1))
    else:
        plt.legend().remove()
    plt.ylim(0.05,0.3)

    plt.tight_layout()
    os.makedirs(pathlib.Path(path_name).parent, exist_ok=True)
    plt.savefig(path_name)


def compute_AUC(report_list, dataset):
    df = pd.concat(report_list)
    assert len(df.method.unique())==1, 'unexpected data'

    auc_dict = []
    for features in df.features.unique():
        df_sel = df[df['features']==features]
        mean_nmd_df = df_sel.groupby(['method', 'tr_size'])['nmd'].mean().reset_index()

        for method, group in mean_nmd_df.groupby('method'):
            x = group['tr_size'].values
            y = group['nmd'].values

            sorted_idx = np.argsort(x)
            x_sorted = x[sorted_idx]
            y_sorted = y[sorted_idx]
            auc = np.trapz(y_sorted, x_sorted)
            auc_dict.append({'features': features, 'auc': auc, 'dataset':dataset})

    auc_df = pd.DataFrame(auc_dict)
    return auc_df


def generate_trends(method_names, out_dir='../fig/random_split_features/'):

    for dataset in ['activity', 'toxicity', 'diversity']:
        for n_classes in [5]: #[3, 5]:
            result_path = f'../results/random_split_features/samplesize500/{dataset}/{n_classes}_classes/*.csv'

            reports = {}
            for csv_path in glob(result_path):
                method_features = pathlib.Path(csv_path).name.replace('.csv', '')
                method_name = method_features.split('__')[0]
                if method_name in method_names:
                    df = pd.read_csv(csv_path, index_col=0)
                    reports[method_features]=df

            # plots for each type of features
            for features in ['all'] + FEATURE_GROUP_PREFIXES:
                path_name = join(out_dir, dataset, f'{n_classes}_classes', f'{features}_features.png')
                report_list = [reports[method_name+'__'+(features if method_name!='MLPE' else 'all')] for method_name in method_names]

                plotsize = (5,5)
                legend = False
                if features=='all':
                    plotsize = (10, 5)
                    legend = True

                plot_trend(report_list, path_name, dataset, n_classes, plotsize, legend, title=features)
                # auc_df.append(compute_AUC(report_list, dataset))

def generate_auc(method, n_classes=5, out_dir='../tables'):
    auc_df = []
    for dataset in ['activity', 'toxicity', 'diversity']:
        result_path = f'../results/random_split_features/samplesize500/{dataset}/{n_classes}_classes/*.csv'

        reports = []
        for csv_path in glob(result_path):
            method_features = pathlib.Path(csv_path).name.replace('.csv', '')
            method_name = method_features.split('__')[0]
            if method_name==method:
                df = pd.read_csv(csv_path, index_col=0)
                reports.append(df)

        auc_df.append(compute_AUC(reports, dataset))

    path_name = join(out_dir, f'auc_features_{method}_{n_classes}_classes.pdf')

    df = pd.concat(auc_df)
    table = LatexTable.from_dataframe(df, method='features', benchmark='dataset', value='auc')
    table.format.configuration.show_std = False
    table.format.configuration.stat_test = None
    table.name = f'{method}-{n_classes}classes'
    table.latexPDF(path_name)


if __name__ == '__main__':
    from submodules.result_table.src.new_table import LatexTable
    a = LatexTable()

    method_names = ['MLPE', 'CC', 'EMQ']
    generate_trends(method_names)
    generate_auc(method='EMQ', n_classes=5)