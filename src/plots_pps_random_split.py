import os
import pathlib
from os.path import join
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

def plot_trend(report_list, path_name, title_suffix=''):
    df = pd.concat(report_list)

    print(df)

    df["method_features"] = df["method"] + " (" + df["features"] + ")"
    df["method_features"] = df["method_features"].str.replace(r"MLPE .*", "MLPE", regex=True)

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="tr_size", y="nmd", hue="method_features", marker="o", palette="tab10")

    plt.xlabel("Training Size")
    plt.ylabel("NMD Error")
    plt.title(f"{dataset} ({n_classes} classes) "+title_suffix)

    plt.legend(title="Method (Features)", loc='upper left', bbox_to_anchor=(1,1))

    plt.tight_layout()
    os.makedirs(pathlib.Path(path_name).parent, exist_ok=True)
    plt.savefig(path_name)




outpath = '../fig/random_split/'

for dataset in ['activity', 'toxicity', 'diversity']:
    for n_classes in [3, 5]:
        result_path = f'../results/random_split/samplesize500/{dataset}/{n_classes}_classes/*.csv'

        reports = {}
        for csv_path in glob(result_path):
            method_features = pathlib.Path(csv_path).name.replace('.csv', '')
            df = pd.read_csv(csv_path, index_col=0)
            reports[method_features]=df

        # method_names = sorted(set([method_features.split('__')[0] for method_features in reports.keys()]))
        # print(method_names)
        method_names = ['MLPE', 'CC', 'PCC', 'ACC', 'PACC', 'EMQ', 'KDEy-ML']

        # plots for each type of features
        for features in ['new', 'old', 'both']:
            path_name = join(outpath, dataset, f'{n_classes}_classes', f'{features}_features.png')
            report_list = [reports[method_name+'__'+features] for method_name in method_names]
            # plot_trend(report_list, path_name, title_suffix=features+' features')

        # plot some selected methods comparing the performance with different types of features
        method_names = ['MLPE__old']
        selected_methods_names = ['EMQ']
        for method in selected_methods_names:
            for features in ['new', 'old', 'both']:
                method_names.append(f'{method}__{features}')
            # method_names.append('EMQ-b__new')
        path_name = join(outpath, dataset, f'{n_classes}_classes', 'features_comparison.png')
        report_list = [reports[method_name] for method_name in method_names]
        plot_trend(report_list, path_name, title_suffix='features comparison')

