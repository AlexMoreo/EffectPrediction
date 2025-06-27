import os
import pathlib
from os.path import join
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from commons import SAMPLE_SIZE
from comparison_group import SelectByName, ColourGroup
from data import FEATURE_GROUP_PREFIXES, FEATURE_SUBGROUP_PREFIXES
from format import FormatModifierSelectColor

from new_table import LatexTable, save_text
from tools import tex_table, tex_document, latex2pdf
from utils import AUC_from_result_df
from feature_block_selection import load_precomputed_result

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

n_classes = 5
result_dir = '../results'


def plot_trend_by_methods(report_list, path_name, plotsize, legend, title=''):
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


def plot_trend_by_feats(report_list, path_name, dataset, plotsize, legend, title='', sel_method=None):
    df = pd.concat(report_list)

    if sel_method is not None:
        df = df[df.method==sel_method]

    print(df)

    # df["method_features"] = df["method"] #+ " (" + df["features"] + ")"
    # df["method_features"] = df["method_features"].str.replace(r"MLPE .*", "MLPE", regex=True)

    print(df)

    sns.set(style="whitegrid")

    plt.figure(figsize=plotsize)
    sns.lineplot(data=df, x="tr_size", y="nmd", hue="features", marker="o", palette="tab10")

    plt.xlabel("Training Size")
    plt.ylabel("NMD Error")
    plt.title(title)

    if legend:
        plt.legend(title="Features", loc='upper left', bbox_to_anchor=(1,1))
    else:
        plt.legend().remove()
    plt.ylim(0.05,0.3)

    plt.tight_layout()
    os.makedirs(pathlib.Path(path_name).parent, exist_ok=True)
    plt.savefig(path_name)


def compute_AUC(report_list, dataset):
    df = pd.concat(report_list)
    assert len(df.method.unique())==1, 'unexpected number of methods'
    datasets = df.dataset.unique()
    assert len(datasets) == 1, 'unexpected number of datasets'
    assert datasets[0] == dataset, 'unexpected dataset'

    auc_dict = []
    for features in df.features.unique():
        df_sel = df[df['features']==features]
        auc = AUC_from_result_df(df_sel)

        if features == 'all':
            father, soon = 'root', 'all'
        elif '--' in features:
            father, soon = features.split('--')
        else:
            father, soon = 'root', features

        auc_dict.append({'features': features, 'father': father, 'soon': soon, 'auc': auc, 'dataset': dataset})

        # adds a duplicate that serves as a reference method for the children
        if father=='root' and soon != 'all':
            auc_dict.append({'features': features, 'father': soon, 'soon': soon + " (full)", 'auc': auc, 'dataset': dataset})

    auc_df = pd.DataFrame(auc_dict)
    return auc_df


def load_exploration_report(method, result_dir, config_path, dataset):
    import json

    result_all_features = load_precomputed_result('../results/random_split_features', dataset,
                                                  n_classes=5, method=method, feature_block='all')

    result_path = join(result_dir, 'exploration', config_path, f'{method}_exploration.json')
    with open(result_path, "r", encoding="utf-8") as f:
        exploration_data = json.load(f)
        exploration_data['reference_all_score'] = result_all_features
        exploration_data['rel_all_err_reduction'] = 100 * (result_all_features - exploration_data['final_score']) / result_all_features
        return exploration_data


def generate_trends_plots(method_names, out_dir='../fig/random_split_features/'):

    for dataset in ['activity', 'toxicity', 'diversity']:
        config_path = f'samplesize{SAMPLE_SIZE}/{dataset}/{n_classes}_classes'
        results = []
        for m in method_names:
            result_path = join(result_dir, 'random_split_features', config_path, f'{m}__all.csv')
            df = pd.read_csv(result_path, index_col=0)
            results.append(df)

            # for the selected method, it also loads the exploration report and shows the optimized features trend
            if m == method:
                optim_path = load_exploration_report(method, result_dir, config_path, dataset)['best_conf_path']
                df = pd.read_csv(optim_path, index_col=0)
                df['method'] = df['method'].replace('EMQ', 'EMQ optimized')
                results.append(df)


            # generates a plot comparing the trends of all methods
            path_name = join(out_dir, f'samplesize{SAMPLE_SIZE}_{n_classes}_classes', f'{dataset}.pdf')
            plotsize = (15, 8)
            legend = True

            plot_trend_by_methods(results, path_name, plotsize, legend, title='Methods comparison')


def generate_auc_tables(method, out_dir='../tables'):
    auc_df_by_dataset = []
    for dataset in ['activity', 'toxicity', 'diversity']:
        config_path = f'samplesize{SAMPLE_SIZE}/{dataset}/{n_classes}_classes'
        result_path = join(result_dir, 'random_split_features', config_path, f'EMQ__*.csv')

        reports = []
        for csv_path in glob(result_path):
            method_features = pathlib.Path(csv_path).name.replace('.csv', '')
            method_name = method_features.split('__')[0]
            if method_name==method:
                df = pd.read_csv(csv_path, index_col=0)
                reports.append(df)

        auc_df = compute_AUC(reports, dataset)
        auc_df_by_dataset.append(auc_df)

    final_df = pd.concat(auc_df_by_dataset)

    tables = []
    for father in final_df.father.unique():
        auc_sel_df = final_df[final_df.father==father]

        table = LatexTable.from_dataframe(auc_sel_df, method='soon', benchmark='dataset', value='auc', name=father)
        table.format.configuration.show_std = False
        table.format.configuration.stat_test = None
        table.format.configuration.side_columns = True
        table.format.configuration.mean_prec=1
        columns = list(table.methods)
        first = [f for f in columns if '(full)' in f or f=='all']  # the father is the only one without "(full)"
        rest  = sorted([f for f in columns if f not in first])
        table.reorder_methods(first+rest)
        for i in range(table.n_methods):
            table.methods
        tables.append(table)

    pdf_path_name = join(out_dir, f'auc_features_{method}_{n_classes}_classes.pdf')
    LatexTable.LatexPDF(pdf_path=pdf_path_name, tables=tables, dedicated_pages=False)


def generate_selection_table(method, out_dir='../tables'):

    exploration_reports = {}
    selected_features = {}
    datasets = ['activity', 'toxicity', 'diversity']
    for dataset in datasets:
        config_path = f'samplesize{SAMPLE_SIZE}/{dataset}/{n_classes}_classes'

        exploration_report = load_exploration_report(method, result_dir, config_path, dataset)
        selected_features[dataset] = exploration_report['selected_features']
        exploration_reports[dataset] = exploration_report

    n_features = len(FEATURE_SUBGROUP_PREFIXES)
    n_datasets = 3
    table = LatexTable(name='selection')
    table.format.configuration.show_std=False
    table.format.configuration.mean_prec = 0
    table.format.configuration.with_color = False
    table.format.configuration.best_in_bold = False
    table.format.configuration.stat_test = None
    table.add_format_modifier(
        format_modifier=FormatModifierSelectColor(
            comparison=ColourGroup(selection_mask=SelectByName())
        )
    )
    for feature in FEATURE_SUBGROUP_PREFIXES:
        for dataset in datasets:
            selected = 1 if feature in selected_features[dataset] else 0
            table.add(benchmark=feature, method=dataset, v=selected)

    # generate the latex table
    pdf_path_name = join(out_dir, f'selected_features.pdf')

    table_arr = table.as_str_array()

    def escape_latex_underscore(s):
        import re
        return re.sub(r'(?<!\\)_', r'\_', s)

    def sideways(text):
        return r'\begin{sideways}' + text + r'\;\end{sideways}'

    for i in range(1, table_arr.shape[0]):
        table_arr[i, 0] = escape_latex_underscore(table_arr[i, 0])

    # add one column to the left, with the first level of feature names
    empty_col = np.full((n_features+1, 1), '', dtype=object)
    table_arr = np.hstack([empty_col, table_arr])
    feat_group_head = None
    for i in range(1,table_arr.shape[0]):
        feat_group, feat_subgroup = table_arr[i, 1].split('--')
        print(feat_group, feat_subgroup)
        # table_arr[i,0] = feat_group
        # write the feature group once with multirow
        if feat_group_head is None or feat_group_head != feat_group:
            feat_group_head = feat_group
            # look ahead
            n_rows = 1
            while i+n_rows < table_arr.shape[0] and table_arr[i+n_rows,1].split('--')[0] == feat_group: n_rows+=1
            feat_group_head_short = {
                'EMOTIONS': 'EMO.',
                'RELATIONAL': 'RELAT.',
                'EMBEDDINGS': '',
            }.get(feat_group_head, feat_group_head)
            table_arr[i,0] = f'\multirow{{{n_rows}}}{{*}}{{{sideways(feat_group_head_short)}}}'
            if table_arr[i,1] == 'EMBEDDINGS--HIDDEN':
                feat_subgroup = 'EMBEDDINGS'

        table_arr[i, 1] = feat_subgroup

    # replace 1 and 0 with tick and cross symbols
    for i in range(1,n_features+1):
        for j in range(2,n_datasets+2):
            value_0_1 = table_arr[i,j]
            tick_or_cross = value_0_1.replace('$1$', r'\ding{51}').replace('$0$', r'\ding{55}')
            table_arr[i,j] = tick_or_cross

    # print(table_arr)

    lines = []

    # begin tabular and column format
    col_format = 'c'*(n_datasets+2)
    lines.append('\\footnotesize')
    lines.append('\\begin{tabular}{' + col_format + r'} \toprule')
    lines.append(' & '.join(table_arr[0]) + r'\\ \midrule')
    for i, row_elements in enumerate(table_arr[1:-1]):
        if row_elements[0].startswith(r'\multirow') and i > 0:
            lines.append(r'\hline')
        lines.append(' & '.join(row_elements) + r'\\')
    # lines.append(' & '.join(table_arr[-1]) +  r'\\ \bottomrule')
    lines.append(' & '.join(table_arr[-1]) + r'\\ \midrule')

    # add reference value (all features), optimized value, and relative error reduction

    ref_values = [f'{exploration_reports[d]["reference_all_score"]:.3f}' for d in datasets]
    ref_values_str = ' & '.join(ref_values)
    lines.append(r'\multicolumn{2}{c}{All features} & '+ref_values_str+r' \\')

    ref_values = [f'{exploration_reports[d]["final_score"]:.3f}' for d in datasets]
    ref_values_str = ' & '.join(ref_values)
    lines.append(r'\multicolumn{2}{c}{Optimized features} & '+ref_values_str+r' \\')

    ref_values = [f'{exploration_reports[d]["rel_all_err_reduction"]:.2f}\%' for d in datasets]
    ref_values_str = ' & '.join(ref_values)
    lines.append(r'\multicolumn{2}{c}{Rel. Error Reduction (\%)} & ' + ref_values_str + r' \\')

    lines.append(r'\bottomrule')
    lines.append('\\end{tabular}')

    tabular_str = '\n'.join(lines)

    tabular_dir = 'tables'
    tabular_name = 'selection'
    resizebox = False

    tabular_rel_path = join(tabular_dir, tabular_name + '.tex')
    save_text(text=tabular_str, path=join(out_dir, tabular_rel_path))

    table_str = tex_table(tabular_rel_path, resizebox=resizebox)

    doc_path = pdf_path_name.replace('.pdf', '.tex')
    doc_str = tex_document(doc_path, [table_str], landscape=False, add_package='pifont')
    latex2pdf(pdf_path_name, delete_tex=True, )




if __name__ == '__main__':
    method = 'EMQ'
    baselines = ['MLPE', 'CC', 'PACC']
    method_names = baselines + [method]

    #generate_trends_plots(method_names)
    #generate_auc_tables(method=method)
    generate_selection_table(method=method)

