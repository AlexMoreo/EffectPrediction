import os
import pathlib
from os.path import join
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from commons import SAMPLE_SIZE, N_CLASSES, RESULT_DIR, N_BATCHES, N_RUNS
from comparison_group import SelectByName, ColourGroup
from data import FEATURE_SUBGROUP_PREFIXES, load_dataset
from evaluate_feature_blocks import replicable_partition
from format import FormatModifierSelectColor
# from format import FormatModifierSelectColor
import quapy as qp
from new_table import LatexTable, save_text
from tools import tex_table, tex_document, latex2pdf
from utils import AUC_from_result_df, _load_exploration_report

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

DATASETS = ['activity', 'toxicity', 'diversity']

FIGURES_DIR = '../results/figures'
TABLES_DIR  = '../results/tables'


def plot_trend_by_methods(report_list, path_name, plotsize, legend, title=''):
    """
    Main plots showing NMD error trends (y-axis) as a function of the training size (x-axis) for
    different methods across the three datasets.
    """
    df = pd.concat(report_list)

    df["method_features"] = df["method"]
    df["method_features"] = df["method_features"].str.replace(r"MLPE .*", "MLPE", regex=True)
    df["method_features"] = df["method_features"].str.replace(r"EMQ optimized.*", "EMQO", regex=True)

    sns.set(style="whitegrid")

    plt.figure(figsize=plotsize)
    sns.lineplot(data=df, x="tr_size", y="nmd", hue="method_features", marker="o", palette="tab10")

    plt.xlabel("Training size")
    plt.ylabel("NMD error")
    plt.title(title)

    if legend:
        plt.legend(loc='upper right')
    else:
        plt.legend().remove()
    plt.ylim(0.05,0.3)

    plt.tight_layout()
    os.makedirs(pathlib.Path(path_name).parent, exist_ok=True)
    plt.savefig(path_name)


def plot_err_by_shift(report_list, training_prevalences, path_name, plotsize, legend, title=''):
    """
    Plots showing NMD error trends (y-axis) as a function of the amount of shift (x-axis) as measured
    by the absolute error between the training prevalence and the test prevalence of each experiment.
    Results are binned in quantiles of experiments.
    """
    os.makedirs(pathlib.Path(path_name).parent, exist_ok=True)

    df = pd.concat(report_list)

    df["method_features"] = df["method"]
    df["method_features"] = df["method_features"].str.replace(r"MLPE .*", "MLPE", regex=True)
    df["method_features"] = df["method_features"].str.replace(r"EMQ optimized.*", "EMQO", regex=True)

    def compute_shift(row):
        train_prev = training_prevalences[row.dataset][row.run][row.tr_size]
        return qp.error.mae(row["true-prev"], train_prev)

    # convert string prevalences to np.ndarray
    df["true-prev"] = df["true-prev"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
    )
    df["shift"] = df.apply(compute_shift, axis=1)

    print(df)

    sns.set(style="whitegrid")

    plt.figure(figsize=plotsize)

    # df["shift_bin"] = pd.cut(df["shift"], bins=20)
    df["shift_bin"] = pd.qcut(df["shift"], q=10, duplicates='drop')
    df["shift_mid"] = df["shift_bin"].apply(lambda x: x.mid)
    sns.lineplot(data=df, x="shift_mid", y="nmd", hue="method_features", marker="o", palette="tab10")

    plt.xlabel("Shift (AE)")
    plt.ylabel("NMD error")
    plt.title(title)

    if legend:
        plt.legend(loc='upper right')
    else:
        plt.legend().remove()
    plt.ylim(0.05,0.3)

    plt.tight_layout()
    os.makedirs(pathlib.Path(path_name).parent, exist_ok=True)
    plt.savefig(path_name)


def _compute_AUC(report_list, dataset, add_duplicate=True):
    # Auxiliary function: returns area under the curve for a given method
    df = pd.concat(report_list)
    assert len(df.method.unique())==1, 'unexpected number of methods'
    datasets = df.dataset.unique()
    assert len(datasets) == 1, 'unexpected number of datasets'
    assert datasets[0] == dataset, 'unexpected dataset'

    auc_dict = []
    for features in df.features.unique():
        df_sel = df[df['features']==features]
        aucs = AUC_from_result_df(df_sel)

        if features == 'all':
            father, soon = 'root', 'all'
        elif '--' in features:
            father, soon = features.split('--')
        else:
            father, soon = 'root', features

        for run, auc_run in enumerate(aucs):
            auc_dict.append({'features': features, 'father': father, 'soon': soon, 'auc': auc_run, 'dataset': dataset, 'run': run})

            # adds a duplicate that serves as a reference method for the children
            if add_duplicate and father=='root' and soon != 'all':
                auc_dict.append({'features': features, 'father': soon, 'soon': soon + " (full)", 'auc': auc_run, 'dataset': dataset, 'run': run})

    auc_df = pd.DataFrame(auc_dict)
    return auc_df


def _load_dataset_results(dataset, method_names):
    config_path = f'samplesize{SAMPLE_SIZE}/{dataset}/{N_CLASSES}_classes'
    results = []
    for m in method_names:
        result_path = join(RESULT_DIR, 'random_split_features', config_path, f'{m}__all.csv')
        df = pd.read_csv(result_path, index_col=0)
        results.append(df)

        # for the selected method, it also loads the exploration report and shows the optimized features trend
        if m == method:
            optim_path = _load_exploration_report(method, RESULT_DIR, config_path, dataset)['best_conf_path']
            df = pd.read_csv(optim_path, index_col=0)
            df['method'] = df['method'].replace('EMQ', 'EMQ optimized')
            results.append(df)
    return results


def _reconstruct_experimental_training_prevalences(dataset_dir):
    # this script reconstructs all training prevalences (that we had not initially saved in our runs);
    # since everything is seeded, the splits are perfectly replicable
    training_prevs = {}
    for dataset in tqdm(DATASETS, desc='datasets'):
        data = load_dataset(f'{dataset_dir}/{dataset}_dataset', n_classes=N_CLASSES)
        training_prevs[dataset] = {}
        for seed in range(N_RUNS):
            training_prevs[dataset][seed] = {}
            training_pool, _, batch_size, random_order, _ = replicable_partition(data.X, data.y, np.arange(N_CLASSES), N_BATCHES, seed=seed)
            for batch in range(N_BATCHES):
                tr_selection = random_order[: (batch + 1) * batch_size]  # incremental selections
                train = training_pool.sampling_from_index(tr_selection)
                train_size = len(train)
                train_prev = train.prevalence()
                training_prevs[dataset][seed][train_size] = train_prev

    return training_prevs


def generate_trends_plots(method_names, out_dir=f'{FIGURES_DIR}/random_split_features/'):
    # invokes the plot_trend_by_methods for every dataset
    for idx, dataset in enumerate(DATASETS):
        results = _load_dataset_results(dataset, method_names)

        # generates a plot comparing the trends of all methods
        path_name = join(out_dir, f'samplesize{SAMPLE_SIZE}_{N_CLASSES}_classes', f'{dataset}.pdf')
        plot_trend_by_methods(results, path_name, plotsize=(5,5), legend=(idx==1), title=f'{dataset}')


def generate_err_by_shift_plots(method_names, out_dir=f'{FIGURES_DIR}/err_by_shift'):
    # invokes the plot_err_by_shift for every dataset

    # prepare folder
    training_prevs = _reconstruct_experimental_training_prevalences('../datasets')

    for idx, dataset in enumerate(DATASETS):
        results = _load_dataset_results(dataset, method_names)

        # generates a plot comparing the trends of all methods
        path_name = join(out_dir, f'samplesize{SAMPLE_SIZE}_{N_CLASSES}_classes', f'{dataset}.pdf')
        plot_err_by_shift(results, training_prevs, path_name, plotsize=(5,5), legend=(idx==1), title=f'{dataset}')


def generate_auc_tables(method, out_dir=TABLES_DIR):
    # generates tables of AUC for all datasets and for all feature groups, including "ALL", the first level, and the
    # subgroups
    auc_df_by_dataset = []
    for dataset in DATASETS:
        config_path = f'samplesize{SAMPLE_SIZE}/{dataset}/{N_CLASSES}_classes'
        result_path = join(RESULT_DIR, 'random_split_features', config_path, f'EMQ__*.csv')

        reports = []
        for csv_path in glob(result_path):
            method_features = pathlib.Path(csv_path).name.replace('.csv', '')
            method_name = method_features.split('__')[0]
            if method_name==method:
                df = pd.read_csv(csv_path, index_col=0)
                reports.append(df)

        auc_df = _compute_AUC(reports, dataset)
        auc_df_by_dataset.append(auc_df)

    final_df = pd.concat(auc_df_by_dataset)

    tables = []
    for father in final_df.father.unique():
        # if father!='root': continue
        auc_sel_df = final_df[final_df.father==father]

        table = LatexTable.from_dataframe(auc_sel_df, method='soon', benchmark='dataset', value='auc', name=father)
        table.format.configuration.show_std = True
        table.format.configuration.stat_test = None
        table.format.configuration.side_columns = True
        table.format.configuration.mean_prec=1
        table.format.configuration.std_prec=2
        table.format.configuration.transpose = True
        columns = list(table.methods)
        last = [f for f in columns if '(full)' in f or f=='all']  # the father is the only one without "(full)"
        rest  = sorted([f for f in columns if f not in last])
        table.reorder_methods(rest+last)
        table.reorder_benchmarks(['activity', 'toxicity', 'diversity'])
        tables.append(table)

    pdf_path_name = join(out_dir, f'auc_features_{method}_{N_CLASSES}_classes.pdf')
    LatexTable.LatexPDF(pdf_path=pdf_path_name, tables=tables, dedicated_pages=False)


def generate_featorder_table(method, out_dir=f'{TABLES_DIR}/tables'):
    # generates a table displaying the order in which the features are explored (which depends on the individual
    # AUC for NMD); the elements in the table are color-coded by super group.

    table = {}
    for dataset in DATASETS:

        feature_names = []
        feature_aucs = []
        feature_aucs_std = []

        config_path = f'samplesize{SAMPLE_SIZE}/{dataset}/{N_CLASSES}_classes'
        result_path = join(RESULT_DIR, 'random_split_features', config_path, f'EMQ__*.csv')

        for csv_path in glob(result_path):
            if '--' not in pathlib.Path(csv_path).name:
                # keep only feature blocks and not feature groups nor config "all"
                continue
            method_features = pathlib.Path(csv_path).name.replace('.csv', '')
            method_name, feature_name = method_features.split('__')
            if method_name==method:
                df = pd.read_csv(csv_path, index_col=0)
                auc = AUC_from_result_df(df)
                auc_ave = np.mean(auc)
                auc_std = np.std(auc)
                feature_name = feature_name.replace('_', r'\_').replace('--', ':')
                feature_names.append(feature_name)
                feature_aucs.append(auc_ave)
                feature_aucs_std.append(auc_std)

        feature_names = np.asarray(feature_names)
        feature_aucs = np.asarray(feature_aucs)
        feature_aucs_std = np.asarray(feature_aucs_std)

        order = np.argsort(-feature_aucs)
        feature_names = feature_names[order]
        feature_aucs  = feature_aucs[order]
        feature_aucs_std = feature_aucs_std[order]

        table[f'{dataset}']={'names': feature_names, 'aucs': feature_aucs, 'aucs_std': feature_aucs_std}
    print(table)

    num_features = len(table[DATASETS[0]]['names'])

    lines = []

    parent_colors = {
        r'SOC\_PSY': 'lightblue',
        'RELATIONAL': 'lightgreen',
        'TOXICITY': 'lightpink',
        'EMBEDDINGS': 'lightyellow',
        'LIWC': 'lightgray',
        'SENTIMENT': 'lightpurple',
        'ACTIVITY': 'lightorange',
        r'WRITING\_STYLE': 'lightteal',
        'EMOTIONS': 'lightred',
    }

    # fallback color if any unexpected parent name appears
    default_color = 'white'

    # Extract parent name from feature
    def get_parent(feature_name):
        return feature_name.split(':')[0].strip()

    # Begin tabular environment with one column for rank and two for each dataset
    lines.append(r'\begin{tabular}{c' + '|rc' * len(DATASETS) + '}')
    lines.append(r'\toprule')

    # First header row: dataset names as multicolumns
    header = [r'\multicolumn{1}{c}{}']  # empty cell for rank column
    for dataset in DATASETS:
        header.append(r'\multicolumn{2}{c}{\textsc{' + dataset + '}}')
    lines.append(' & '.join(header) + r' \\')

    # Second header row: column names under each dataset
    subheader = [r'\multicolumn{1}{c}{\textbf{rank}}']
    for _ in DATASETS:
        subheader.extend([r'\multicolumn{1}{c}{feature name}', r'\multicolumn{1}{c}{MNMD}'])
    lines.append(' & '.join(subheader) + r' \\')

    lines.append(r'\midrule')

    # Add the table rows with rank
    for i in range(num_features):
        row = [str(i + 1)]  # rank starts at 1
        for dataset in DATASETS:
            name = table[dataset]['names'][i]
            auc = table[dataset]['aucs'][i]
            std = table[dataset]['aucs_std'][i]
            parent = get_parent(name)
            color = parent_colors.get(parent, default_color)
            colored_name = r'\cellcolor{' + color + '}' + r'\texttt{' + name + '}'
            colored_auc = r'\cellcolor{' + color + '}' + f'{auc:.3f}$\pm${std:.3f}'
            row.extend([colored_name, colored_auc])
        lines.append(' & '.join(row) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    tabular = '\n'.join(lines)

    # show
    print(tabular)

    tabular_path_name = join(out_dir, f'auc_featuorder_{method}_{N_CLASSES}_classes.tex')
    with open(tabular_path_name, 'wt') as foo:
        foo.write(tabular)


def generate_selection_table(method, out_dir=TABLES_DIR):

    exploration_reports = {}
    selected_features = {}
    for dataset in DATASETS:
        config_path = f'samplesize{SAMPLE_SIZE}/{dataset}/{N_CLASSES}_classes'

        exploration_report = _load_exploration_report(method, RESULT_DIR, config_path, dataset)
        selected_features[dataset] = exploration_report['selected_features']
        exploration_reports[dataset] = exploration_report

    n_features = len(FEATURE_SUBGROUP_PREFIXES)
    n_datasets = len(DATASETS)
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
        for dataset in DATASETS:
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
            table_arr[i,0] = f'\\multirow{{{n_rows}}}{{*}}{{{sideways(feat_group_head_short)}}}'
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

    ref_values = [f'{exploration_reports[d]["reference_all_score"]:.3f}$\pm${exploration_reports[d]["reference_all_score_std"]:.3f}'  for d in DATASETS]
    ref_values_str = ' & '.join(ref_values)
    lines.append(r'\multicolumn{2}{c}{All features} & '+ref_values_str+r' \\')

    ref_values = [f'{exploration_reports[d]["final_score"]:.3f}$\pm${exploration_reports[d]["final_score_std"]:.3f}' for d in DATASETS]
    ref_values_str = ' & '.join(ref_values)
    lines.append(r'\multicolumn{2}{c}{Optimized features} & '+ref_values_str+r' \\')

    ref_values = [f'{exploration_reports[d]["rel_all_err_reduction"]:.2f}\\%' for d in DATASETS]
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

    # generate_trends_plots(method_names)
    # generate_err_by_shift_plots(method_names)
    # generate_auc_tables(method=method)
    # generate_selection_table(method=method)
    generate_featorder_table(method=method)



