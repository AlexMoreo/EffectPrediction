import os
import argparse
from os.path import join
from itertools import product
from pathlib import Path

import pandas as pd
import numpy as np

from commons import get_full_path, SAMPLE_SIZE
# from classification import BlockEnsembleClassifier
from data import load_dataset, FEATURE_SUBGROUP_PREFIXES, FEATURE_HIERARCHY, FEATURE_GROUP_PREFIXES
from evaluate_feature_blocks import experiment_label_shift

from utils import AUC_from_result_df

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)




def load_precomputed_reports(result_dir, feature_blocks, dataset_name, n_classes, method):
    # load the AUC obtained by considering each of the feature (sub)blocks independently, and
    # returns a list of the feature blocks in increasing order of quality, i.e., the first element
    # in the list corresponds to the feature (sub)block yielding the worst (highest) result in terms of AUC,
    # meaning that this is the first candidate to be ablated from the pool.
    result_dir = get_full_path(result_dir, dataset_name, n_classes)
    AUCs = []
    for featblock in feature_blocks:
        result_file = join(result_dir, f'{method}__{featblock}.csv')
        assert os.path.exists(result_file), f'result file {result_file} does not exist'
        df = pd.read_csv(result_file, index_col=0)
        auc = AUC_from_result_df(df, logscale=False)
        AUCs.append(auc)
    sorted_feats = sorted(zip(feature_blocks, AUCs), key=lambda x:x[1], reverse=True)
    print(sorted_feats)
    sorted_feats = [f for f, auc in sorted_feats]
    return sorted_feats


def inspect_best_departing_configuration(result_dir, dataset_name, n_classes, method):
    # returns the best configuration (yielding the smallest AUC) for all 1st level feature groups (ACTIVITY, SENTIMENT, ...)
    results = []
    reference_block_names = ['all']+FEATURE_GROUP_PREFIXES
    for level in reference_block_names:
        auc = load_precomputed_result(result_dir, dataset_name, n_classes, method, feature_block=level)
        results.append(auc)
    results = np.asarray(results)
    best_block = reference_block_names[np.argmin(results)]
    return best_block


def load_precomputed_result(result_dir, dataset_name, n_classes, method, feature_block='all'):
    # returns the AUC for the "ALL" features setup. This is the reference value we want to beat
    result_dir = get_full_path(result_dir, dataset_name, n_classes)
    result_file = join(result_dir, f'{method}__{feature_block}.csv')
    assert os.path.exists(result_file), f'result file {result_file} does not exist'
    df = pd.read_csv(result_file, index_col=0)
    auc = AUC_from_result_df(df, logscale=False)
    return auc


def greedy_feature_exploration(baseline_score, baseline_features, featblock_scores_sorted, n_rounds=3):
    best_score = baseline_score
    best_path = None
    contributing_features = list(baseline_features)
    n_blocks = len(featblock_scores_sorted)
    selection_code = []
    for round_idx in range(n_rounds):
        selection_code += [2]*n_blocks  # 1 means the feature set has been chosen, 0 not chosen, 2 not tested
        for i, feat_block in enumerate(featblock_scores_sorted):
            selection_pos = n_blocks*round_idx + i
            print('deciding for feature block: ', feat_block)

            new_candidates = list(contributing_features)
            if feat_block in contributing_features:
                # if it was present, the test consists of removing it
                new_candidates.remove(feat_block)
                ablated = True
                selection_code[selection_pos] = 0
            else:
                # if it was NOT present, the test consists of adding it
                new_candidates = new_candidates + [feat_block]
                ablated = False
                selection_code[selection_pos] = 1

            auc, path = evaluate_candidates(new_candidates, selection_code, method, dataset_name, n_classes, exploratory_results_dir, dataset_dir)

            print(f'after {"deleting" if ablated else "adding"} {feat_block} we got {auc:.1f} (best so far={best_score:.1f})')
            if auc < best_score:
                # keep the modification
                best_score = auc
                best_path = path
                contributing_features = new_candidates
                # selection_code[selection_pos] = 0 if ablated else 1
                print('keep change')
            else:
                # revert the modification
                selection_code[selection_pos] = 1 - selection_code[selection_pos]
                print('revert change')

            print(f'[{dataset_name}] {round_idx=}/{n_rounds} ({i}/{n_blocks}) as for now, we have {len(contributing_features)}/{len(featblock_scores_sorted)} features')
            print(contributing_features)
            print(f'[{dataset_name}] {round_idx=}/{n_rounds} ({i}/{n_blocks}) best {best_score}')

        # replace code "2" (unexplored) with "1" (keep) in the best path prior to returning it
        # best_path = Path(best_path)
        # parent_dir, (method_str, code_str) = best_path.parent, best_path.name.split('__')
        # best_path = join(parent_dir, f'{method_str}__{code_str.replace('2', '1')}')

    return contributing_features, best_score, best_path


def evaluate_candidates(contributing_features, selection_code, method, dataset_name, n_classes, result_dir, dataset_dir):
    features_short_id = ''.join([f'{v}' for v in selection_code])
    eval_report_df, report_path = experiment_label_shift(
        dataset_name,
        n_classes,
        method,
        sample_size=SAMPLE_SIZE,
        features=contributing_features,
        features_short_id=features_short_id,
        result_dir=result_dir,
        dataset_dir=dataset_dir
    )
    candidate_score = AUC_from_result_df(eval_report_df, logscale=False)
    return candidate_score, report_path

def write_exploration_report(report_path, contributing_features, final_score, all_score, best_path):
    import json

    json_report={
        'dataset':dataset_name,
        'n_classes': n_classes,
        'method':method,
        'selected_features':contributing_features,
        'reference_score': all_score,
        'final_score':final_score,
        'rel_err_reduction': 100 * (all_score - final_score) / all_score,
        'best_conf_path':best_path
    }

    with open(report_path, 'w', encoding='utf-8') as foo:
        json.dump(json_report, foo, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch feature block selection.")
    parser.add_argument('--dataset', type=str, default='all',
                        help='Choose the dataset; set to "all" (default) for all datasets')
    args = parser.parse_args()

    precomputed_results_dir = '../results/random_split_features'
    exploratory_results_dir = '../results/exploration'
    dataset_dir = '../datasets'

    n_classes_list = [5]
    dataset_names = ['activity', 'toxicity', 'diversity'] if args.dataset=='all' else [args.dataset]
    feature_blocks = FEATURE_SUBGROUP_PREFIXES
    method = 'EMQ'


    for dataset_name, n_classes in product(dataset_names, n_classes_list):
        best_block = inspect_best_departing_configuration(precomputed_results_dir, dataset_name, n_classes, method)
        baseline_features = FEATURE_HIERARCHY[best_block]

        baseline_score = load_precomputed_result(precomputed_results_dir, dataset_name, n_classes, method, feature_block=best_block)
        featblock_scores_sorted = load_precomputed_reports(precomputed_results_dir, feature_blocks, dataset_name, n_classes, method)

        print(f'Best score: {baseline_score:.3f} for block={best_block}')
        print(f'features sorted: {featblock_scores_sorted}')

        contributing_features, final_score, best_path = greedy_feature_exploration(
            baseline_score, baseline_features, featblock_scores_sorted
        )

        report_path = get_full_path(exploratory_results_dir, dataset_name, n_classes)
        report_path = join(report_path, f'{method}_exploration.json')
        write_exploration_report(report_path, contributing_features, final_score, baseline_score, best_path)


