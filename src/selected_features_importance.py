import argparse
import os.path
from os.path import join
from itertools import product
from pathlib import Path

from commons import SAMPLE_SIZE
from data import FEATURE_SUBGROUP_PREFIXES
from feature_block_selection import evaluate_candidates
from plots_and_tables import load_exploration_report



def relative_feature_importance(best_auc_found, selected_features, featureblocks, dataset_dir):
    feat_importance_dict = {}
    for i, feat_block in enumerate(selected_features):
        print(f'measuring importance of: {feat_block} [progress {i}/{len(selected_features)}]')

        new_candidates = list(selected_features)
        new_candidates.remove(feat_block)
        selection_code = [(1 if f in new_candidates else 0) for f in featureblocks]

        auc, path = evaluate_candidates(new_candidates, selection_code, method, dataset_name, n_classes, exploratory_results_dir, dataset_dir)

        if auc > best_auc_found:
            importance = (auc - best_auc_found)/best_auc_found
        else:
            importance = 0.0

        print(f'after deleting {feat_block} we got {auc:.1f} (importance={importance:.4f}/1, optim score={best_auc_found:.1f})')
        feat_importance_dict[feat_block]=importance

    return feat_importance_dict


def write_feature_importance_report(report_path, feat_importance_dict):
    import json
    parent_dir = Path(report_path).parent
    os.makedirs(parent_dir, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as foo:
        json.dump(feat_importance_dict, foo, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the relative importance of each of the selected features")
    parser.add_argument('--dataset', type=str, default='all',
                        help='Choose the dataset; set to "all" (default) for all datasets')
    args = parser.parse_args()

    results_dir = '../results'
    precomputed_results_dir = join(results_dir, 'random_split_features')
    exploratory_results_dir = join(results_dir, 'exploration')
    dataset_dir = '../datasets'
    importance_dir = join(results_dir, 'feat_importance')


    n_classes_list = [5]
    dataset_names = ['diversity', 'toxicity', 'activity' ] if args.dataset=='all' else [args.dataset]
    feature_blocks = FEATURE_SUBGROUP_PREFIXES
    method = 'EMQ'

    for dataset_name, n_classes in product(dataset_names, n_classes_list):
        config_path = f'samplesize{SAMPLE_SIZE}/{dataset_name}/{n_classes}_classes'
        exploration_report = load_exploration_report(method, results_dir, config_path, dataset_name)
        print(exploration_report)

        best_auc_found = exploration_report['final_score']
        selected_features = exploration_report['selected_features']
        featureblocks = FEATURE_SUBGROUP_PREFIXES

        feature_importance_dict = relative_feature_importance(best_auc_found, selected_features, featureblocks, dataset_dir)

        importance_path = join(importance_dir, config_path, 'feat_importance.json')
        write_feature_importance_report(importance_path, feature_importance_dict)
