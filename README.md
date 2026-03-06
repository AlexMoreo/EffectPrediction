# Quantification for Content Monitoring Effect Prediction

This repo contains experimental code for assessing the importance of behavioral features in the 
task of effect prediction of the application of content monitoring policies.

The Reddit datasets (activity / toxicity / diversity) contains 9 feature blocks, and a total of
69 feature subgroups. The task is to infer which feature subgroups carry over the most 
meaningful signal for effect prediction.

The repo (_src/_) consists of two main scripts:

* _evaluate_feature_blocks.py_:  evaluates feature blocks using different quantification algorithms 
  and stores results in disc, so they are available as pre-computed resources for other scripts. This script
  defines a main experiment in which a specific set features are evaluated in the task of predicting
  behavioral changes after the policy intervention, in terms of normalized match distance and using a standard
  sampling generation protocol for simulating prior probability shift.
* _feature_block_selection.py_: implements a greedy search on combinations of feature groups. Relies
  on the intermediate outputs of the former script and reuses its core _experiment_ method

Other scripts are mainly devoted to analyze the results:

* _selected_features_importance.py_: for each of the selected features, compares the performance
  with and without it and analyzes the delta, as a proxy for relative feature importance
* _plots_and_tables.py_: as the name suggests, this scripts generates plots and tables we report
  in the paper (under submission).


