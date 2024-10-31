import numpy as np
import quapy as qp
from quapy.data import LabelledCollection, Dataset
import quapy.functional as F
from quapy.method.aggregative import PACC, EMQ, CC, KDEyML, PCC
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from quapy.protocol import UPP
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
from classification import BlockEnsembleClassifier


from data import load_dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

path = '../datasets/activity_dataset'
# path = '../datasets/toxicity_dataset'
# path = '../datasets/diversity_dataset'


def fit_and_test(quantifier, train, test, error_fn):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
    quantifier.fit(train)
    pred_prev = quantifier.quantify(test.X)
    err = error_fn(test.prevalence(), pred_prev)
    return err, pred_prev


def experiment(i):
    sel = subreddits[:, i].astype(bool)
    train = LabelledCollection(covariates[~sel], labels[~sel])
    test = LabelledCollection(covariates[sel], labels[sel])

    zscorer = StandardScaler()
    train.instances = zscorer.fit_transform(train.instances)
    test.instances = zscorer.transform(test.instances)

    qp.environ["SAMPLE_SIZE"] = len(test)

    # quant = EMQ(CalibratedClassifierCV(LogisticRegression()), n_jobs=-1)
    # quant = PCC(CalibratedClassifierCV(LogisticRegression(), n_jobs=-1))

    mlpe = MaximumLikelihoodPrevalenceEstimation()
    cc = CC(LogisticRegression())
    # pcc = PCC(LogisticRegression())
    pcc = PCC(BlockEnsembleClassifier(LogisticRegression(), column_names=cov_names))
    quant_pps = PACC(LogisticRegression())
    # quant_pps = PACC(BlockEnsembleClassifier(LogisticRegression(), column_names=cov_names))

    # train, val = train.split_stratified(train_prop=0.6, random_state=0)
    # quant = qp.model_selection.GridSearchQ(
    #     model= quant,
    #     param_grid={'classifier__C': np.logspace(-4,4,7), 'classifier__class_weight': ['balanced', None]}, # 'bandwidth': np.linspace(0.01, 0.2, 20)},
    #     protocol=UPP(val),
    #     refit=True,
    #     n_jobs=-1,
    #     verbose=True
    # ).fit(train).best_model()

    err_noshift, _ = fit_and_test(mlpe, train, test, error_fn=qp.error.ae)
    err_cc, _ = fit_and_test(cc, train, test, error_fn=qp.error.ae)
    err_pcc, pcc_prev_hat = fit_and_test(pcc, train, test, error_fn=qp.error.ae)
    err_quant, quant_prev_hat = fit_and_test(quant_pps, train, test, error_fn=qp.error.ae)

    print(f"test-prevalence=\t{F.strprev(test.prevalence())}")
    print(f"train-prevalence=\t{F.strprev(train.prevalence())}\t{err_noshift:.5f}")
    print(f"PCC-prevalence=\t{F.strprev(pcc_prev_hat)}\t{err_pcc:.5f}")
    print(f"Quant-prevalence=\t{F.strprev(quant_prev_hat)}\t{err_quant:.5f}")
    print()

    return err_noshift, err_cc, err_pcc, err_quant


cov_names, covariates, labels, subreddits_names, subreddits = qp.util.pickled_resource(path + '.pkl', load_dataset, path)

n_subreddits = len(subreddits_names)
print(cov_names)

results = qp.util.parallel(experiment, np.arange(n_subreddits), n_jobs=-1, asarray=False, backend='loky')

mlpe_errors, cc_errors, pcc_errors, q_errors = list(zip(*results))



print('-'*80)
print('ERRORS:')
print('-'*80)
print(f'MLPE = {np.mean(mlpe_errors):.4f}+-{np.std(mlpe_errors):.4f}')
print(f'CC = {np.mean(cc_errors):.4f}+-{np.std(cc_errors):.4f}')
print(f'PCC = {np.mean(pcc_errors):.4f}+-{np.std(pcc_errors):.4f}')
print(f'Quant = {np.mean(q_errors):.4f}+-{np.std(q_errors):.4f}')


"""
activity:
MLPE = 0.0736+-0.0207
CC = 0.1214+-0.0353
PCC = 0.0415+-0.0235 <- with LR
PCC = 0.0403+-0.0240 <- with blocks of LR
Quant = 0.3297+-0.0616

toxicity:
MLPE = 0.0461+-0.0253
CC = 0.1361+-0.0409
PCC = 0.0335+-0.0259 <- with LR
PCC = 0.0330+-0.0255 <- with blocks of LR
Quant = 0.3016+-0.0915

diversity:
MLPE = 0.0629+-0.0298
CC = 0.1374+-0.0371
PCC = 0.0595+-0.0226 <- with LR
PCC = 0.0576+-0.0266 <- with blocks of LR
Quant = 0.3528+-0.0967
"""