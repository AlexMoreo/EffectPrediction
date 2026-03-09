from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import numpy as np
import quapy as qp
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')

# implementations from: https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py

def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_rbf_matrix(X, Y, gamma=1.0):
    Z = np.vstack([X,Y])
    ZZ = metrics.pairwise.rbf_kernel(Z, gamma=gamma)
    return ZZ


def mmd_rbf_blocks(X, Y, blocks_idx, gammas=1.):
    if isinstance(gammas, float):
        gammas = [gammas]*len(blocks_idx)

    mmd = 0
    for block_idx, gamma in zip(blocks_idx, gammas):
        mmd+=mmd_rbf(X[:,block_idx], Y[:,block_idx], gamma)

    return mmd


def mmd_rbf_blocks_matrix(X, Y, blocks_idx, gammas=1.):
    if isinstance(gammas, float):
        gammas = [gammas]*len(blocks_idx)

    M = None
    for block_idx, gamma in zip(blocks_idx, gammas):
        M_block = mmd_rbf_matrix(X[:,block_idx], Y[:,block_idx], gamma)
        if M is None:
            M = M_block
        else:
            M += M_block

    return M

def mmd_pairwise_rbf_blocks(Xs, blocks_idx, gammas=1., **kwargs):
    n = len(Xs)
    mmd_matrix = np.zeros(shape=(n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            mmd = mmd_rbf_blocks(Xs[i], Xs[j], blocks_idx, gammas)
            mmd_matrix[i, j] = mmd
            mmd_matrix[j, i] = mmd
    return mmd_matrix


def mmd_rbf_blocks_pval(X, Y, blocks_idx, gammas=1., trials=1000):
    M = mmd_rbf_blocks_matrix(X, Y, blocks_idx, gammas)

    nX, nY = X.shape[0], Y.shape[0]
    nZ = nX + nY

    def MMD_from_matrix(M, nX, nY):
        XX = M[:nX,:nX]
        YY = M[nX:,nX:]
        XY = M[:nX,nX:]
        return XX.mean()+YY.mean()-2*XY.mean()

    mmd = MMD_from_matrix(M, nX, nY)

    mmds_null = []
    for _ in tqdm(range(trials), total=trials):
        idx = np.random.permutation(nZ)
        M = M[np.ix_(idx, idx)]
        mmds_null.append(MMD_from_matrix(M, nX, nY))

    mmds_null = np.asarray(mmds_null)
    p_value = np.mean(mmds_null >= mmd)

    return mmd, p_value


def mmd_pairwise_rbf_blocks_pval(Xs, blocks_idx, gammas=1.):
    n = len(Xs)
    mmd_matrix = np.zeros(shape=(n, n))
    mmd_pvals = np.zeros(shape=(n, n))

    def job(args):
        Xi, Xj, blocks_idx, gammas = args
        return mmd_rbf_blocks_pval(Xi, Xj, blocks_idx, gammas)

    mmds_pvals = qp.util.parallel(job, [(Xs[i], Xs[j], blocks_idx, gammas) for i in range(n-1) for j in range(i+1,n)], n_jobs=-1, asarray=False)
    it=0
    for i in range(n-1):
        for j in range(i+1, n):
            mmd, pval = mmds_pvals[it]
            it+=1
            # mmd, pval = mmd_rbf_blocks_pval(Xs[i], Xs[j], blocks_idx, gammas)
            mmd_matrix[i, j] = mmd
            mmd_matrix[j, i] = mmd
            mmd_pvals[i, j] = pval
            mmd_pvals[j, i] = pval
    return mmd_matrix, mmd_pvals


def AUC_from_result_df(result_df, logscale=False):
    grouped = result_df.groupby('tr_size', sort=True)['nmd'].mean().reset_index()
    tr_size = grouped['tr_size'].tolist()
    nmd_means = grouped['nmd'].tolist()
    auc = np.trapz(y=nmd_means, x=tr_size)
    if logscale:
        auc = np.log(auc)
    return auc
