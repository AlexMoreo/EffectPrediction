from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning


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


def mmd_rbf_blocks(X, Y, blocks_idx, gammas=1.):
    if isinstance(gammas, float):
        gammas = [gammas]*len(blocks_idx)

    mmd = 0
    for block_idx, gamma in zip(blocks_idx, gammas):
        mmd+=mmd_rbf(X[:,block_idx], Y[:,block_idx], gamma)

    return mmd


def mmd_rbf_blocks_pval(X, Y, blocks_idx, gammas, trials=1000):
    mmd = mmd_rbf_blocks(X, Y, blocks_idx, gammas)

    nX, nY = X.shape[0], Y.shape[0]
    Z = np.vstack([X,Y])
    mmds_null = []
    for _ in range(trials):
        Z = np.random.permutation(Z)
        X_, Y_ = Z[:nX], Z[nX:]
        mmd_null = mmd_rbf_blocks(X_, Y_, blocks_idx, gammas)
        mmds_null.append(mmd_null)

    mmds_null = np.asarray(mmds_null)
    p_value = np.mean(mmds_null >= mmd)

    return p_value


if __name__ == '__main__':
    import numpy as np

    from data import load_dataset
    from classification import separate_blocks_idx

    path = '../datasets/datasets_periods/activity_dataset'

    print('loading dataset...')
    cov_names, covariates, labels, subreddits_names, subreddits = load_dataset(path, with_periods=True)
    column_prefixes, prefix_idx = separate_blocks_idx(cov_names)
    blocks_idx = prefix_idx.values()
    X, y = covariates, labels
    n = X.shape[0]
    X1 = X[subreddits[:,0].astype(bool)]
    X2 = X[subreddits[:,1].astype(bool)]
    print('start')
    # print(mmd_rbf(X1, X2))
    print(mmd_rbf_blocks(X1, X2, blocks_idx))



    # a = np.arange(1, 10).reshape(3, 3)
    # b = [[7, 6, 5], [4, 3, 2], [1, 1, 8], [0, 2, 5]]
    # b = np.array(b)
    # print(a)
    # print(b)
    # # print(mmd_linear(a, b))  # 6.0
    # print(mmd_rbf(a, b))  # 0.5822
    # mmd = mmd_blocks(a, b, slices=[slice(0,2),slice(2,3)], gammas = 1.)
    # print(mmd)
