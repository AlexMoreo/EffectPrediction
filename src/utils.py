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

if __name__ == '__main__':
    import numpy as np
    from time import time

    from data import load_dataset
    from classification import separate_blocks_idx

    path = '../datasets/activity_dataset'

    print('loading dataset...')
    cov_names, covariates, labels, subreddits_names, subreddits = load_dataset(path, with_periods=True)
    column_prefixes, prefix_idx = separate_blocks_idx(cov_names)
    blocks_idx = prefix_idx.values()
    X, y = covariates, labels
    n_subreddits = len(subreddits_names)

    for i in range(n_subreddits-1):
        for j in range(i+1, n_subreddits):
            X1 = X[subreddits[:,i].astype(bool)]
            X2 = X[subreddits[:,j].astype(bool)]

            # print('start')
            #
            # tinit = time()
            # print(mmd_rbf_blocks(X1, X2, blocks_idx))
            # tend = time()-tinit
            # print(f'took {tend}s')
            #
            # tinit = time()
            # n1=X1.shape[0]
            # m = mmd_rbf_blocks_matrix(X1, X2, blocks_idx)
            # xx = m[:n1,:n1]
            # yy = m[n1:, n1:]
            # xy = m[:n1, n1:]
            # mmd = xx.mean() + yy.mean() - 2 * xy.mean()
            # print(f'{mmd:.5f} took {time()-tinit}s')

            tinit = time()
            mmd, pval = mmd_rbf_blocks_pval(X1, X2, blocks_idx)
            print(f'{mmd:.5f} took {time()-tinit}s {pval=:.5f}')
            if pval > 0.1:
                # Fail to reject the null hypothesis, meaning we do not have enough evidence
                # to claim that X and Y come from different distributions.
                # In other words, X and Y could come from the same distribution,
                # but we cannot be certain.
                print(f'\t take {i}, {j}')
            if pval <= 0.1:
                # Reject the null hypothesis, meaning it is very unlikely that
                # X and Y come from the same distribution. This suggests that
                # there are significant differences between the distributions of X and Y.
                pass

            # print(mmd_rbf_blocks(X1, X2, blocks_idx))



            # a = np.arange(1, 10).reshape(3, 3)
            # b = [[7, 6, 5], [4, 3, 2], [1, 1, 8], [0, 2, 5]]
            # b = np.array(b)
            # print(a)
            # print(b)
            # # print(mmd_linear(a, b))  # 6.0
            # print(mmd_rbf(a, b))  # 0.5822
            # mmd = mmd_blocks(a, b, slices=[slice(0,2),slice(2,3)], gammas = 1.)
            # print(mmd)
