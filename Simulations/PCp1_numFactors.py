import numpy as np

def PCp1(X, rmax):
    n = X.shape[0]
    p = X.shape[1]
    PC = np.zeros(rmax+1)
    X_C = X - np.mean(X, axis=0)
    SX = X_C / np.std(X_C, axis=0, ddof=1)
    mXX = SX @ SX.T
    for k in range(rmax, -1, -1):
        #print('k = {}'.format(k))
        if k == 0:
            PC[k] = np.sum(SX ** 2 / (n * p))
        else:
            w, v = np.linalg.eigh(mXX)
            meigvec = v[:, -k:]
            mF = np.sqrt(n) * meigvec
            Lam = (mF.T @ SX) / n
            if k == rmax:
                sigma2 = np.sum((SX - mF @ Lam) ** 2) / (n*p)
            PC[k] = np.sum((SX - mF @ Lam) ** 2) / (n*p) + \
                    k * sigma2 * ((n + p) / (n * p)) * np.log((n * p) / (n + p))
    return np.where(PC == np.min(PC))[0]
