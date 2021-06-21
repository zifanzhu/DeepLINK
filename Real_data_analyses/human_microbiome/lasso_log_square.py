import DeepLINK as dl
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from pairwise_connected_layer import PairwiseConnected
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegressionCV

dataset = ['Zeller_CRC']

def clr(arr):
    geomean = np.mean(np.log(arr[arr > 0]))
    return np.array([np.log(n) - geomean if n > 0 else 0 for n in arr])

for ds in dataset:
    X = np.genfromtxt(ds + '_known_raw.csv', delimiter=',', skip_header=1)
    X = np.apply_along_axis(clr, 1, X)
    # center_scale data
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0, ddof=1)
    n = X.shape[0]
    p = X.shape[1]

    r_hat = 10
    # print('r_hat:' + str(r_hat))

    aut_epoch = 100
    aut_loss = 'mean_squared_error'
    aut_verb = 0
    aut_met = 'relu'
    q = 0.2
    rep = 30

    result = np.repeat(np.repeat([[0]], 5, 0), rep + 2, axis=1)
    colnames = ['mean', 'SE'] + [str(obj) for obj in range(1, rep + 1)]
    result = pd.DataFrame(result, index=['FDR', 'Power', 'FDR+', 'Power+', 'Pred_errors'], columns=colnames)

    for i in range(rep):
        print(ds + '_y_' + str(i) + '.npy')
        Gy = np.load(ds + '_y_' + str(i) + '.npy')
        y = Gy[1]
        true_beta = Gy[0]

        # es = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
        autoencoder = Sequential()
        autoencoder.add(Dense(r_hat, activation=aut_met, use_bias=False, input_shape=(p,)))
        autoencoder.add(Dense(p, activation=aut_met, use_bias=False))
        autoencoder.compile(loss=aut_loss, optimizer=keras.optimizers.Adam())
        autoencoder.fit(X, X, epochs=aut_epoch, batch_size=8, verbose=aut_verb)
        C = autoencoder.predict(X)
        E = X - C
        sigma = np.sqrt(np.sum(E ** 2) / (n * p))
        X_ko = C + sigma * np.random.randn(n, p)
        Xnew = np.hstack((X, X_ko))

        log = LogisticRegressionCV(penalty='l1', solver='liblinear', n_jobs=-1, cv=10).fit(Xnew, y.reshape((n, )))
        beta = log.coef_[0]
        W = (beta[:p]) ** 2 - (beta[p:]) ** 2

        t = np.sort(np.concatenate(([0], abs(W))))

        ratio = [sum(W <= -tt) / max(1, sum(W >= tt)) for tt in t[:p]]
        ind = np.where(np.array(ratio) <= q)[0]
        if len(ind) == 0:
            T = float('inf')
        else:
            T = t[ind[0]]
        selected = np.where(W >= T)[0]

        ratio_plus = [(1 + sum(W <= -tt)) / max(1, sum(W >= tt)) for tt in t[:p]]
        ind_plus = np.where(np.array(ratio_plus) <= q)[0]
        if len(ind_plus) == 0:
            T_plus = float('inf')
        else:
            T_plus = t[ind_plus[0]]
        selected_plus = np.where(W >= T_plus)[0]

        result.iloc[0, i + 2] = dl.fdp(selected, true_beta)
        result.iloc[1, i + 2] = dl.pow(selected, true_beta)
        result.iloc[2, i + 2] = dl.fdp(selected_plus, true_beta)
        result.iloc[3, i + 2] = dl.pow(selected_plus, true_beta)
        result.iloc[4, i + 2] = sum(log.predict(Xnew) != y.flatten())

        # backup

        # result.to_csv(ds + '_lasso_log_square_backup', index=True, header=True, sep=',')

    result.iloc[:, 0] = np.mean(result.iloc[:, 2:], axis=1)
    result.iloc[:, 1] = np.std(result.iloc[:, 2:], axis=1, ddof=1) / np.sqrt(rep)

    result.to_csv(ds + '_lasso_log_square.csv', index=True, header=True, sep=',')
