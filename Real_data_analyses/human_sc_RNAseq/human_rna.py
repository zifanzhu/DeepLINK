import os
import random

import DeepLINK as dl
import numpy as np
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from pairwise_connected_layer import PairwiseConnected
import pandas as pd
from keras.callbacks import EarlyStopping

ds = 'human_rna'
vs = 1

d = 20

X0 = np.genfromtxt(ds + '.csv', delimiter=',', skip_header=1)
y = X0[:, 23257]
X = X0[:,0:23257]

indmat_dist = np.genfromtxt('indmat_dist_p50.csv', delimiter=',', skip_header=1)
top200 = np.genfromtxt('top200_p50.csv', delimiter=',', skip_header=1) - 1

# center_scale data
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0, ddof=1)
n = X.shape[0]
p = X.shape[1]

ep = 500
bs = 15

aut_epoch = ep
aut_loss = 'mean_squared_error'
aut_verb = 0
dnn_epoch = ep
dnn_loss = 'binary_crossentropy'
dnn_verb = 0
aut_met = 'relu'
dnn_met = 'elu'
q = 0.2
nrep = 100

mat_selected = np.zeros([nrep, d])
mat_selected_plus = np.zeros([nrep, d])

# n_train = int(round(n*0.6))
# n_test = int(round(n*0.1))
n_train = 253
n_test = 63
indmat = np.zeros([n_train, nrep])
yhat_train = np.zeros([nrep, n_train])
yhat_test = np.zeros([nrep, n_test])
pe_test = [0.]*nrep
pe_train = [0.]*nrep

for i in range(nrep):
    
    ind_col = top200[range(d),i].astype(int)
    X1 = X[:,ind_col]
    ## autoencoder ##
    r_hat = 3
    autoencoder = Sequential()
    autoencoder.add(Dense(d, activation=aut_met, use_bias=True, input_shape=(d,)))
    # autoencoder.add(Dense(p, activation=aut_met, use_bias=True))
    autoencoder.add(Dense(r_hat, activation=aut_met, use_bias=True))
    # autoencoder.add(Dense(p, activation=aut_met, use_bias=True))
    autoencoder.add(Dense(d, activation=aut_met, use_bias=True))
    autoencoder.compile(loss=aut_loss, optimizer=keras.optimizers.Adam())
    autoencoder.fit(X1, X1, epochs=aut_epoch, batch_size=bs, verbose=aut_verb)
    C = autoencoder.predict(X1)
    E = X1 - C
    sigma = np.sqrt(np.sum(E ** 2) / (n * d))
    X1_ko = C + sigma * np.random.randn(n, d)
    Xnew = np.hstack((X1, X1_ko))
    #################

    random.seed(58*i)
    ######### load data #########
    ind_dist = indmat_dist[i,:] - 1
    ind_dnn = list(set(np.arange(n)) - set(ind_dist))
    
    ind_train = np.random.choice(ind_dnn, n_train, False)
    ind_test = list(set(ind_dnn) - set(ind_train))
    
    Xnew_train = Xnew[ind_train,:]
    Xnew_test = Xnew[ind_test,:]
    y_train = y[ind_train]
    y_test = y[ind_test]
    
    indmat[:,i] = ind_train
    #############################    
    
    dp = Sequential()
    dp.add(PairwiseConnected(input_shape=(2 * d,)))
    dp.add(Dense(d, activation=dnn_met,
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
    dp.add(Dropout(0.4)) # play with this number, such as 0.4, 0.6, 0.7
    dp.add(Dense(1, activation='relu',
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
    dp.compile(loss=dnn_loss, optimizer=keras.optimizers.Adam())
    dp.fit(Xnew_train, y_train, epochs=dnn_epoch, batch_size=bs, verbose=dnn_verb)

    weights = dp.get_weights()
    w3 = np.matmul(weights[1], weights[2]).reshape(d,)
    w1 = np.multiply(weights[0][:d], w3)
    w2 = np.multiply(weights[0][d:], w3)
    W = w1**2 - w2**2

    t = np.sort(np.concatenate(([0], abs(W))))
    
    ratio = [float(sum(W <= -tt)) / float(max(1, sum(W >= tt))) for tt in t[:d]]
    ind = np.where(np.array(ratio) <= q)[0]
    if len(ind) == 0:
        T = float('inf')
    else:
        T = t[ind[0]]
    
    selected = np.where(W >= T)[0]

    print(selected)
    mat_selected[i,:] = W >= T
    
    ratio_plus = [float((1 + sum(W <= -tt))) / float(max(1, sum(W >= tt))) for tt in t[:d]]
    ind_plus = np.where(np.array(ratio_plus) <= q)[0]
    if len(ind_plus) == 0:
        T_plus = float('inf')
    else:
        T_plus = t[ind_plus[0]]
    
    selected_plus = np.where(W >= T_plus)[0]
    
    mat_selected_plus[i,:] = W >= T_plus

    yhat_train[i,:] = [1 if a > 0.5 else 0 for a in dp.predict(Xnew_train).flatten()]
    pe_train[i] = np.mean(yhat_train[i,:] != y_train.flatten())
    
    ## refit to calculate PE ##
    if len(selected) == 0:
        if np.mean(y_train) >= 0.5:
            yhat_test[i,:] = [1.]*n_test
        else:
            yhat_test[i,:] = [0.]*n_test
        pe_test[i] = np.mean(yhat_test[i,:] != y_test.flatten())
    else:
        s = len(selected)
        mrefit = Sequential()
        mrefit.add(Dense(s, input_dim=s, activation='relu'))
        mrefit.add(Dense(s, activation='relu'))
        mrefit.add(Dense(1, activation='sigmoid'))
        mrefit.compile(loss=dnn_loss, optimizer=keras.optimizers.Adam())
        mrefit.fit(X1[ind_train,:][:,selected], y[ind_train], epochs=300, batch_size=bs, verbose=dnn_verb)
        
        yhat_test[i,:] = [1 if a > 0.5 else 0 for a in mrefit.predict(X1[ind_test,:][:,selected]).flatten()]
        pe_test[i] = np.mean(yhat_test[i,:] != y_test.flatten())
    ###########################
    
nam1 = 'v'+str(vs)+'.csv'
pd.DataFrame(pe_train).to_csv(ds + '_real_pe_train_' + nam1, index=True, header=True, sep=',')
pd.DataFrame(pe_test).to_csv(ds + '_real_pe_test_' + nam1, index=True, header=True, sep=',')

print('Training average PE: ')
print(np.mean(pe_train))
print('Test average PE: ')
print(np.mean(pe_test))

pd.DataFrame(mat_selected).to_csv(ds + '_real_selected_' + nam1, index=True, header=True, sep=',')
pd.DataFrame(mat_selected_plus).to_csv(ds + '_real_selected_plus_' + nam1, index=True, header=True, sep=',')
pd.DataFrame(indmat).to_csv(ds + '_real_indmat_' + nam1, index=True, header=True, sep=',')
pd.DataFrame(yhat_test).to_csv(ds + '_real_yhat_test_' + nam1, index=True, header=True, sep=',')
pd.DataFrame(yhat_train).to_csv(ds + '_real_yhat_train_' + nam1, index=True, header=True, sep=',')
pd.DataFrame(pe_train).to_csv(ds + '_real_pe_train_' + nam1, index=True, header=True, sep=',')
pd.DataFrame(pe_test).to_csv(ds + '_real_pe_test_' + nam1, index=True, header=True, sep=',')


