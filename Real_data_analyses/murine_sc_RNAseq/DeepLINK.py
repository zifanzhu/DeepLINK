import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from pairwise_connected_layer import PairwiseConnected
from itertools import combinations

# Design matrix generation from factor model
# Input:
#   n: number of observations
#   p: number of variables
#   r: number of factors
#   theta: signal to noise ratio
#   design: form of the factor model, choose from:
#           1. 'linear': linear factor model
#           2. 'add_quad': additive quadratic factor model
#           3. 'logistic': logistic factor model
# Output:
#   [factor matrix, loading matrix, design matrix]


def dmg(n, p, r, theta, design):
    if design == 'linear':
        Fa = np.random.randn(n, r)
        Lam = np.random.randn(r, p)
        E = np.random.randn(n, p)
        # X = Fa @ Lam + np.sqrt(r * theta) * E
        X = np.matmul(Fa, Lam) + np.sqrt(r * theta) * E
    elif design == 'add_quad':
        def interactions(arr):
            return [a * b for a, b in combinations(arr, r=2)]

        Fa = np.random.randn(n, r)
        Inte = np.apply_along_axis(interactions, axis=1, arr=Fa)
        Quad = np.hstack((Fa, Fa ** 2, Inte))
        Lam = np.random.randn(Quad.shape[1], p)
        E = np.random.randn(n, p)
        # X = Quad @ Lam + np.sqrt(r * theta) * E
        X = np.matmul(Quad, Lam) + np.sqrt(r * theta) * E
    elif design == 'logistic':
        def logistic(f, lam):
            # f : r-dim
            # lam : (r + 2)-dim
            return lam[0] / (1 + np.exp(lam[1] - np.dot(lam[2:], f)))

        Fa = np.random.randn(n, r)
        Lam = np.random.randn(r + 2, p)
        FL = np.array([[logistic(f, lam) for lam in Lam.T] for f in Fa])
        E = np.random.randn(n, p)
        X = FL + np.sqrt(r * theta) * E

    return [Fa, Lam, X]


# Response vector generation
# Input:
#   X: design matrix
#   s: level of sparsity
#   A: amplitude of signals
#   c: signal to noise ratio
#   design: link function form, choose from:
#           1. 'linear': linear model
#           2. 'nonlinear': noninear sin*exp model
# Output:
#   [coefficient vector, response vector, noiseless response]

def rvg(X, s, A, c, design):
    n = X.shape[0]
    p = X.shape[1]
    btrue = np.random.choice([A, -A], size=s)
    bfalse = np.repeat(0, p - s)
    beta = np.concatenate((btrue, bfalse)).reshape((p, 1))
    np.random.shuffle(beta)
    epsilon = np.random.randn(n, 1)
    if design == 'linear':
        # nl_y = X @ beta
        nl_y = np.matmul(X, beta)
        y = nl_y + np.sqrt(c) * epsilon
    elif design == 'nonlinear':
        nl_y = np.sin(np.matmul(X, beta)) * np.exp(np.matmul(X, beta))
        y = nl_y + np.sqrt(c) * epsilon
    return [beta, y, nl_y]


# Knockoff matrix generation by autoencoder
# Input:
#   X: design matrix
#   r: number of factors
#   met: activation method
#   epoch: number of training epochs
#   loss: loss function used in training (default: 'mean_squared_error')
#   verb: verbose level (default: 2)
# Output:
#   Xnew: [X, X_knockoff]

def knockoff_construct(X, r, met, epoch, loss='mean_squared_error', verb=2):
    n = X.shape[0]
    p = X.shape[1]

    # use autoencoder to estimate C
    autoencoder = Sequential()
    autoencoder.add(Dense(r, activation=met, use_bias=False, input_shape=(p,)))
    autoencoder.add(Dense(p, activation=met, use_bias=False))
    autoencoder.compile(loss=loss, optimizer=keras.optimizers.Adam())
    autoencoder.fit(X, X, epochs=epoch, batch_size=32, verbose=verb)

    C = autoencoder.predict(X)

    # construct X_knockoff

    E = X - C
    sigma = np.sqrt(np.sum(E ** 2) / (n * p))
    X_ko = C + sigma * np.random.randn(n, p)
    Xnew = np.hstack((X, X_ko))
    return Xnew


# DeepPINK deep neural network with knockoffs
# Input:
#   X: design matrix ([X, X_knockoff])
#   y: response vector
#   met: activation method
#   epoch: number of training epochs
#   loss: loss function used in training (default: 'mean_squared_error')
#   verb: verbose level (default: 2)
# Output:
#   W: knockoff statistics


def knockoff_stats(X, y, met, epoch, loss='mean_squared_error', verb=2):
    p = X.shape[1] // 2

    # implement DeepPINK
    dp = Sequential()
    dp.add(PairwiseConnected(input_shape=(2 * p,)))
    dp.add(Dense(p, activation=met,
                    kernel_regularizer=keras.regularizers.l1(0.01)))
    dp.add(Dense(1, activation=None,
                    kernel_regularizer=keras.regularizers.l1(0.01)))
    dp.compile(loss=loss, optimizer=keras.optimizers.Adam())
    dp.fit(X, y, epochs=epoch, batch_size=32, verbose=verb)

    # calculate knockoff statistics W_j
    weights = dp.get_weights()
    # w = weights[1] @ weights[3]
    w = np.matmul(weights[1], weights[3])
    w = w.reshape(p, )
    z = weights[0][:p]
    z_tilde = weights[0][p:]
    W = (w * z) ** 2 - (w * z_tilde) ** 2
    return W


# Feature selection with knockoff/knockoff+ threshold
# Input:
#   W: knockoff statistics
#   q: FDR level
#   ko_plus: indicate whether to use knockoff+ (True) or
#            knockoff (False) threshold [default: True]
# Output:
#   array of discovered variables


def knockoff_select(W, q, ko_plus=True):
    # find the knockoff threshold T
    p = len(W)
    t = np.sort(np.concatenate(([0], abs(W))))
    if ko_plus:
        ratio = [(1 + sum(W <= -tt)) / max(1, sum(W >= tt)) for tt in t[:p]]
    else:
        ratio = [sum(W <= -tt) / max(1, sum(W >= tt)) for tt in t[:p]]
    ind = np.where(np.array(ratio) <= q)[0]
    if len(ind) == 0:
        T = float('inf')
    else:
        T = t[ind[0]]

    # set of discovered variables
    return np.where(W >= T)[0]


# FDP calculation
# Input:
#   S: array of discovered variables
#   beta_true: true coefficient vector
# Output:
#   false discovery proportion


def fdp(S, beta_true):
    return float(sum(beta_true[S] == 0)) / float(max(1, len(S)))
    #return sum(beta_true[S] == 0) / max(1, len(S))


# Power calculation
# Input:
#   S: array of discovered variables
#   beta_true: true coefficient vector
# Output:
#   true discovery proportion


def pow(S, beta_true):
    return float(sum(beta_true[S] != 0)) / float(sum(beta_true != 0))
    # return sum(beta_true[S] != 0) / sum(beta_true != 0)
