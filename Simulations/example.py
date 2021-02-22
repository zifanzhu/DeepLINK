import DeepLINK as dl
import numpy as np
import pandas as pd
from PCp1_numFactors import PCp1 as PCp1

x_design = 'linear' # factor model design
y_design = 'linear' # link function design
r = 3 # number of factors
n = 1000 # number of observations
p = 500 # number of features
s = 10 # number of true signals
A = 5 # signal amplitude
q = 0.2 # target FDR level
it = 100 # number of iterations
aut_epoch = 300 # number of autoencoder training epochs
aut_loss = 'mean_squared_error' # loss function used in autoencoder training
aut_verb = 0 # verbose level of autoencoder
mlp_epoch = 300 # number of mlp training epochs
mlp_loss = 'mean_squared_error' # loss function used in mlp training
mlp_verb = 0 # verbose level of mlp
l1 = 0.001 # l1 regularization factor in mlp
lr = 0.001 # learning rate for mlp training

result = np.repeat(np.repeat([[0]], 4, 0), it + 2, axis=1)
r_est = np.zeros(it, dtype=int)
colnames = ['mean', 'sd'] + [str(obj) for obj in range(1, it + 1)]
result = pd.DataFrame(result, index=['FDR', 'Power', 'FDR+', 'Power+'], columns=colnames)

for i in range(it):
  print('Run_' + str(i + 1))
  GX = dl.dmg(n, p, r, x_design)
  X = GX[2]
  X /= np.sqrt(np.sum(X ** 2, axis = 0))
  r_hat = PCp1(X, 15)[0]
  r_est[i] = r_hat
  Gy = dl.rvg(X, s, A, y_design)
  y = Gy[1]
  true_beta = Gy[0]

  # construct knockoffs

  Xnew = dl.knockoff_construct(X, r_hat, 'elu', aut_epoch, aut_loss, aut_verb)

  # compute knockoff statistics

  W = dl.knockoff_stats(Xnew, y, 'elu', mlp_epoch, mlp_loss, l1, lr, mlp_verb)

  # feature selection

  # selected = dl.knockoff_select(W, q, ko_plus=False)
  selected_plus = dl.knockoff_select(W, q, ko_plus=True)

  # result.iloc[0, i + 2] = dl.fdp(selected, true_beta)
  # result.iloc[1, i + 2] = dl.pow(selected, true_beta)
  result.iloc[2, i + 2] = dl.fdp(selected_plus, true_beta)
  result.iloc[3, i + 2] = dl.pow(selected_plus, true_beta)

  # back-up

  result.to_csv('elu_elu_backup', index=True, header=True, sep=',')

result.iloc[:, 0] = np.mean(result.iloc[:, 2:], axis=1)
result.iloc[:, 1] = np.std(result.iloc[:, 2:], axis=1, ddof=1)
result.to_csv('elu_elu.csv', index=True, header=True, sep=',')
