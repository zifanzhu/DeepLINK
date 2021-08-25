# Title             :deeplink.py
# Description       :Deep Learning Inference Using Knockoffs
# Author            :Zifan Zhu
# Contact           :zifanzhu@usc.edu
# Version           :1.0.0

#### parse arguments ####

import os, sys, optparse

parser = optparse.OptionParser()
parser.add_option("-X", "--data", action="store", type="string",
                  dest="data", help="data matrix in '.npy' format (row sample, column feature)")
parser.add_option("-y", "--response", action="store", type="string",
                  dest="response", help="response vector in '.npy' format")
parser.add_option("-o", "--out", action="store", type="string",
                  dest="output_dir", help="output directory")
parser.add_option("-s", action="store_true", dest="skip_preprocess",
                  help="use this flag if you do not want to center and scale data (center/scale makes every feature column have mean 0/sd 1)")
parser.add_option("-l", "--l1", action="store", type="float", dest="l1",
                  default=0.001, help="l1 regularization coefficient used in the feature selection MLP [default: %default]")
parser.add_option("-r", "--lr", action="store", type="float", dest="lr",
                  default=0.001, help="learning rate used in the feature selection MLP [default: %default]")
parser.add_option("-a", "--act", action="store", type="string",
                  dest="act", default="elu", help="activation function used in the feature selection MLP [default: %default]")
parser.add_option("-L", "--loss", action="store", type="string",
                  dest="loss", default="mean_squared_error", help="loss function used in the feature selection MLP [default: %default]")
parser.add_option("-q", "--fdr_level", action="store", type="float", default=0.2,
                  dest="q", help="fdr level [default: %default]")


options, _ = parser.parse_args()
data = options.data
response = options.response
output_dir = options.output_dir
if data is None or response is None or output_dir is None :
    sys.stderr.write("ERROR: at least one of the required command-line arguments (X, y, o) is missing!\n")
    parser.print_help()
    sys.exit(0)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
l1 = options.l1
lr = options.lr
act = options.act
loss = options.loss
q = options.q

#### import packages ####

import numpy as np
import keras
from keras.layers import Dense, Layer
from keras.models import Sequential
from keras.callbacks import EarlyStopping

#### define pairwise connected layers ####

class PairwiseConnected(Layer):
    def __init__(self, **kwargs):
        super(PairwiseConnected, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0
        self.feat_dim = input_shape[-1] // 2
        self.w = self.add_weight(name="weight", shape=(input_shape[-1],),
                                 initializer="uniform", trainable=True)
        super(PairwiseConnected, self).build(input_shape)

    def call(self, x):
        elm_mul = x * self.w
        output = elm_mul[:, 0:self.feat_dim] + elm_mul[:, self.feat_dim:]

        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.feat_dim
        return tuple(output_shape)

#### load and preprocess data ####

X = np.load(data)
y = np.load(response)
n = X.shape[0]
p = X.shape[1]
if not options.skip_preprocess:
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0, ddof=1)

#### construct knockoffs using autoencoder ####

print("knockoffs construction starts!\n")
es = EarlyStopping(monitor="val_loss", patience=50, verbose=2)
autoencoder = Sequential()
autoencoder.add(Dense(10, activation="elu", use_bias=False, input_shape=(p,)))
autoencoder.add(Dense(p, activation="elu", use_bias=False))
autoencoder.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam())
autoencoder.fit(X, X, epochs=700, batch_size=16, verbose=2, validation_split=0.1, callbacks=[es])
C = autoencoder.predict(X)
E = X - C
sigma = np.sqrt(np.sum(E ** 2) / (n * p))
X_ko = C + sigma * np.random.randn(n, p)
Xnew = np.hstack((X, X_ko))
print("knockoffs construction done!\n")

#### get knockoff statistics using DeepPINK ####

print("knockoff statistics computation starts!\n")
es = EarlyStopping(monitor="val_loss", patience=50, verbose=2)
dp = Sequential()
dp.add(PairwiseConnected(input_shape=(2 * p,)))
dp.add(Dense(p, activation=act,
                kernel_regularizer=keras.regularizers.l1(l1)))
dp.add(Dense(1, activation=act,
                kernel_regularizer=keras.regularizers.l1(l1)))
dp.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=lr))
dp.fit(Xnew, y, epochs=700, batch_size=16, verbose=2, validation_split=0.1, callbacks=[es])

weights = dp.get_weights()
w = weights[1] @ weights[3]
w = w.reshape(p, )
z = weights[0][:p]
z_tilde = weights[0][p:]
W = (w * z) ** 2 - (w * z_tilde) ** 2
print("knockoff statistics computation done!\n")

#### variable selection using knockoff/knockoff+ threshold ####

print("variable selection starts!\n")
t = np.sort(np.concatenate(([0], abs(W))))

ratio = [sum(W <= -tt) / max(1, sum(W >= tt)) for tt in t[:p]]
ind = np.where(np.array(ratio) <= q)[0]
if len(ind) == 0:
    T = float("inf")
else:
    T = t[ind[0]]
selected = np.where(W >= T)[0]
np.savetxt(output_dir + "/selected_variable_ko", selected, fmt='%i')

ratio_plus = [(1 + sum(W <= -tt)) / max(1, sum(W >= tt)) for tt in t[:p]]
ind_plus = np.where(np.array(ratio_plus) <= q)[0]
if len(ind_plus) == 0:
    T_plus = float("inf")
else:
    T_plus = t[ind_plus[0]]
selected_plus = np.where(W >= T_plus)[0]
np.savetxt(output_dir + "/selected_variable_ko+", selected_plus, fmt='%i')
print("variable selection done!\n")
