import lasagne
import theano
import theano.tensor as T
import gzip
import numpy as np
import matplotlib.pyplot as plt
import config as Cfg
from datasets.preprocessing import extract_norm_and_out
# create Theano variables for input and target minibatch
input_var = T.tensor4('X')
target_var = T.ivector('y')

# create a small convolutional neural network
from lasagne.nonlinearities import leaky_rectify, softmax

network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
network = lasagne.layers.Conv2DLayer(network, num_filters=8, filter_size=(5, 5))
network = lasagne.layers.Pool2DLayer(network, (2, 2), mode='max')
network = lasagne.layers.Conv2DLayer(network, num_filters=4, filter_size=(5, 5))
network = lasagne.layers.Pool2DLayer(network, (2, 2), stride=2, mode='max')
network = lasagne.layers.DenseLayer(network, num_units=32)
network = lasagne.layers.DenseLayer(network, num_units=10)
network = lasagne.layers.NonlinearityLayer(network, name="softmax",nonlinearity=lasagne.nonlinearities.softmax)

# create loss function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()  #+ 1e-4 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.0001,
                                            momentum=0.9)
# compile training function that updates parameters and returns training loss
train_fn = theano.function([input_var, target_var], loss, updates=updates)






def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # reshaping and normalizing
    data = data.reshape(-1, 1, 28, 28).astype(np.float32)
    return data


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

data_path = "/data/tjdx_user/Desktop/LeNet/data/"

X = load_mnist_images('%strain-images-idx3-ubyte.gz' %
                              data_path)
Y = load_mnist_labels('%strain-labels-idx1-ubyte.gz' %
                              data_path)
X_test = load_mnist_images('%st10k-images-idx3-ubyte.gz' %
                                   data_path)
y_test = load_mnist_labels('%st10k-labels-idx1-ubyte.gz' %
                                   data_path)

# # set normal and anomalous class
# normal = [0,1]
# outliers = [2,3]
#
# # extract normal and anomalous class
# X_norm, X_out, y_norm, y_out = extract_norm_and_out(X, y, normal=normal, outlier=outliers)
#
# # reduce outliers to fraction defined
# n_norm = len(y_norm)
# n_out = int(np.ceil(Cfg.out_frac * n_norm / (1 - Cfg.out_frac)))
#
# # shuffle to obtain random validation splits
# np.random.seed(0)
# perm_norm = np.random.permutation(len(y_norm))
# perm_out = np.random.permutation(len(y_out))
#
# # split into training and validation set
# n_norm_split = int(Cfg.mnist_val_frac * n_norm)
# n_out_split = int(Cfg.mnist_val_frac * n_out)
# _X_train = np.concatenate((X_norm[perm_norm[n_norm_split:]],
#                                 X_out[perm_out[:n_out][n_out_split:]]))
# _y_train = np.append(y_norm[perm_norm[n_norm_split:]],
#                           y_out[perm_out[:n_out][n_out_split:]])
# _X_val = np.concatenate((X_norm[perm_norm[:n_norm_split]],
#                               X_out[perm_out[:n_out][:n_out_split]]))
# _y_val = np.append(y_norm[perm_norm[:n_norm_split]],
#                         y_out[perm_out[:n_out][:n_out_split]])
#
# # shuffle data (since batches are extracted block-wise)
# n_train = len(_y_train)
# n_val = len(_y_val)
# perm_train = np.random.permutation(n_train)
# perm_val = np.random.permutation(n_val)
# _X_train = _X_train[perm_train]
# _y_train = _y_train[perm_train]
# _X_val = _X_train[perm_val]
# _y_val = _y_train[perm_val]
#
# # Subset train set such that we only get batches of the same size
# n_train = (n_train / Cfg.batch_size) * Cfg.batch_size
# subset = np.random.choice(len(_X_train), n_train, replace=False)
# _X_train = _X_train[subset]
# _y_train = _y_train[subset]
#
# # Adjust number of batches
# Cfg.n_batches = int(np.ceil(n_train * 1. / Cfg.batch_size))
#
# # test set
# X_norm, X_out, y_norm, y_out = extract_norm_and_out(X_test, y_test, normal=normal, outlier=outliers)
# _X_test = np.concatenate((X_norm, X_out))
# _y_test = np.append(y_norm, y_out)
# perm_test = np.random.permutation(len(_y_test))
# _X_test = _X_test[perm_test]
# _y_test = _y_test[perm_test]
# n_test = len(_y_test)


# train network (assuming you've got some training data in numpy arrays)
for epoch in range(100):
    loss = 0
    for i in range(0,len(Y)-130,128):
        input_batch = X[i:i+128]
        target_batch = Y[i:i+128]
        # plt.imshow(input_batch[0],cmap='gray')
        # plt.axis('off')
        # plt.show()
        # print input_batch
        # print target_batch
        # exit()
        loss1 = train_fn(input_batch, target_batch)
        loss += loss1
        print loss1/128
    print("Epoch %d: Loss %g" % (epoch + 1, loss / len(X)))

# use trained network for predictions
test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
print("Predicted class for first test input: %r" % predict_fn(X_test[0]))
