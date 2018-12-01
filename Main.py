import numpy as np
from Network import Network

import pandas as pd


train = pd.read_csv('data/emnist-letters-train.csv')
test = pd.read_csv('data/emnist-letters-test.csv')
trainMat = train.as_matrix()
testMat = test.as_matrix()

"""Taking the first 9 alphabets for training and testing"""
trainMat = trainMat[trainMat[:, 0] <= 9]   # Train: 30677
testMat = testMat[testMat[:, 0] <= 9]      # Test: 7199

#
Y_temp = trainMat[:,0]
Y_test_temp = testMat[:,0]
#
X = trainMat[:,1:]
X_test = testMat[:,1:]
Y = np.zeros(shape = [Y_temp.shape[0], 9])
Y_test = np.zeros(shape = [Y_test_temp.shape[0], 9])


k = 0
"""Creating one hot encoding"""
for t in Y_temp:
    Y[k,t-1] = 1
    k += 1

k = 0
for t in Y_test_temp:
    Y_test[k,t-1] = 1
    k = k + 1

"""Adding Noise to X"""
# X = X + np.random.normal(0,0.1, 784)

"""Normalizing train and test data"""
X = X.astype(float)
X = X/255.0
mean = np.mean(X, axis=0)
std = np.std(X, axis=0) + 1e-4
X = (X - mean)/std

X_test = X_test.astype(float)
X_test = X_test/255.0
mean_test = np.mean(X_test, axis=0)
std_test = np.std(X_test, axis=0) + 1e-4
X_test = (X_test - mean_test)/std_test



isBatchNorm = True
isDropout = False
weight_initialization = 'Gaussian' # Use 'Xavier' and 'He' for Xavier and He initializations
                                # and Gaussian for normal initialization
layers = [(784, 'input'), (128, 'sigmoid'), (64, 'sigmoid'), (32, 'sigmoid'), (9, 'output')]
net = Network(layers,  weight_initialization, isBatchNorm, isDropout)


epochs = 100
learning_rate = 0.1
mini_batch_size = 100
lmbda = 1e-1 # Decay factor for regularization
reg = 'None' # Use 'L1' and 'L2' for L1 and L2
isMSE = False

"""Train on train data and test on test data"""
"""Prints loss and accuracy for every iteration"""

net.trainAndTest(X,Y,epochs,learning_rate, mini_batch_size, X_test
           , Y_test, lmbda, reg, isMSE)

