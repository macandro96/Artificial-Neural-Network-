import numpy as np
from numpy import newaxis
from Activations import relu, sigmoid, tanh


class NormalLayer(object):
    def __init__(self, activation_function, neurons, isDropout):
        self.num_neurons = neurons
        self.input = None # Previous Layer activations
        self.zs = None # Wx
        self.activation = activation_function


        if isDropout:
            self.p = 0.5

        else:
            self.p = 1.0

        self.mask = None

    def forwardWithCache(self, z, weights, bias):
        self.input = z

        self.zs = np.dot(weights, z) + bias
        out = self.activation.calc(self.zs)
        self.mask = np.random.randn(out.shape[0], out.shape[1]) < self.p
        out = out*self.mask
        return out

    def forward(self, z, weights, bias):
        out = self.activation.calc(np.dot(weights, z) + bias)*self.p
        return out

    def backward(self, dell):
        func_deriv = self.activation.derivative(self.zs)
        newDell = np.multiply(dell,  func_deriv)
        newDell = newDell*self.mask
        # newDell = newDell*self.mask
        weightUpdate = np.dot(newDell, self.input.transpose())
        return newDell, weightUpdate


class BatchNormalLayer(object):
    def __init__(self, activation_function, neurons, isDropout = False):
        self.num_neurons = neurons
        self.activation = activation_function
        self.cache = None
        self.input = None
        self.running_mean = None
        self.running_variance = None
        self.out = None
        if isDropout:
            self.p = 0.5

        else:
            self.p = 1.0

        self.mask = None

    def forwardWithCache(self, z, weights, gamma, beta):
        self.input = z
        zs = np.dot(weights, z)
        self.out, self.cache = self.batchnorm_forward(zs.transpose(), gamma.transpose(), beta.transpose(), True)
        ans = self.activation.calc(self.out)
        self.mask = np.random.randn(ans.shape[0], ans.shape[1]) < self.p
        ans = ans*self.mask
        return ans


    def forward(self, z, weights, gamma, beta):
        zs = np.dot(weights, z)
        out, _ = self.batchnorm_forward(zs.transpose(), gamma.transpose(), beta.transpose(), False)
        ans = self.activation.calc(out)
        ans = ans*self.p
        return ans

    def backward(self, dell):
        x_hat_deriv = self.activation.derivative(self.out)
        newDell = np.multiply(dell, x_hat_deriv)
        newDell = newDell*self.mask
        dx, dgamma, dbeta = self.batchnorm_backward(newDell.transpose())
        weightUpdate = np.dot(dx, self.input.transpose())
        return dx, weightUpdate, dgamma, dbeta




    def batchnorm_forward(self, X, gamma, beta, train, eps=1e-8):

        if train:
            mu = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            if self.running_mean is None:
                self.running_mean = 0.1*mu
                self.running_variance = 0.1*var
            else:
                self.running_mean = 0.9*self.running_mean + 0.1*(mu)
                self.running_variance = 0.9*self.running_variance + 0.1*(var)
        else:
            mu = self.running_mean
            var = self.running_variance
        X_norm = (X - mu)/np.sqrt(var + eps)
        out = gamma*X_norm + beta
        cache = (X, X_norm, mu, var, gamma, beta)
        return out.transpose(), cache


    def batchnorm_backward(self, dell, eps=1e-8):
        X, X_norm, mu, var, gamma, beta = self.cache
        N, D = X.shape

        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + eps)

        dX_norm = dell * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)


        dx1 = dX_norm * std_inv
        dx2 = (dvar * 2 * X_mu / N)
        dx3 = (dmu / N)
        dX = dx1 + dx2 + dx3
        dgamma = np.sum(dell * X_norm, axis=0)
        dbeta = np.sum(dell, axis=0)
        return dX.transpose(), dgamma, dbeta
