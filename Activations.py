import numpy as np


class sigmoid(object):
    @staticmethod
    def calc(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def derivative(x):
        z = sigmoid.calc(x)
        return z*(1 - z)


class relu(object):

    @staticmethod
    def calc(x):
        y = x.copy()
        y[x <= 0] = 0.0
        # y = np.maximum(x, 0, x)
        return y

    @staticmethod
    def derivative(x):
        z = x.copy()
        z[x <= 0] = 0.0
        z[x > 0] = 1.0

        return z


class tanh(object):
    @staticmethod
    def calc(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        z = tanh.calc(x)
        return 1.0 - z**2