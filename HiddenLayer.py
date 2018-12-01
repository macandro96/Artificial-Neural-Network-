from Layers import NormalLayer, BatchNormalLayer
from Activations import relu, tanh, sigmoid

class HiddenLayer(object):

    def __init__(self, type, neurons, isBatchNormal, isDropout):
        self.isBatchNormal = isBatchNormal

        if type == 'relu':
            if isBatchNormal:

                self.layer = BatchNormalLayer(relu, neurons)
            else:
                self.layer = NormalLayer(relu, neurons, isDropout)
        elif type == 'sigmoid':
            if isBatchNormal:
                self.layer = BatchNormalLayer(sigmoid, neurons)
            else:
                self.layer = NormalLayer(sigmoid, neurons, isDropout)
        else:
            if isBatchNormal:
                self.layer = BatchNormalLayer(tanh, neurons)
            else:
                self.layer = NormalLayer(tanh, neurons, isDropout)




    def forwardWithCache(self, inputs, weights, bias, gamma = 0.0, beta = 0.0):

        # change for batch norm
        if self.isBatchNormal:
            out = self.layer.forwardWithCache(inputs, weights, gamma, beta)
        else:
            out = self.layer.forwardWithCache(inputs, weights, bias)
        return out

    def forward(self, inputs, weights, bias, gamma = 0.0, beta = 0.0):
        # change for batch norm
        if self.isBatchNormal:
            out = self.layer.forward(inputs, weights, gamma, beta)
        else:
            out = self.layer.forward(inputs, weights, bias)
        return out

    def backward(self, dell, gamma = 0.0, beta = 0.0):
        dgamma = None
        dbeta = None
        # change for batch norm
        if self.isBatchNormal:
            newDell, weightUpdate, dgamma, dbeta = self.layer.backward(dell)
        else:
            newDell, weightUpdate = self.layer.backward(dell)
        return newDell, weightUpdate, dgamma, dbeta

