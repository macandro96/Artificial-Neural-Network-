import numpy as np
from numpy import newaxis
from HiddenLayer import HiddenLayer
import matplotlib.pyplot as plt
import random

class Network(object):
    def __init__(self, layers, weight_initialization, isBatchNorm = False, isDropout = False ):
        self.numLayers = list()
        self.hiddenLayers = list()
        self.isBatchNorm = isBatchNorm
        self.running_mean = None
        self.running_variance = None
        counter = 0
        for layer,type in layers:
            if counter == 0:
                self.numLayers.append(layer)
            elif counter == len(layers) - 1:
                self.numLayers.append(layer)

            else:
                self.numLayers.append(layer)
                hiddenLayer = HiddenLayer(type, layer, self.isBatchNorm, isDropout)
                self.hiddenLayers.append(hiddenLayer)
            counter += 1

        # Different Weight Initializations
        if weight_initialization == 'He':
            self.weights = [np.random.randn(y, x) / np.sqrt(x/2) for x, y in zip(self.numLayers[:-1], self.numLayers[1:])]
        elif weight_initialization == 'Xavier':
            self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in
                            zip(self.numLayers[:-1], self.numLayers[1:])]
        else:
            self.weights = [np.random.randn(y, x) for x, y in
                            zip(self.numLayers[:-1], self.numLayers[1:])]


        if isBatchNorm:

            # self.biases = [np.zeros(x, 1) for x in self.numLayers[1:]]

            self.gamma = [np.random.randn(x, 1) for x in self.numLayers[1:len(self.numLayers)]]
            self.beta = [np.random.randn(x, 1) for x in self.numLayers[1:len(self.numLayers)]]
            self.biases = [np.random.randn(x, 1) for x in self.numLayers[1:]]
            self.biases = np.array(self.biases)*0.0

        else:

            self.biases = [np.random.randn(x, 1) for x in self.numLayers[1:]]

            self.gamma = [np.random.randn(x, 1) for x in self.numLayers[1:len(self.numLayers)]]
            self.gamma = np.array(self.gamma) * 0.0
            self.beta = [np.random.randn(x, 1) for x in self.numLayers[1:len(self.numLayers)]]
            self.beta = np.array(self.beta)*0.0


    def forwardWithCache(self, x):
        z = x
        for i in xrange(0, len(self.hiddenLayers)):
            z = self.hiddenLayers[i].forwardWithCache(z, self.weights[i], self.biases[i], self.gamma[i], self.beta[i])
        zs = np.dot(self.weights[-1], z) + self.biases[-1]

        out = self.softMax(zs)
        return out,z

    def forward(self, x):
        z = x
        for i in xrange(0, len(self.hiddenLayers)):
            z = self.hiddenLayers[i].forward(z, self.weights[i], self.biases[i], self.gamma[i], self.beta[i])


        zs = np.dot(self.weights[-1], z) + self.biases[-1]
        out = self.softMax(zs)
        return out

    def softMax(self, out):
        z = np.exp(out)
        sums = np.sum(z, axis=0)
        return z/sums[None,:]


    def softMaxDerivative(self, out, mini_batch_size):

        deriv_temp = out[:,:,newaxis]
        deriv_temp = deriv_temp.reshape([1,self.numLayers[-1],mini_batch_size])


        deriv1 = np.repeat(deriv_temp[:,:,:], self.numLayers[-1], axis=0)
        eye = np.identity(self.numLayers[-1], dtype=np.float64)

        temp = eye[:,:,newaxis]
        temp = np.repeat(temp[:,:,:], mini_batch_size, axis=2)
        deriv2 = deriv_temp.reshape([self.numLayers[-1],1,mini_batch_size])
        deriv2 = np.repeat(deriv2[:,:,:], self.numLayers[-1], axis=1)

        deriv2 = temp - deriv2
        deriv = np.multiply(deriv1, deriv2)
        return deriv

    def trainAndTest(self, X, Y, epochs, eps, mini_batch_size, X_test, Y_test, lmbda = 0.0, reg = 'None', mse = False):
        train_costs, train_accuracies, test_costs, test_accuracies = list(), list(), list(), list()
        for j in xrange(0, epochs):
            train_sample = zip(X, Y)
            random.shuffle(train_sample)
            #
            batches = [train_sample[k:k + mini_batch_size] for k in range(0, len(X), mini_batch_size)]

            for batch in batches:
                x, y = zip(*batch)
                x = np.array(x)
                y = np.array(y)

                dw, db, dgamma, dbeta = self.backprop(x, y, x.shape[0], mse)

                if reg == 'L2':
                    self.weights = self.weights - eps*np.array(dw) - ((eps*lmbda)/mini_batch_size)*np.array(self.weights)
                elif reg == 'L1':

                    for i in xrange(0, len(self.numLayers)-1):
                        self.weights[i] = self.weights[i] - eps*dw[i] - ((lmbda*eps)/mini_batch_size)*np.sign(dw[i])


                else:
                    self.weights = self.weights - eps * np.array(dw)

                for i in xrange(0, len(self.numLayers) - 1):

                    kp = db[i].reshape([max(db[i].shape), 1])
                    kp2 = dgamma[i].reshape([dgamma[i].shape[0], 1])
                    kp3 = dbeta[i].reshape([dbeta[i].shape[0], 1])
                    self.biases[i] = self.biases[i] - eps * kp
                    self.gamma[i] = self.gamma[i] - eps * kp2
                    self.beta[i] = self.beta[i] - eps * kp3
            if j % 1 == 0:
                test_out = self.forward(X_test.transpose())
                test_cost, test_acc = self.calc_loss(test_out, Y_test.transpose(), mse)
                train_out = self.forward(X.transpose())
                train_cost, train_acc = self.calc_loss(train_out, Y.transpose(), mse)
                train_costs.append(train_cost)
                train_accuracies.append(train_acc)
                test_costs.append(test_cost)
                test_accuracies.append(test_acc)
                print "Train: ", train_cost, train_acc, "Test: ",test_cost, test_acc

        plt.plot(range(0, epochs), train_costs, 'r', label = 'train_costs')
        plt.plot(range(0, epochs), test_costs, 'b', label='test_costs')
        plt.xlabel('epoch')
        plt.ylabel('cost')
        plt.legend()
        plt.show()

        plt.plot(range(0, epochs), train_accuracies, 'r', label = 'train_accuracy')
        plt.plot(range(0, epochs), test_accuracies, 'b', label='test_accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()



    def calc_loss(self, out, Y, lmbda=0.0, reg = 'None', mse=False):
        if not mse:
            temp = np.multiply(Y, np.nan_to_num(np.log(out)))
            loss = -1*np.sum(temp)/out.shape[1]
        else:
            error = Y - out
            loss = .5*np.sum(np.multiply(error, error))

        l2_w = 0.0
        l1_w = 0.0
        for w in self.weights:
            l2_w = l2_w + np.sum(w*w)
            l1_w = l1_w + np.sum(abs(w))


        reg_term = 0.0
        if reg == 'L1':
            reg_term = lmbda/(2*out.shape[1])*l1_w
        elif reg == 'L2':
            reg_term = lmbda/(2*out.shape[1])*l2_w
        loss = loss + reg_term
        outArg = np.argmax(out, axis=0)
        yArg = np.argmax(Y, axis=0)
        diff = (outArg-yArg)
        mishits = np.count_nonzero(diff)
        hits = 1.0 - mishits/float(Y.shape[1])

        return loss, hits

    def backprop(self, X, Y, mini_batch_size, mse = False):
        dw = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.numLayers[:-1], self.numLayers[1:])]
        db = [np.random.randn(x, 1) for x in self.numLayers[1:]]
        dg = [np.random.randn(x,1) for x in self.numLayers[1:]]
        dbeta = [np.random.randn(x, 1) for x in self.numLayers[1:]]
        dg = np.array(dg) * 0.0
        dbeta = np.array(dbeta) * 0.0


        if self.isBatchNorm:
            db = np.array(db)*0.0

        out,z = self.forwardWithCache(X.transpose())
        error = out - Y.transpose()
        if mse:
            softMaxDeriv = self.softMaxDerivative(out, mini_batch_size)

            delt = error[:, :, newaxis]
            delt = delt.reshape([1, self.numLayers[-1], mini_batch_size])
            delta2 = np.einsum('mnr,ndr->mdr', delt, softMaxDeriv)
            # if not self.isBatchNorm:
            db[-1] = np.sum(delta2, axis=2) / mini_batch_size
            dell = delta2[0, :, :]
            dw_temp = np.dot(dell, z.transpose())
            dw[-1] = dw_temp / mini_batch_size
        else:
            dell = error

            db[-1] = np.sum(error, axis=1)/mini_batch_size

            dw_temp = np.dot(dell, z.transpose())
            dw[-1] = dw_temp/mini_batch_size


        for i in xrange(2, len(self.numLayers)):
            hiddenLayer = self.hiddenLayers[-i+1]
            dell = np.dot(self.weights[-i + 1].transpose(), dell)

            if self.isBatchNorm:
                dell, weightUpdate, dgamma, dbet = hiddenLayer.backward(dell=dell,
                                                                        )
                dg[-i] = dgamma / mini_batch_size
                dbeta[-i] = dbet / mini_batch_size
            else:
                dell, weightUpdate, _, _ = hiddenLayer.backward(dell=dell)
                db[-i] = np.sum(dell, axis=1)/mini_batch_size

            dw[-i] = weightUpdate / mini_batch_size

        return dw, db, dg, dbeta
