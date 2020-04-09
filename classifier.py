"""
Multilayer-perceptron for deep classification.
Trained by stochastic gradient descent with momentum.

"""
from autograd import numpy as np, value_and_grad
import pickle

##################################################

class MLP:
    def __init__(self, dimensions, stdev0=1.0):
        self.parameters = [(np.random.normal(0.0, stdev0, size=(outdim, indim)),  # weights
                            np.random.normal(0.0, stdev0, size=outdim))           # biases
                           for indim, outdim in zip(dimensions[:-1], dimensions[1:])]
        self._dldp = value_and_grad(self._l)

    def save(self, filename):
        if filename[-4:] != ".mlp":
            filename = filename + ".mlp"
        with open(filename, 'wb') as file:
            pickle.dump(self.parameters, file, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        if filename[-4:] != ".mlp":
            filename = filename + ".mlp"
        with open(filename, 'rb') as file:
            self.parameters = pickle.load(file)

    def predict(self, inputs):
        return self._f(self.parameters, inputs)

    def train(self, indata, outdata, epochs, step, gain):
        print("==================================================")
        print("MLP: began training")
        self.v_W = [0.0]*len(self.parameters)
        self.v_b = [0.0]*len(self.parameters)
        for epoch in range(epochs):
            loss_avg = 0.0
            for sample in np.random.permutation(len(indata)):
                loss_avg += self.correct(np.atleast_1d(indata[sample]),
                                         np.atleast_1d(outdata[sample]),
                                         step, gain)
            loss_avg /= len(indata)
            if epoch % 10 == 0:
                print("Epoch: {0} | Loss: {1}".format(epoch, loss_avg))
        print("MLP: finished training")
        print("==================================================")

    def correct(self, inputs, outputs, step, gain):
        loss, gradients = self._dldp(self.parameters, inputs, outputs)
        for i, ((W, b), (dldW, dldb)) in enumerate(zip(self.parameters, gradients)):
            self.v_W[i] += gain*(dldW - self.v_W[i])
            self.v_b[i] += gain*(dldb - self.v_b[i])
            W -= step*self.v_W[i]
            b -= step*self.v_b[i]
        return loss

    def _f(self, parameters, inputs):
        for W, b in parameters:
            outputs = np.dot(W, inputs) + b
            inputs = np.tanh(outputs)
        return 0.5*(inputs + 1.0)

    def _l(self, parameters, inputs, outputs):
        error = outputs - self._f(parameters, inputs)
        return np.dot(error, error)

##################################################
