#!/usr/bin/env python3
"""
Messing around with regression.

"""
from autograd import numpy as np, value_and_grad
from matplotlib import pyplot

##################################################

class VanillaNeuralNetwork:
    def __init__(self, dimensions, stdev0):
        self.dimensions = dimensions
        self.parameters = []
        for indim, outdim in zip(dimensions[:-1], dimensions[1:]):
            self.parameters.append((np.random.normal(0.0, stdev0, size=(outdim, indim)),  # weights
                                    np.random.normal(0.0, stdev0, size=outdim)))  # biases
        self._dldp = value_and_grad(self._l)

    def predict(self, inputs):
        return self._f(self.parameters, inputs)

    def correct(self, inputs, outputs, step):
        loss, gradients = self._dldp(self.parameters, inputs, outputs)
        for (W, b), (dldW, dldb) in zip(self.parameters, gradients):
            W -= step*dldW
            b -= step*dldb
        return loss

    def train(self, indata, outdata, epochs, step):
        print("VanillaNeuralNetwork is training!")
        for epoch in range(epochs):
            loss_avg = 0.0
            for sample in np.random.permutation(len(indata)):
                loss_avg += self.correct(np.atleast_1d(indata[sample]),
                                         np.atleast_1d(outdata[sample]),
                                         step)
            loss_avg /= len(indata)
            if epoch % 10 == 0:
                print("Epoch: {0} | Loss: {1}".format(epoch, loss_avg))
        print("VanillaNeuralNetwork done training!")

    def _f(self, parameters, inputs):
        for W, b in parameters:
            outputs = np.dot(W, inputs) + b
            inputs = np.tanh(outputs)
        return outputs

    def _l(self, parameters, inputs, outputs):
        errors = outputs -   self._f(parameters, inputs)
        return np.dot(errors, errors)

##################################################

X = np.arange(-10.0, 10.0, 0.02)
# Y = np.exp(-X**2/10)
Y = (np.exp(np.sin(np.sqrt(np.abs(X)))*X) + 10.0*np.exp(-(X+5)**2))/100

vnn = VanillaNeuralNetwork([1, 5, 1], stdev0=1.0)
vnn.train(X, Y, epochs=50, step=0.01)

Y_approx = np.array([vnn.predict([x]) for x in X])

pyplot.plot(X, Y, label="target")
pyplot.plot(X, Y_approx, label="approx")
pyplot.legend()
pyplot.show()

            # self.v_W += gain*(dldW - self.v_W)
            # self.v_b += gain*(dldb - self.v_b)
