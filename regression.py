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

    def correct(self, inputs, outputs, step, gain):
        loss, gradients = self._dldp(self.parameters, inputs, outputs)
        for i, ((W, b), (dldW, dldb)) in enumerate(zip(self.parameters, gradients)):
            self.v_W[i] += gain*(dldW - self.v_W[i])
            self.v_b[i] += gain*(dldb - self.v_b[i])
            W -= step*self.v_W[i]
            b -= step*self.v_b[i]
        return loss

    def train(self, indata, outdata, epochs, step, gain):
        print("VanillaNeuralNetwork is training!")
        self.v_W = [0.0]*(len(self.dimensions)-1)
        self.v_b = [0.0]*(len(self.dimensions)-1)
        for epoch in range(epochs):
            loss_avg = 0.0
            for sample in np.random.permutation(len(indata)):
                loss_avg += self.correct(np.atleast_1d(indata[sample]),
                                         np.atleast_1d(outdata[sample]),
                                         step, gain)
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

class DoubleHump:
    def __init__(self):
        self.parameters = np.array([0.1, 2, -10.0, 0.2, 2, 10.0])
        self._dldp = value_and_grad(self._l)

    def predict(self, inputs):
        return self._f(self.parameters, inputs)

    def correct(self, inputs, outputs, step, gain):
        loss, gradients = self._dldp(self.parameters, inputs, outputs)
        self.v += gain*(gradients - self.v)
        self.parameters -= step*self.v
        return loss

    def train(self, indata, outdata, epochs, step, gain):
        print("DoubleHump is training!")
        self.v = np.zeros(6, dtype=float)
        for epoch in range(epochs):
            loss_avg = 0.0
            for sample in np.random.permutation(len(indata)):
                loss_avg += self.correct(np.atleast_1d(indata[sample]),
                                         np.atleast_1d(outdata[sample]),
                                         step, gain)
            loss_avg /= len(indata)
            if epoch % 10 == 0:
                print("Epoch: {0} | Loss: {1}".format(epoch, loss_avg))
        print("DoubleHump done training!")

    def _f(self, parameters, inputs):
        hump0 = parameters[0]*np.exp(-parameters[1]**2*(inputs[0] - parameters[2])**2)
        hump1 = parameters[3]*np.exp(-parameters[4]**2*(inputs[0] - parameters[5])**2)
        return hump0 + hump1

    def _l(self, parameters, inputs, outputs):
        errors = outputs - self._f(parameters, inputs)
        return np.dot(errors, errors)

##################################################

X = np.arange(-10.0, 20.0, 0.05)
Y = np.exp(-X**2/10) + np.exp(-(X-10)**2/4)
# Y = (np.exp(np.sin(np.sqrt(np.abs(X)))*X) + 10.0*np.exp(-(X+5)**2))/100

model = VanillaNeuralNetwork([1, 10, 1], stdev0=1.0)
# model = DoubleHump()
model.train(X, Y, epochs=50, step=0.01, gain=0.9)
Y_approx = np.array([model.predict([x]) for x in X])

pyplot.plot(X, Y, label="target")
pyplot.plot(X, Y_approx, label="approx")
pyplot.legend()
pyplot.show()
