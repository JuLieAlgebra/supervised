"""
Multilayer-perceptron for deep classification (or optionally regression).
Trained by stochastic gradient descent with momentum.

Uses autograd library for automatic differentiation.

"""
from autograd import numpy as np, value_and_grad
import pickle

##################################################

class MLP:
    def __init__(self, dimensions, stdev0=1.0, classifier=True):
        self.parameters = [(np.random.normal(0.0, stdev0, size=(indim, outdim)),  # weights
                            np.random.normal(0.0, stdev0, size=outdim))           # biases
                           for indim, outdim in zip(dimensions[:-1], dimensions[1:])]
        self.classifier = classifier
        self._dldp = value_and_grad(self._l)

    def save(self, filename):
        """
        Saves model.
        """
        if filename[-4:] != ".mlp":
            filename = filename + ".mlp"
        with open(filename, 'wb') as file:
            pickle.dump(self.parameters, file, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """
        Loads saved model.
        """
        if filename[-4:] != ".mlp":
            filename = filename + ".mlp"
        with open(filename, 'rb') as file:
            self.parameters = pickle.load(file)

    def train(self, features, targets, step, gain, tolerance=1e-3, max_epochs=400):
        """
        features:   A two dimensional np array of the features of the data.
                    Each line is a sample from the data, without the corresponding
                    label.
        targets:    Same as labels
        """
        assert len(features) == len(targets)
        print("==================================================")
        print("MLP: began training")

        self.v_W = [0.0]*len(self.parameters)
        self.v_b = [0.0]*len(self.parameters)
        for epoch in range(max_epochs):
            loss_avg = 0.0

            # randomizing how the data is presented to the neural networkd
            for sample in np.random.permutation(len(features)):
                # np.atleast_1d to handle scalars
                loss_avg += self.correct(np.atleast_1d(features[sample]),
                                         np.atleast_1d(targets[sample]),
                                         step, gain)
            loss_avg /= len(features)
            if epoch % 10 == 0:
                print("Epoch: {0} | Loss: {1}".format(epoch, loss_avg))

        print("MLP: finished training")
        print("==================================================")

    def evaluate(self, features, labels, classes):
        """
        features:   A two dimensional np array of the features of the data.
                    Each line is a sample from the data, without the corresponding
                    label.
        labels:     A two dimensional np array of the classifications of the data.
                    Each line is the class label (an int from 0 to n-1 number of classes)
                    of the corresponding sample of data.

                    features[i] + labels[i] gives a sample of the data with the
                    class it belongs to
        classes:    A list of possible label values.
        """
        assert len(features) == len(labels)
        confusion_matrix = np.zeros((len(classes), len(classes)), dtype=float)
        for feature, label in zip(features, labels):
            prediction = np.argmax(self.predict(feature))
            target = list(classes).index(label)
            confusion_matrix[prediction, target] += 1
        if len(features):
            confusion_matrix *= 100.0 / len(features)
        return confusion_matrix

    def predict(self, inputs):
        """
        Wrapper function for _f
        To calculate the gradient, autograd needs our functions to pass
        in the weights and biases of the neural network

        Inputs: can be vector of many inputs
        """
        return self._f(self.parameters, inputs)

    def correct(self, inputs, outputs, step, gain):
        """
        One step of gradient descent.
        """
        loss, gradients = self._dldp(self.parameters, inputs, outputs)
        for i, ((W, b), (dldW, dldb)) in enumerate(zip(self.parameters, gradients)):
            self.v_W[i] += gain*(dldW - self.v_W[i])
            self.v_b[i] += gain*(dldb - self.v_b[i])
            W -= step*self.v_W[i]
            b -= step*self.v_b[i]
        return loss

    def _f(self, parameters, inputs):
        """
        Feed forward function.
        """
        for W, b in parameters:
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        if self.classifier:
            return 0.5*(inputs + 1.0)
        else:
            return outputs

    def _l(self, parameters, inputs, outputs):
        """
        Loss function.
        """
        error = outputs - self._f(parameters, inputs)
        return np.dot(error, error)

##################################################

if __name__ == "__main__":
    print("MLP: began unit-test")

    from matplotlib import pyplot

    X = np.arange(-10.0, 20.0, 0.05)
    Y = np.exp(-X**2/10) + 2*np.exp(-(X-7)**2/5.0) + np.random.randn(np.shape(X)[0])/10

    model = MLP([1, 10, 1], classifier=False)
    model.train(X, Y, 0.005, 0.9, tolerance=1e-3, max_epochs=200)
    Y_approx = np.array([model.predict([x]) for x in X])

    pyplot.title("MLP Fit of Double-Hump", fontsize=16)
    pyplot.plot(X, Y, label="target")
    pyplot.plot(X, Y_approx, label="approx")
    pyplot.grid(True)
    pyplot.legend()

    print("MLP: finished unit-test")
    pyplot.show()
