#!/usr/bin/env python3
"""
Using a multilayer-perceptron to model the iris dataset.

"""
import numpy as np
from matplotlib import pyplot

from mlp import MLP
from pca import PCA

np.set_printoptions(suppress=True)

##################################################

filename = "/home/bacon/code/datasets/IRIS.csv"

features = np.genfromtxt(fname=filename,
                         dtype=float,
                         comments='#',
                         delimiter=',',
                         skip_header=1,
                         usecols=(0, 1, 2, 3))

labels = np.genfromtxt(fname=filename,
                       dtype=str,
                       comments='#',
                       delimiter=',',
                       skip_header=1,
                       usecols=(4))

ndata = len(features)
assert ndata == len(labels)

##################################################

classes = np.unique(labels)

embedding = {}
for i, c in enumerate(classes):
    vector = np.zeros(len(classes), dtype=float)
    vector[i] = 1.0
    embedding[c] = vector

targets = np.array([embedding[label] for label in labels])

##################################################

train_frac = 0.5
ntrain = int(train_frac*ndata)

shuffle = np.random.permutation(ndata)
shuffle_train = shuffle[:ntrain]
shuffle_valid = shuffle[ntrain:]

##################################################

dimensions = [4, 10, 3]
model = MLP(dimensions)

name = "iris_results/iris_model"
for d in dimensions[1:-1]:
    name += '_' + str(d)
print(name)

do_training = True
if do_training:
    model.train(features=features[shuffle_train],
                targets=targets[shuffle_train],
                max_epochs=800,
                step=0.001,
                gain=0.9)
    model.save(name)
else:
    model.load(name)

##################################################

cm_train = model.evaluate(features[shuffle_train],
                          labels[shuffle_train],
                          classes)

print("==================================================")
print("Training Confusion Matrix")
print(cm_train)
print("Accuracy: {0}".format(np.sum(np.diag(cm_train))))
print("==================================================")

cm_valid = model.evaluate(features[shuffle_valid],
                          labels[shuffle_valid],
                          classes)

print("==================================================")
print("Validation Confusion Matrix")
print(cm_valid)
print("Accuracy: {0}".format(np.sum(np.diag(cm_valid))))
print("==================================================")

##################################################

print("Plotting PCA projection of data-set and classifier.")

pca = PCA()
pca.analyze(features)
pca.save("iris_results/iris")
features_compressed = pca.compress(features, 2)

fig = pyplot.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('MLP-Classification of the Iris Data-Set', fontsize=16)

ax.set_xlim([-4.0, 4.0])
ax.set_xlabel("PCA Component 0", fontsize=12)
ax.set_ylim([-1.5, 1.5])
ax.set_ylabel("PCA Component 1", fontsize=12)

XX, YY = np.meshgrid(np.arange(*ax.get_xlim(), 0.005),
                     np.arange(*ax.get_ylim(), 0.005))
XY = np.vstack((XX.ravel(), YY.ravel())).T
ZZ = np.argmax(model.predict(pca.decompress(XY)), axis=1).reshape(XX.shape)
ax.contourf(XX, YY, ZZ+1e-6, levels=3, colors=['g', 'b', 'r'], alpha=0.2)

ax.scatter(features_compressed[:50, 0], features_compressed[:50, 1], c='g', edgecolors='k', label="setosa")
ax.scatter(features_compressed[50:100, 0], features_compressed[50:100, 1], c='r', edgecolors='k', label="versicolor")
ax.scatter(features_compressed[100:, 0], features_compressed[100:, 1], c='b', edgecolors='k', label="virginica")
ax.legend()

print("Close plots to finish...")
pyplot.show()
