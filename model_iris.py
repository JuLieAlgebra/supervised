#!/usr/bin/env python3
"""
Using a multilayer-perceptron to model the iris dataset.

"""
import numpy as np
from classifier import MLP

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

name = "iris_model"
for d in dimensions[1:-1]:
    name += '_' + str(d)
print(name)

model.train(indata=features[shuffle_train],
            outdata=targets[shuffle_train],
            epochs=400,
            step=0.005,
            gain=0.85)

model.save(name)

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
