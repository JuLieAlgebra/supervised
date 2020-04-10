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

##################################################

classes = np.unique(labels)

embedding = {}
for i, c in enumerate(classes):
    vector = np.zeros(len(classes), dtype=float)
    vector[i] = 1.0
    embedding[c] = vector

targets = np.array([embedding[label] for label in labels])

##################################################

model = MLP([4, 20, 3])

# model.load("iris_trial")

model.train(indata=features,
            outdata=targets,
            epochs=200,
            step=0.01,
            gain=0.9)

model.save("iris_trial")

##################################################

confusion_matrix = model.evaluate(features, labels, classes)

print("==================================================")
print("Confusion Matrix")
print(confusion_matrix)
print("Accuracy: {0}".format(np.sum(np.diag(confusion_matrix))))
print("==================================================")
