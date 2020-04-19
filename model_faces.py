#!/usr/bin/env python3
"""
Using PCA and an MLP to model the faces dataset.

"""
import numpy as np
from matplotlib import pyplot

from mlp import MLP
from pca import PCA

np.set_printoptions(suppress=True)

##################################################

directory = "/home/bacon/code/datasets/faces"

def load_image_vector(subject, instance):
    return pyplot.imread(directory+"/images/subject{0}_img{1}.pgm".format(subject, instance)).flatten() / 255.0

def show_image_vector(vector, ax=None):
    if ax is None:
        ax = pyplot
    ax.imshow(vector.reshape(50, 50), interpolation="bicubic", cmap="Greys")

##################################################

samples_train = []
samples_test = []

labels_train = []
labels_test = []

with open(directory+"/genders.csv", 'r') as genders:
    for i, l in enumerate(genders):
        gender = l.strip().split(',')[1]

        samples_train.append(load_image_vector(i+1, 1))
        labels_train.append(gender)

        samples_test.append(load_image_vector(i+1, 2))
        labels_test.append(gender)

        samples_test.append(load_image_vector(i+1, 3))
        labels_test.append(gender)

samples_train = np.array(samples_train)
samples_test = np.array(samples_test)

##################################################

classes = np.unique(labels_train)

embedding = {}
for i, c in enumerate(classes):
    onehot = np.zeros(len(classes), dtype=float)
    onehot[i] = 1.0
    embedding[c] = onehot

targets_train = np.array([embedding[label] for label in labels_train])
targets_test = np.array([embedding[label] for label in labels_test])

##################################################

pca = PCA()
eigs = pca.analyze(samples_train)
pca.save("faces_results/faces")
pca.load("faces_results/faces")


# QUICK TEST
fig = pyplot.figure()
show_image_vector(-pca.W[:, 0], fig.add_subplot(1, 3, 1))
show_image_vector(-pca.W[:, 1], fig.add_subplot(1, 3, 2))
show_image_vector(-pca.W[:, 2], fig.add_subplot(1, 3, 3))
fig = pyplot.figure()
pyplot.plot(np.abs(eigs[:150])/np.max(np.abs(eigs[:150])))
pyplot.show()
quit()
