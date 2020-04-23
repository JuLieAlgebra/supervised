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

dimensions = [40, 100, 50, 2]

name = "faces_results/faces_model"
for d in dimensions[:-1]:
    name += '_' + str(d)
print(name)

##################################################

pca = PCA()
new_pca = True

if new_pca:
    eigs = pca.analyze(samples_train)
    pca.save("faces_results/faces")
else:
    pca.load("faces_results/faces")

samples_train_compressed = pca.compress(samples_train, dimensionality=dimensions[0])
samples_test_compressed = pca.compress(samples_test, dimensionality=dimensions[0])

##################################################

mlp = MLP(dimensions)
new_mlp = True

if new_mlp:
    mlp.train(samples_train_compressed,
              targets_train,
              max_epochs=200,
              step=0.1,
              gain=0.9)
    mlp.save(name)
else:
    mlp.load(name)

##################################################

cm_train = mlp.evaluate(samples_train_compressed,
                        labels_train,
                        classes)

print("==================================================")
print("Training Confusion Matrix")
print(cm_train)
print("Accuracy: {0}".format(np.sum(np.diag(cm_train))))
print("==================================================")

cm_valid = mlp.evaluate(samples_test_compressed,
                        labels_test,
                        classes)

print("==================================================")
print("Validation Confusion Matrix")
print(cm_valid)
print("Accuracy: {0}".format(np.sum(np.diag(cm_valid))))
print("==================================================")
