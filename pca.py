"""
Principal-component analysis class.

"""
import numpy as np
import pickle

##################################################

class PCA:
    def __init__(self):
        self.W = None
        self.b = None

    def analyze(self, data):
        print("PCA: began analysis")
        covar = np.cov(data, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(covar)
        self.W = eigvecs[:, ::-1]
        self.b = np.mean(data, axis=0)
        print("PCA: finished analysis")

    def compress(self, data, dimensionality):
        assert self.W is not None
        return (data - self.b).dot(self.W[:, :dimensionality])

    def decompress(self, data):
        assert self.W is not None
        data = np.atleast_2d(data)
        if data.shape[1] < self.W.shape[1]:
            data = np.column_stack((data, np.zeros((len(data), self.W.shape[1]-data.shape[1]), dtype=float)))
        return data.dot(self.W.T) + self.b

    def save(self, filename):
        if filename[-4:] != ".pca":
            filename = filename + ".pca"
        with open(filename, 'wb') as file:
            pickle.dump((self.W, self.b), file, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        if filename[-4:] != ".pca":
            filename = filename + ".pca"
        with open(filename, 'rb') as file:
            self.W, self.b = pickle.load(file)

##################################################

if __name__ == "__main__":
    print("PCA: began unit-test")

    from matplotlib import pyplot
    import seaborn
    seaborn.set()

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

    pca = PCA()
    pca.analyze(features)
    features_compressed = pca.compress(features, 2)

    should_be_features = pca.decompress(pca.compress(features, 4))
    print("PCA: methods compress and decompress are inverses? {0}".format(np.allclose(features, should_be_features)))

    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('KDE of the Iris Data-Set', fontsize=16)

    ax.set_xlim([-4.0, 4.0])
    ax.set_xlabel("PCA Component 0", fontsize=12)
    ax.set_ylim([-1.5, 1.5])
    ax.set_ylabel("PCA Component 1", fontsize=12)

    seaborn.kdeplot(features_compressed[:50, 0], features_compressed[:50, 1], cmap="Greens", shade=True, shade_lowest=False)
    seaborn.kdeplot(features_compressed[50:100, 0], features_compressed[50:100, 1], cmap="Reds", shade=True, shade_lowest=False)
    seaborn.kdeplot(features_compressed[100:, 0], features_compressed[100:, 1], cmap="Blues", shade=True, shade_lowest=False)
    ax.text(1.2, 1.2, "setosa", size=16, color=seaborn.color_palette("Greens")[-2])
    ax.text(-0.5, 0.7, "versicolor", size=16, color=seaborn.color_palette("Reds")[-2])
    ax.text(-3.5, 0.9, "virginica", size=16, color=seaborn.color_palette("Blues")[-2])

    print("PCA: finished unit-test")
    pyplot.show()
