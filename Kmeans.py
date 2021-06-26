import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


class Kmeans(object):
    def __init__(self, k, d, X_train, Y_train):
        self.k = k
        self.d = d
        self.X_train = X_train
        self.Y_train = Y_train
        self.clusters = []
        self.centers = []
        self.clusterToLabel = {}

    def cluster(self, init='random'):
        centers = initCenters(self, init)
        clusters = calcClusters(self.X_train, centers)
        converged = False
        while not converged:
            prevCenters = centers
            centers = calcCenters(self, self.X_train, clusters, init)
            clusters = calcClusters(self.X_train, centers)
            converged = isConverged(prevCenters, centers)

        self.clusters = clusters
        self.centers = centers
        updateClusterToLabel(self)
        return clusters

    def predict(self, X, Y):
        distances = cdist(X, self.centers, 'euclidean')  # matrix n,k - each Mi,j is the distance between X[i] and centers[j]
        predictions = np.array([self.clusterToLabel[np.argmin(row)] for row in distances])
        totalCorrects = np.sum(predictions == Y)
        return totalCorrects / np.size(Y)

    def predictSingleLabel(self, x):
        distances = [np.linalg.norm(np.subtract(x, center)) for center in self.centers]
        return self.clusterToLabel[np.argmin(distances)]

    def showClusters(self):
        for i in range(self.k):
            self.showSingleCluster(i)

    def showSingleCluster(self, index):
        cluster = self.X_train[self.clusters == index]
        cluster_labels = self.Y_train[self.clusters == index]
        imgs = []
        labels = []
        amount = 10
        for i in range(0, amount):
            r = random.randint(1, cluster.shape[0]) - 1
            img = cluster[r].reshape(28, 28)
            imgs.append(img)
            labels.append('training image [' + str(r) + '] = ' + str(cluster_labels[r]))

        print("labels histo")
        for i in range(self.k):
            print(str(i) + ":" + str(np.size(cluster_labels[cluster_labels == i])))

        show_images(imgs, labels, calcGuessLabel(self, cluster_labels))


def initCenters(self, init):
    if init == 'random':
        return initRandomCenters(self)
    if init == 'known':
        return initKnownCenters(self)
    raise Exception("invalid init type")


def initRandomCenters(self):
    centers = np.empty((0, self.d), dtype=float)
    for i in range(self.k):
        centers = np.append(centers, np.random.rand(1, self.d), axis=0)
    return centers


def initKnownCenters(self):
    idx = [55991, 3427, 16837, 30077, 896, 28315, 28400, 562, 4803, 52570]
    return self.X_train[idx, :]


def calcClusters(X, centers):
    distances = cdist(X, centers, 'euclidean')  # matrix n,k - each Mi,j is the distance between X[i] and centers[j]
    return np.array([np.argmin(row) for row in distances])


def calcCenters(self, X, clusters, init):
    centers = []
    for i in range(self.k):
        # Updating Centroids by taking mean of Cluster it belongs to
        cluster = X[clusters == i]
        if np.size(cluster) == 0:
            return initCenters(self, init)
        center = np.mean(cluster, axis=0)
        centers.append(center)
    return np.vstack(centers)


def isConverged(prevCenters, centers):
    return np.array_equal(prevCenters, centers)


def calcGuessLabel(self, Y):
    histo = np.zeros(self.k)
    for i in range(len(Y)):
        histo[Y[i]] += 1
    y = np.argmax(histo)
    if np.size(y) > 1:
        return y[0]
    return y


def updateClusterToLabel(self):
    for i in range(self.k):
        cluster_labels = self.Y_train[self.clusters == i]
        self.clusterToLabel[i] = calcGuessLabel(self, cluster_labels)


def show_images(images, title_texts, main_title):
    cols = 5
    rows = int(len(images) / cols)
    plt.figure(figsize=(20, 10))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    print(main_title)
    plt.show()