import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


class Kmeans(object):
    def __init__(self, k, X_train, Y_train):
        self.k = k
        self.d = X_train.shape[1]
        self.X_train = X_train
        self.Y_train = Y_train
        self.clusters = []
        self.centers = []

    def cluster(self):
        centers = initRandomCenters(self)
        clusters = calcClusters(self.X_train, centers)
        converged = False
        while not converged:
            prevCenters = centers
            centers = calcCenters(self, self.X_train, clusters)
            clusters = calcClusters(self.X_train, centers)
            converged = isConverged(prevCenters, centers)
        print("Kmeans converged")
        self.clusters = clusters
        self.centers = centers
        return clusters

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

        show_images(imgs, labels)


def initRandomCenters(self):
    centers = np.empty((0, self.d), dtype=float)
    for i in range(self.k):
        centers = np.append(centers, np.random.rand(1, self.d), axis=0)
    return centers


def calcClusters(X, centers):
    distances = cdist(X, centers, 'euclidean')  # matrix n,k - each Mi,j is the distance between X[i] and centers[j]
    return np.array([np.argmin(row) for row in distances])


def calcCenters(self, X, clusters):
    centers = []
    for i in range(self.k):
        # Updating Centroids by taking mean of Cluster it belongs to
        cluster = X[clusters == i]
        if np.size(cluster) == 0:
            return initRandomCenters(self)
        center = np.mean(cluster, axis=0)
        centers.append(center)
    return np.vstack(centers)


def isConverged(prevCenters, centers):
    return np.array_equal(prevCenters, centers)


def show_images(images, main_title):
    cols = 5
    rows = int(len(images) / cols)
    plt.figure(figsize=(20, 10))
    index = 1
    for image in images:
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        index += 1
    print(main_title)
    plt.show()
