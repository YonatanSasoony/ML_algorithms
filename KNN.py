import numpy as np
from scipy.spatial.distance import cdist


class KNN(object):
    def __init__(self, k, X_train, Y_train):
        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train

    def classify(self, x):
        distances = cdist(self.X_train, [x], 'euclidean')
        distances = [item for sublist in distances for item in sublist]

        k_sorted_indices = np.argpartition(distances, self.k)[0:self.k]
        # sorted_indices = [item for sublist in sorted_indices for item in sublist]
        k_labels = self.Y_train[k_sorted_indices]
        frequent_label = np.bincount(k_labels).argmax()
        return frequent_label




