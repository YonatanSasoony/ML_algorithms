import numpy as np


class NaiveBayes(object):
    def __init__(self, X_train, Y_train):
        self.allpos, self.ppos, self.pneg = self.bayes_learn(X_train, Y_train)
        self.d = X_train.shape[1]

    def classify(self, x):  # x: {0,1}^d
        pos_sum, neg_sum, is_nan = 0, 0, 0
        if np.any(np.isnan(self.ppos)) or np.any(np.isnan(self.pneg)):
            return 0
        for i in range(self.d):
            if x[i] == 1:
                pos_sum += np.log(self.ppos[i] / self.pneg[i])
            else:
                neg_sum += np.log((1-self.pneg[i]) / (1-self.ppos[i]))

        y = np.sign(np.log(self.allpos / (1-self.allpos)) + pos_sum - neg_sum)  # -1/1 / nan
        if np.isnan(y):
            return 0
        guess = (y + 1) / 2  # 0/1
        return guess

    @staticmethod
    def bayes_learn(X, Y):  # x: {0,1}^d, y: {0,1}
        m, d = X.shape

        pos_indices = np.where(Y > 0)[0]
        neg_indices = np.delete(np.asarray(list(range(m))), pos_indices)

        pos_count = pos_indices.shape[0]
        neg_count = m - pos_count

        allpos = pos_count / m
        X_pos = X[pos_indices, :]
        X_neg = X[neg_indices, :]
        ppos = np.sum(X_pos, axis=0) / pos_count
        pneg = np.sum(X_neg, axis=0) / neg_count

        return allpos, ppos, pneg


