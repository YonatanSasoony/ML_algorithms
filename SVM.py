import numpy as np
from cvxopt import solvers
from cvxopt import matrix as cvxopt_matrix


class SVM(object):
    def __init__(self, regulator, X_train, Y_train):
        self.w = self.calc_separator(regulator, X_train, Y_train)

    def classify(self, x):
        y = np.sign(np.dot(self.w, x))[0]
        return y

    @staticmethod
    def calc_separator(regulator, X, Y):  # y: {-1,1}
        m, d = X.shape
        u1 = np.zeros((d, 1))
        u2 = np.ones((m, 1)) * (1.0 / m)
        u = np.concatenate((u1, u2), axis=0)

        H1 = np.eye(d)
        H2 = np.zeros((d, m))
        H12 = np.concatenate((H1, H2), axis=1)
        H3 = np.zeros((m, d))
        H4 = np.zeros((m, m))
        H34 = np.concatenate((H3, H4), axis=1)
        H = np.concatenate((H12, H34), axis=0) * 2 * regulator

        A1 = np.zeros((m, d))
        A2 = np.eye(m)
        A12 = np.concatenate((A1, A2), axis=1)
        Y = Y.reshape(Y.shape[0], 1)
        A3 = X * Y
        A4 = np.eye(m)
        A34 = np.concatenate((A3, A4), axis=1)
        A = np.concatenate((A12, A34), axis=0)

        v1 = np.zeros((m, 1))
        v2 = np.ones((m, 1))
        v = np.concatenate((v1, v2), axis=0)

        H = cvxopt_matrix(H)
        u = cvxopt_matrix(u)
        A = cvxopt_matrix(A)
        v = cvxopt_matrix(v)

        z = solvers.qp(H, u, G=-A, h=-v)['x']
        return np.asarray(z[0:d].T)
