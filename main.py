import random
import matplotlib.pyplot as plt
import MNISTLoader
from KNN import KNN
from SVM import SVM
import numpy as np

d = 28 * 28  # 784


def calculate_err(classifier, X_test, Y_test):
    err_count = 0
    sum = 0
    m = Y_test.shape[0]
    for i in range(m):
        x = X_test[i, :]
        y = classifier.classify(x)
        sum += y
        if y != Y_test[i]:
            err_count += 1

    print(sum)
    return err_count / m


def main():
    # mnist.init()
    x_train, y_train, x_test, y_test = MNISTLoader.load_all()
    x_train = x_train / 255
    x_test = x_test / 255

    x01_train, y01_train, x01_test, y01_test = MNISTLoader.load_part([0, 1])

    m = x01_train.shape[0]
    perm = np.random.permutation(m)[0:2000]
    X = x01_train[perm]
    Y = y01_train[perm]
    svm = SVM(1, X, Y)
    m = x01_test.shape[0]
    perm = np.random.permutation(m)[0:100]
    err = calculate_err(svm, x01_test[perm, :], y01_test[perm])
    print(err)



    # run KNN with 5 fold cross validation

    # m = x_train.shape[0]
    # perm = np.random.permutation(m)[0:2000]
    # X = x_train[perm]
    # Y = y_train[perm]
    # m = X.shape[0]
    # p = 5
    # k_args = list(range(1, 11))
    # chunk_size = m / p
    # errs = []
    #
    # for k in k_args:
    #     err = 0
    #     for i in range(p):
    #         S_i_indices = list(range((int)(i*chunk_size), (int)((i+1)*chunk_size)))
    #         VX = X[S_i_indices, :]
    #         VY = Y[S_i_indices]
    #         SX_tag = np.delete(X, S_i_indices, axis=0)
    #         SY_tag = np.delete(Y, S_i_indices)
    #
    #         knn = KNN(k, SX_tag, SY_tag)
    #         err += calculate_err(knn, VX, VY)
    #     err /= p
    #     errs.append(err)
    #
    # best_k = k_args[np.argmin(np.asarray(errs))]
    # knn = KNN(best_k, X, Y)  # (return)
    # print(errs)
    # print(best_k)
    # m = x_test.shape[0]
    # perm = np.random.permutation(m)[0:100]
    # err = calculate_err(knn, x_test[perm, :], y_test[perm])
    # print("final err:")
    # print(err)


def show_images(X, Y):
    imgs = []
    labels = []
    amount = 10
    for i in range(0, amount):
        r = random.randint(1, X.shape[0]) - 1
        img = X[r, :].reshape(28, 28)
        imgs.append(img)
        labels.append('training image [' + str(r) + '] = ' + str(Y[r]))

    cols = 5
    rows = int(len(imgs) / cols)
    plt.figure(figsize=(20, 10))
    index = 1
    for x in zip(imgs, labels):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()


if __name__ == "__main__":
    main()
