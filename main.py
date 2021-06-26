import random
import matplotlib.pyplot as plt
import MNISTLoader
from KNN import KNN
from SVM import SVM
from NaiveBayes import NaiveBayes
from Kmeans import Kmeans
import numpy as np

d = 28 * 28  # 784


def run(initializer, x_train, y_train, x_test, y_test, arg=None, cross_validation=None):
    if cross_validation is not None:
        classifier = cross_validation(cross_validation[0], x_train, y_train, cross_validation[1], initializer)
    elif arg is not None:
        classifier = initializer(arg, x_train, y_train)
    else:
        classifier = initializer(x_train, y_train)
    err = calculate_err(classifier, x_test, y_test)
    print(err)
    return err


def calculate_err(classifier, X_test, Y_test):
    err_count = 0
    # sum = 0
    m = Y_test.shape[0]
    for i in range(m):
        x = X_test[i, :]
        y = classifier.classify(x)
        # sum += y
        if y != Y_test[i]:
            err_count += 1

    # print(sum)
    return err_count / m


def main():
    # mnist.init()
    x_train, y_train, x_test, y_test = MNISTLoader.load_all()
    x_train = x_train / 255
    x_test = x_test / 255

    perm = np.random.permutation(x_train.shape[0])[0:2000]
    x_train = x_train[perm]
    y_train = y_train[perm]

    perm = np.random.permutation(x_test.shape[0])[0:500]
    x_test = x_test[perm]
    y_test = y_test[perm]

    x_good_train, y_good_train, x_good_test, y_good_test = MNISTLoader.load_part([0, 1])
    x_good_train = x_good_train / 255
    x_good_test = x_good_test / 255
    perm = np.random.permutation(x_good_train.shape[0])[0:2000]
    x_good_train = x_good_train[perm]
    y_good_train = x_good_train[perm]

    perm = np.random.permutation(x_good_test.shape[0])[0:500]
    x_good_test = x_good_train[perm]
    y_good_test = x_good_train[perm]

    x_bad_train, y_bad_train, x_bad_test, y_bad_test = MNISTLoader.load_part([3, 5])
    x_bad_train = x_bad_train / 255
    x_bad_test = x_bad_test / 255

    y_bad_train = (y_bad_train - 3) / 2  # 0/1
    y_bad_test = (y_bad_test - 3) / 2  # 0/1

    perm = np.random.permutation(x_bad_train.shape[0])[0:2000]
    x_bad_train = x_bad_train[perm]
    y_bad_train = y_bad_train[perm]

    perm = np.random.permutation(x_bad_test.shape[0])[0:500]
    x_bad_test = x_bad_test[perm]
    y_bad_test = y_bad_test[perm]

    # run(initializer, x_train, y_train, x_test, y_test, arg=None, cross_validation=None):

    ## Naive Bayes
    ## need to transform x to {0,1}^d
    ## easy pair- 0 1
    run(NaiveBayes, np.ceil(x_good_train), y_good_train, np.ceil(x_good_test), y_good_test)

    ## hard pair - 3 5
    run(NaiveBayes, np.ceil(x_bad_train), y_bad_train, np.ceil(x_bad_test), y_bad_test)

    ## easy pair- 0 1
    # x_good_ceil_train = np.ceil(x_good_train)
    # perm = np.random.permutation(x_good_ceil_train.shape[0])[0:2000]
    # X = x_good_ceil_train[perm]
    # Y = y_good_train[perm]
    # bayes = NaiveBayes(X, Y)
    # x_good_ceil_test = np.ceil(x_good_test)
    # perm = np.random.permutation(x_good_ceil_test.shape[0])[0:500]
    # err = calculate_err(bayes, x_good_ceil_test[perm, :], y_good_test[perm])
    # print(err)

    ## hard pair - 3 5
    # x_bad_ceil_train = np.ceil(x_bad_train)
    # perm = np.random.permutation(x_bad_ceil_train.shape[0])[0:2000]
    # X = x_bad_ceil_train[perm]
    # Y = y_bad_train[perm]
    # bayes = NaiveBayes(X, Y)
    # x_bad_ceil_test = np.ceil(x_bad_test)
    #
    # perm = np.random.permutation(x_bad_ceil_test.shape[0])[0:500]
    # err = calculate_err(bayes, x_bad_ceil_test[perm, :], y_bad_test[perm])
    # print(err)

    ## SVM
    ## need to transform y to {-1,1}
    ## easy pair- 0 1
    run(NaiveBayes, x_good_train, np.sign(y_good_train - 0.5), x_good_test, np.sign(y_good_test - 0.5))

    ## hard pair - 3 5
    run(NaiveBayes, x_bad_train, np.sign(y_bad_train - 0.5), x_bad_test, np.sign(y_bad_test - 0.5))

    ## easy pair- 0 1
    # perm = np.random.permutation(x_good_train.shape[0])[0:2000]
    # X = x_good_train[perm]
    # Y = y_good_train[perm]
    # Y = np.sign(Y - 0.5)
    # # best_svm = cross_validation(5, X, Y, [1, 10, 100], SVM)
    # svm = SVM(1, X, Y)
    #
    # perm = np.random.permutation(x_good_test.shape[0])[0:500]
    # err = calculate_err(svm, x_good_test[perm, :], np.sign(y_good_test[perm]-0.5))
    # print(err)

    ## hard pair - 3 5
    # perm = np.random.permutation(x_bad_train.shape[0])[0:2000]
    # X = x_bad_train[perm]
    # Y = y_bad_train[perm]
    #
    # Y = np.sign(Y - 0.5)
    # # best_svm = cross_validation(5, X, Y, [1, 10, 100], SVM)
    # svm = SVM(1, X, Y)
    # perm = np.random.permutation(x_bad_test.shape[0])[0:500]
    # err = calculate_err(svm, x_bad_test[perm, :], np.sign(y_bad_test[perm]-0.5))
    # print(err)

    ## KNN
    run(NaiveBayes, x_train, y_train, x_test, y_test)
    # perm = np.random.permutation(x_train.shape[0])[0:2000]
    # X = x_train[perm]
    # Y = y_train[perm]
    # best_knn = cross_validation(5, X, Y, list(range(1, 11)), KNN)  # run KNN with 5 fold cross validation
    # perm = np.random.permutation(x_test.shape[0])[0:500]
    # err = calculate_err(best_knn, x_test[perm, :], y_test[perm])
    # print("final err:")
    # print(err)

    ## Kmeans
    kmeans = Kmeans(10, x_train, y_train)
    kmeans.cluster()
    kmeans.showClusters()


def cross_validation(p, X, Y, args, initializer):
    m = X.shape[0]
    chunk_size = m // p
    errs = []

    for arg in args:
        err = 0
        for i in range(p):
            S_i_indices = list(range((i * chunk_size), ((i + 1) * chunk_size)))
            VX = X[S_i_indices, :]
            VY = Y[S_i_indices]
            SX_tag = np.delete(X, S_i_indices, axis=0)
            SY_tag = np.delete(Y, S_i_indices)

            classifier = initializer(arg, SX_tag, SY_tag)
            err += calculate_err(classifier, VX, VY)
        err /= p
        errs.append(err)

    best_arg = args[np.argmin(np.asarray(errs))]
    best_classifier = initializer(best_arg, X, Y)
    print(errs)
    print(best_arg)
    return best_classifier


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
