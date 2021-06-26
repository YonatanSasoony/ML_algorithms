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
    # print(err)
    return err


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


def pick_random(X, Y, n):
    perm = np.random.permutation(X.shape[0])[0:n]
    return X[perm], Y[perm]


def main():
    # mnist.init()
    x_train, y_train, x_test, y_test = MNISTLoader.load_all()
    x_train = x_train / 255
    x_test = x_test / 255

    x_train, y_train = pick_random(x_train, y_train, 2000)
    x_test, y_test = pick_random(x_test, y_test, 500)

    x_good_train, y_good_train, x_good_test, y_good_test = MNISTLoader.load_part([0, 1])
    x_good_train = x_good_train / 255
    x_good_test = x_good_test / 255

    x_good_train, y_good_train = pick_random(x_good_train, y_good_train, 2000)
    x_good_test, y_good_test = pick_random(x_good_test, y_good_test, 500)

    x_bad_train, y_bad_train, x_bad_test, y_bad_test = MNISTLoader.load_part([3, 5])
    x_bad_train = x_bad_train / 255
    x_bad_test = x_bad_test / 255

    y_bad_train = (y_bad_train - 3) / 2  # 0/1
    y_bad_test = (y_bad_test - 3) / 2  # 0/1

    x_bad_train, y_bad_train = pick_random(x_bad_train, y_bad_train, 2000)
    x_bad_test, y_bad_test = pick_random(x_bad_test, y_bad_test, 500)

    ## Naive Bayes
    ## need to transform x to {0,1}^d
    print("\nNaive Bayes")
    ## easy pair- 0 1
    err = run(NaiveBayes, np.ceil(x_good_train), y_good_train, np.ceil(x_good_test), y_good_test)
    print("err of 0,1: " + str(err * 100) + "%")
    ## hard pair - 3 5
    err = run(NaiveBayes, np.ceil(x_bad_train), y_bad_train, np.ceil(x_bad_test), y_bad_test)
    print("err of 3,5: " + str(err * 100) + "%")

    ## SVM
    print("\nSVM")
    ## need to transform y to {-1,1}
    ## easy pair- 0 1
    err = run(SVM, x_good_train, np.sign(y_good_train - 0.5), x_good_test, np.sign(y_good_test - 0.5), arg=1)
    print("err of 0,1: " + str(err * 100) + "%")
    ## hard pair - 3 5
    err = run(SVM, x_bad_train, np.sign(y_bad_train - 0.5), x_bad_test, np.sign(y_bad_test - 0.5), arg=1)
    print("err of 3,5: " + str(err * 100) + "%")

    ## KNN
    print("\nKNN")
    err = run(KNN, x_train, y_train, x_test, y_test, arg=5)
    print("err of 0-9: " + str(err * 100) + "%")

    ## Kmeans
    print("\nKmeans")
    kmeans = Kmeans(10, x_train, y_train)
    kmeans.cluster()
    kmeans.showClusters()


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
