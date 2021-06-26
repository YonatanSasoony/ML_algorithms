import numpy as np
from urllib import request
import gzip
import pickle

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load_all():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def load_part(digits):
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    training_images, training_labels, test_images, test_labels = mnist["training_images"], mnist["training_labels"], \
                                                                 mnist["test_images"], mnist["test_labels"]

    training_indices = []
    test_indices = []
    for digit in digits:
        training_indices.append(np.where(training_labels == digit)[0].tolist())
        test_indices.append(np.where(test_labels == digit)[0].tolist())

    training_indices = [item for sublist in training_indices for item in sublist]
    test_indices = [item for sublist in test_indices for item in sublist]
    return training_images[training_indices], training_labels[training_indices], \
           test_images[test_indices], test_labels[test_indices]


if __name__ == '__main__':
    init()
