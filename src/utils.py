import numpy as np
import math

from tensorflow.keras.datasets import mnist


def sigmoid(z: int):
    """Computes sigmoid smoothing function on z"""

    return 1/(1+np.exp(-z))


def shuffle_and_batch(X: np.ndarray, Y: np.ndarray, batch_size: int):
    """Splits data into batches, returns list of batches"""

    if np.shape(X)[1] != np.shape(Y)[1]: 
        raise ValueError("Shape mismatch: X has "
                        f"{np.shape(X)[1]} columns, "
                        f"but Y has {np.shape(Y)[1]} columns")

    (_,cols) = np.shape(X)
    
    indices = np.random.permutation(cols)
    
    X = X[:, indices]
    Y = Y[:, indices]


    X_batches = [X[:, batch_size * j : batch_size * (j + 1)] for j in range(math.ceil(cols / batch_size))]
    Y_batches = [Y[:, batch_size * j : batch_size * (j + 1)] for j in range(math.ceil(cols / batch_size))]

    return (X_batches, Y_batches)


def accuracy(y_actual: np.ndarray, y: np.ndarray):
    """Computes percentage accuracy between
    predicted one-hot labels and actual 
    one-hot labels"""

    # both sets should have the same number
    # of categories
    if np.shape(y_actual)[0] != np.shape(y)[0]: 
        raise ValueError("Shape mismatch: y has "
                        f"{np.shape(y)[0]} rows, "
                        f"but y_actual has {np.shape(y_actual)[0]} rows")
    
    # each training example should have the same
    # number of training examples
    if np.shape(y_actual)[1] != np.shape(y)[1]: 
        raise ValueError("Shape mismatch: y has "
                        f"{np.shape(y)[1]} rows, "
                        f"but y_actual has {np.shape(y_actual)[1]} columns")
    
    accuracy = (np.argmax(y_actual, axis=0) == np.argmax(y, axis=0)).sum() / np.shape(y_actual)[1]
    return accuracy 


def get_mnist_data():
    
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train = X_train.reshape(-1, 784) / 255.0
        X_test = X_test.reshape(-1, 784) / 255.0

        def one_hot(Y, num_classes=10):
            n = len(Y)
            one_hot = np.zeros((n, 10))
            one_hot[np.arange(n), Y] = 1
            return one_hot

        Y_train = one_hot(Y_train)
        Y_test = one_hot(Y_test)

        return (X_train, Y_train, X_test, Y_test)