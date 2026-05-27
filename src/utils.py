import numpy as np
import math

def cost(y_actual: np.ndarray, y: np.ndarray):
    """Computes the difference between two (n,1) vectors"""
    
    if np.shape(y_actual) != (len(y_actual),1): 
        raise ValueError("y_actual: expected shape "
                         f"({len(y_actual)}, 1), "
                         f"got {np.shape(y_actual)}")
    
    if np.shape(y) != (len(y),1): 
        raise ValueError(f"y: expected shape ({len(y)},1), " 
                         "got {np.shape(y)}")
    
    if np.shape(y_actual) != np.shape(y):
        raise ValueError("Shape mismatch: y_actual has "
                         f"shape {np.shape(y_actual)}, "
                         f"but y has shape {np.shape(y)}")
    
        
    diff_sq = (y_actual - y)**2
    return np.sum(diff_sq)


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
    # of training examples
    if np.shape(y_actual)[0] != np.shape(y)[0]: 
        raise ValueError("Shape mismatch: y has "
                        f"{np.shape(y)[0]} rows, "
                        f"but y_actual has {np.shape(y_actual)[0]} rows")
    
    # each training example should have the same
    # number of columns
    if np.shape(y_actual)[1] != np.shape(y)[1]: 
        raise ValueError("Shape mismatch: y has "
                        f"{np.shape(y)[1]} rows, "
                        f"but y_actual has {np.shape(y_actual)[1]} rows")
    
    accuracy = (np.argmax(y_actual, axis=1) == np.argmax(y, axis=1)).sum() / len(y)
    return accuracy 