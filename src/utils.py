import numpy as np
import math

def cost(y_actual: np.ndarray, y: np.ndarray):
    """Computes the difference between two (n,1) vectors"""

    if type(y_actual) != np.ndarray: 
        raise TypeError("y_actual expected <np.ndarray>"
                        f", got {type(y_actual)}")
    
    if type(y) != np.ndarray: 
        raise TypeError("y expected <np.ndarray>" 
                        f", got {type(y)}")
    
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

    if np.shape(X)[0] != np.shape(Y)[0]: 
        raise ValueError("Shape mismatch: X has "
                        f"{np.shape(X)[0]} rows, "
                        f"but Y has {np.shape(Y)[0]} rows")

    (rows,_) = np.shape(X)
    
    indices = np.random.permutation(rows)
    
    X = X[indices]
    Y = Y[indices]

    X_batches = [X[batch_size * j : batch_size * (j + 1)] for j in range(math.ceil(rows / batch_size))]
    Y_batches = [Y[batch_size * j : batch_size * (j + 1)] for j in range(math.ceil(rows / batch_size))]

    return (X_batches, Y_batches)
    