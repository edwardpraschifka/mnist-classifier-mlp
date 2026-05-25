import numpy as np
import math

def cost(y_actual: np.ndarray, y: np.ndarray):
    if type(y_actual) != np.ndarray: 
        raise TypeError("y_actual expected <np.ndarray>" 
                        f", got {type(y_actual)}")
    
    if type(y) != np.ndarray: 
        raise TypeError("y expected <np.ndarray>" 
                        f", got {type(y)}")
    
    if len(y_actual) != len(y):
        raise ValueError("Size mismatch: y_actual has size "
                         f"{len(y_actual)}, while "
                         f"y has size {len(y)}")
    
    
        
    diff_sq = (y_actual - y)**2
    return np.sum(diff_sq)

def sigmoid(z: int):
    """Computes sigmoid smoothing function on z"""

    return 1/(1+np.exp(-z))

def shuffle_and_batch(X: np.ndarray, batch_size: int):
    """Splits data into batches, returns list of batches"""

    np.random.shuffle(X)
    (rows,cols) = np.shape(X)

    batches = [X[batch_size * j : batch_size * (j + 1)] for j in range(math.ceil(rows/ batch_size))]
    return batches