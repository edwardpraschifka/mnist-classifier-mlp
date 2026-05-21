from math import exp
import numpy as np

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
    