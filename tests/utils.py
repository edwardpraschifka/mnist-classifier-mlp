import numpy as np
import torch
import torch.nn as nn


from src.network import Network

def feedforward_torch(X: np.ndarray, nw: Network):
    """Conducts feedforward using pytorch (used for testing)"""
    
    X = torch.tensor(X, dtype=torch.float64)
    
    Ws = [torch.tensor(w, dtype=torch.float64) for w in nw.weights]
    Bs = [torch.tensor(b, dtype=torch.float64) for b in nw.biases]

    A = X

    for W, B in zip(Ws, Bs):
        Z = W @ A + B
        A = torch.sigmoid(Z)
    
    return A.detach().numpy()