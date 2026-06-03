import numpy as np
import torch
import torch.nn as nn

from src.cost_functions import QuadraticCost, CrossEntropyCost
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


def backprop_torch(X: np.ndarray, Y: np.ndarray, nw: Network, cost_fn):
        """Conducts backprop using pytorch and the
        quadratic cost function"""
        
        X = torch.tensor(X, dtype=torch.float64)
        Y = torch.tensor(Y, dtype=torch.float64)
        
        Ws = [torch.tensor(w, requires_grad=True, dtype=torch.float64) for w in nw.weights]
        Bs = [torch.tensor(b, requires_grad=True, dtype=torch.float64) for b in nw.biases]
        
        # forward pass
        A = X
        for W, B in zip(Ws, Bs):
            Z = W @ A + B
            A = torch.sigmoid(Z)

        if cost_fn == QuadraticCost:
            cost = torch.sum((Y - A) ** 2)
        
        if cost_fn == CrossEntropyCost:
            cost = -torch.sum(Y * torch.log(A) + (1 - Y) * torch.log(1 - A))

        cost.backward()
        
        return [W.grad.detach().numpy() for W in Ws], [B.grad.detach().numpy() for B in Bs]