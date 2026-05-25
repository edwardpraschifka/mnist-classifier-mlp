import numpy as np
import torch

from src.network import Network

def backprop_torch(X: np.ndarray, Y: np.ndarray, nw: Network):
    """Conducts backprop using pytorch (used for testing)"""

    X = torch.tensor(X, dtype=torch.float64)
    Y = torch.tensor(Y, dtype=torch.float64)
    W1 = torch.tensor(nw.weights[0], requires_grad=True, dtype=torch.float64)
    W2 = torch.tensor(nw.weights[1], requires_grad=True, dtype=torch.float64)
    B1 = torch.tensor(nw.biases[0], requires_grad=True, dtype=torch.float64)
    B2 = torch.tensor(nw.biases[1], requires_grad=True, dtype=torch.float64)  

    Z1 = W1 @ X + B1
    A1 = torch.sigmoid(Z1)
    Z2 = W2 @ A1 + B2
    A2 = torch.sigmoid(Z2)

    cost = torch.sum((Y - A2) ** 2)

    cost.backward()

    return (
            [W1.grad.detach().numpy(),
             W2.grad.detach().numpy()], 

             [B1.grad.detach().numpy(), 
              B2.grad.detach().numpy()]
            )