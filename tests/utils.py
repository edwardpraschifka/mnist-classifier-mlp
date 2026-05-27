import numpy as np
import torch
from tensorflow.keras.datasets import mnist

from src.network import Network

def backprop_torch(X: np.ndarray, Y: np.ndarray, nw: Network):
    """Conducts backprop using pytorch (used for testing)"""
    
    X = torch.tensor(X, dtype=torch.float64)
    Y = torch.tensor(Y, dtype=torch.float64)
    
    Ws = [torch.tensor(w, requires_grad=True, dtype=torch.float64) for w in nw.weights]
    Bs = [torch.tensor(b, requires_grad=True, dtype=torch.float64) for b in nw.biases]
    
    # forward pass
    A = X
    for W, B in zip(Ws, Bs):
        Z = W @ A + B
        A = torch.sigmoid(Z)
    
    cost = torch.sum((Y - A) ** 2)
    cost.backward()
    
    return [W.grad.detach().numpy() for W in Ws], [B.grad.detach().numpy() for B in Bs]

import torch
import torch.nn as nn


def average_loss_torch(X: np.ndarray, Y: np.ndarray, nw: Network):
    """Conducts backprop using pytorch (used for testing)"""
    
    X = torch.tensor(X, dtype=torch.float64)
    Y = torch.tensor(Y, dtype=torch.float64)
    
    Ws = [torch.tensor(w, dtype=torch.float64) for w in nw.weights]
    Bs = [torch.tensor(b, dtype=torch.float64) for b in nw.biases]

    avg_loss = 0

    for x,y in zip(X,Y): 
        A = x.reshape(-1,1)

        for W, B in zip(Ws, Bs):
            Z = W @ A + B
            A = torch.sigmoid(Z)
    
        avg_loss += torch.sum((y.reshape(-1,1) - A) ** 2)

    return (avg_loss / X.shape[0]).item()


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