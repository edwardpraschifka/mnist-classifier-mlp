import numpy as np
from src.network import Network
from src.utils import get_mnist_data

def train_mnist():
    """Train model on MNIST dataset"""

    layers = [784, 128, 64, 10]
    nw = Network(layers)

    (X_train, Y_train, _, _) = get_mnist_data()
    nw.gradient_descent(X_train.T, Y_train.T, batch_size=32, step=0.01, epochs=30, display=True)

    return nw


if __name__ == "__main__":
    nw = train_mnist()
    nw.save("network.pkl")