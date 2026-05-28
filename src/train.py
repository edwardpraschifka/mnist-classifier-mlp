import numpy as np
from src.network import Network
from src.utils import get_mnist_data, save, load

def train_mnist():
    """Test model performance on MNIST"""

    layers = [784, 128, 64, 10]
    nw = Network(layers)
    
    (X_train, Y_train, X_test, Y_test) = get_mnist_data()

    nw.gradient_descent(X_train.T, Y_train.T, batch_size=32, step=0.1, epochs=10, display=True)

    feedforward_results = [nw.feedforward(row.reshape(-1, 1)) for row in X_test]
    predictions = np.array([A[-1].flatten() for (Z, A) in feedforward_results])

    return nw


if __name__ == "__main__":
    nw = train_mnist()
    nw.save("network.pkl")