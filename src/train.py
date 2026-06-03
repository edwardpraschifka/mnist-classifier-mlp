import argparse

from src.network import Network
from src.utils import get_mnist_data

def train_mnist(batch_size: int, learning_rate: float, epochs: int):
    """Train model on MNIST dataset"""

    layers = [784, 128, 64, 10]
    nw = Network(layers)

    (X_train, Y_train, _, _) = get_mnist_data()
    nw.gradient_descent(X_train.T, Y_train.T, batch_size, learning_rate, epochs, display=True)

    return nw


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a neural network on the MNIST dataset")
    parser.add_argument("-bs", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("-lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("-e", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("-o", type=str, default="network.pkl", help="Output file (default: 'network.pkl')")
    args = parser.parse_args()

    nw = train_mnist(batch_size = args.bs, learning_rate = args.lr, epochs = args.e)
    nw.save(filename = args.o)