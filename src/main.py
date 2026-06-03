import numpy as np
import argparse

from src.network import Network
from src.utils import get_mnist_data, accuracy

def eval_mnist(filename: str):
    """Test model performance on MNIST"""

    nw = Network([])
    nw.load(filename)
    
    (_, _, X_test, Y_test) = get_mnist_data()

    feedforward_results = [nw.feedforward(row.reshape(-1, 1)) for row in X_test]
    predictions = np.array([A[-1].flatten() for (Z, A) in feedforward_results])
    acc = accuracy(predictions.T, Y_test.T)

    return acc
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on the MNIST dataset")
    parser.add_argument("-f", type=str, default="network.pkl", help="Input file (default: 'network.pkl')")
    args = parser.parse_args()
    
    acc = eval_mnist(args.f)
    print(f"Accuracy = {acc*100}%")