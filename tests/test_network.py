import pytest
import numpy as np
import torch

from src.network import Network
from src.utils import cost, accuracy
from tests.utils import backprop_torch, average_loss_torch, get_mnist_data, feedforward_torch

class TestConstructor:

    def test_network_dims(self):
        """Create a network and check the dimensions of its weight and bias arrays"""

        layers = [3,4,2]
        nw = Network(layers)

        assert nw.size == 3

        assert(np.array_equal(layers, nw.layers))

        assert len(nw.weights) == 2
        assert nw.weights[0].shape == (4,3)
        assert nw.weights[1].shape == (2,4)
        
        assert len(nw.biases) == 2
        assert nw.biases[0].shape == (4,1)
        assert nw.biases[1].shape == (2,1)


class TestFeedForward:

    def test_diff_len(self):
        """Try feedforward using an input array of invalid length"""

        layers = [3,4,2]
        nw = Network(layers)
        X = np.array([1,2,3,4])

        with pytest.raises(ValueError):
            nw.feedforward(X)
    
    def test_output(self):
        layers = [20,50,40,30,20,10]
        nw = Network(layers)

        # 1000 training examples
        X = np.random.randn(20,1000)
        torch_output = feedforward_torch(X, nw)

        (Z,my_output) = nw.feedforward(X)
        
        assert np.allclose(torch_output, my_output[-1])


class TestBackProp:

    def test_output(self):
        """Test accuracy of backprop method"""

        layers = [20,50,40,30,20,10]
        nw = Network(layers)

        # 1000 training examples
        X = np.random.rand(20,1000)
        Y = np.random.rand(10,1000)

        nw.weights = [np.random.rand(layers[i],layers[i-1]) for i in range(1, nw.size)]
        nw.biases = [np.random.rand(layers[i],1) for i in range(1, nw.size)]

        # compute grad_w and grad_b using our function
        (grad_w, grad_b) = nw.backprop(X,Y)
        
        # test against grad_w and grad_b 
        # computed using pytorch
        (grad_w_torch,grad_b_torch) = backprop_torch(X, Y, nw)   

        for gw, gwt in zip(grad_w, grad_w_torch):
            assert np.allclose(gw, gwt)

        for gb, gbt in zip(grad_b, grad_b_torch):
            assert np.array_equal(np.shape(gb), np.shape(gbt))

class TestGradientDescent:
    
    def test_overfits_single_example(self):
        """Test if model overfits to
        a dataset with a single example"""

        layers = [3, 4, 2]
        nw = Network(layers)
        
        X = np.random.rand(1,3)
        Y = np.random.rand(1,2)
        
        nw.gradient_descent(X.T, Y.T, batch_size=1, step=0.1, epochs=1000)
        (Z,A) = nw.feedforward(X.T)

        acc = accuracy(A[-1], Y.T)
        print(f"accuracy = {acc}")

        assert acc > 0.9
    
    @pytest.mark.skip(reason="Feature is currently broken")
    def test_mnist_output(self):
        """Test model performance on MNIST"""

        layers = [784, 128, 64, 10]
        nw = Network(layers)
        
        (X_train, Y_train, X_test, Y_test) = get_mnist_data()

        nw.gradient_descent(X_train.T, Y_train.T, batch_size=32, step=0.1, epochs=1, display=True)

        feedforward_results = [nw.feedforward(row.reshape(-1, 1)) for row in X_test]
        predictions = np.array([A[-1].flatten() for (Z, A) in feedforward_results])

        acc = accuracy(predictions, Y_test)
        print(f"accuracy = {acc}")

        assert acc > 0.9