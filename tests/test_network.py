import pytest
import numpy as np
import torch
import os

from src.activations import Sigmoid
from src.network import Network
from src.utils import accuracy, get_mnist_data
from tests.utils import feedforward_torch, backprop_torch
from src.cost_functions import CrossEntropyCost


class TestSigmoid:
    def test_output_int(self):
        z = 2
        sig = Sigmoid()
        out = sig.forward(z)

        assert round(out,2) == 0.88
    
    def test_output_vec(self):
        z = np.array([1,2,3])
        sig = Sigmoid()
        out = sig.forward(z)

        expected_out = np.array([0.73, 0.88, 0.95])

        assert np.array_equal(np.round(out,2), expected_out)

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
        """Test accuracy of backprop method with
        with each cost function"""

        layers = [20,50,40,30,20,10]
        cost_functions = [CrossEntropyCost]

        for cost_fn in cost_functions:
            nw = Network(layers, cost_fn=cost_fn)

            # 1000 training examples
            X = np.random.rand(20,1000)
            Y = np.random.rand(10,1000)

            nw.weights = [np.random.rand(layers[i],layers[i-1]) for i in range(1, nw.size)]
            nw.biases = [np.random.rand(layers[i],1) for i in range(1, nw.size)]

            # compute grad_w and grad_b using our function
            (grad_w, grad_b) = nw.backprop(X,Y)
            
            # test against grad_w and grad_b 
            # computed using pytorch
            (grad_w_torch,grad_b_torch) = backprop_torch(X, Y, nw, cost_fn)   

            for gw, gwt in zip(grad_w, grad_w_torch):
                assert np.allclose(gw, gwt)

            for gb, gbt in zip(grad_b, grad_b_torch):
                assert np.allclose(gb, gbt)
    

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

        assert acc > 0.9


class TestReadWrite:
    def test_load(self):
        layers = [3,4,2]
        layers_2 = [2,4,3]

        nw = Network(layers)
        nw_2 = Network(layers_2)

        nw.save("test_network.pkl")
        nw_2.load("test_network.pkl")
        os.remove("test_network.pkl")

        np.array_equal(nw_2.layers, nw.layers)
        np.array_equal(nw_2.size, nw.size)
        np.array_equal(nw_2.weights, nw.weights)
        np.array_equal(nw_2.biases, nw.biases)