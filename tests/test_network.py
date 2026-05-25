import pytest
import numpy as np
import torch

from src.network import Network
from src.utils import cost
from tests.utils import backprop_torch

class TestConstructor:

    def test_network_dims(self):
        """Create a network and check the dimensions of its weight and bias arrays"""

        layers = np.array([3,4,2])
        nw = Network(layers)

        assert nw.size == 3

        assert(np.array_equal(layers, nw.layers))

        assert len(nw.weights) == 2
        assert nw.weights[0].shape == (4,3)
        assert nw.weights[1].shape == (2,4)
        
        assert len(nw.biases) == 2
        assert nw.biases[0].shape == (4,)
        assert nw.biases[1].shape == (2,)

    def test_bad_type(self):
        """Create a network using a layers array of invalid type"""

        with pytest.raises(TypeError):
            layers = list([])
            nw = Network(layers)


class TestFeedForward:

    def test_bad_type(self):
        """Try feedforward using an input array of invalid type"""

        layers = np.array([3,4,2])
        nw = Network(layers)
        X = [1,2,3]

        with pytest.raises(TypeError):
            nw.feedforward(X)

    def test_diff_len(self):
        """Try feedforward using an input array of invalid length"""

        layers = np.array([3,4,2])
        nw = Network(layers)
        X = np.array([1,2,3,4])

        with pytest.raises(ValueError):
            nw.feedforward(X)
    
    def test_output(self):
        """Try feedforward and check accuracy of the result"""

        layers = np.array([3,4,2])
        nw = Network(layers)
        X = np.array([0.1, 0.1, 0.2]).reshape((3,))

        nw.weights[0] = np.array([[0.1, 0.2, 0.3], [0.5, 0.4, 0.2], 
                               [0.1, 0.1, 0.2], [0.3, 0.2 , 0.1]])
        nw.weights[1] = np.array([[0.2, 0.1, 0.2, 0.2], [0.3, 0.1, 0.3, 0.1]])

        
        nw.biases[0] = np.array([0.1, 0.2, -0.1, 0.3])
        nw.biases[1] = np.array([0.1, 0.2])

        (Z,A) = nw.feedforward(X)        

        assert len(Z) == 3
        assert len(A) == 3

        assert np.array_equal(A[0], X)
        assert np.array_equal(np.round(Z[1],3), [0.190, 0.330, -0.040, 0.370])
        assert np.array_equal(np.round(A[1],3), [0.547, 0.582, 0.490, 0.591])
        assert np.array_equal(np.round(Z[2],3), [0.484, 0.629])
        assert np.array_equal(np.round(A[2],3), [0.619, 0.652])


class TestBackProp:

    def test_output(self):
        """Test accuracy of backprop method"""

        layers = np.array([3,4,2])
        nw = Network(layers)
        X = np.array([0.1, 0.1, 0.2]).reshape((3,))

        nw.weights[0] = np.random.randn(4,3)
        nw.weights[1] = np.random.randn(2,4)

        
        nw.biases[0] = np.random.randn(4,)
        nw.biases[1] = np.random.randn(2,)

        Y = np.array([1, 0]).reshape((2,))

        # compute grad_w and grad_b using our function
        (grad_w, grad_b) = nw.backprop(X,Y)

        # test against grad_w and grad_b 
        # computed using pytorch
        (grad_w_torch,grad_b_torch) = backprop_torch(X, Y, nw)

        assert len(nw.weights) == len(grad_w)
        assert len(nw.biases) == len(grad_b)

        assert np.shape(nw.weights[0]) == (4,3)
        assert np.shape(nw.weights[1]) == (2,4)
        
        assert np.shape(nw.biases[0]) == (4,)
        assert np.shape(nw.biases[1]) == (2,)

        assert np.allclose(grad_w[0], grad_w_torch[0])
        assert np.allclose(grad_w[1], grad_w_torch[1])
        assert np.allclose(grad_b[0], grad_b_torch[0].reshape(-1,1))
        assert np.allclose(grad_b[1], grad_b_torch[1].reshape(-1,1))

    def test_output_big(self):
        """Test accuracy of backprop method
            for a bigger network"""

        layers = np.array([20,50,40,30,20,10])
        nw = Network(layers)
        X = np.random.rand(20,)

        nw.weights = [np.random.rand(layers[i+1],layers[i]) for i in range(nw.size - 1)]
        nw.biases = [np.random.rand(layers[i],) for i in range(1, nw.size)]

        Y = np.random.rand(10,)

        # compute grad_w and grad_b using our function
        (grad_w, grad_b) = nw.backprop(X,Y)

        # test against grad_w and grad_b 
        # computed using pytorch
        (grad_w_torch,grad_b_torch) = backprop_torch(X, Y, nw)

        assert len(nw.weights) == len(grad_w)
        assert len(nw.biases) == len(grad_b)

        for i in range(nw.size - 1):
            assert np.shape(nw.weights[i]) == (layers[i+1],layers[i])
            assert np.shape(nw.biases[i]) == (layers[i+1],)
            assert np.allclose(grad_w[i], grad_w_torch[i])
            assert np.allclose(grad_b[i], grad_b_torch[i].reshape(-1,1))