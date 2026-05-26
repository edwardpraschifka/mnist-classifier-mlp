import pytest
import numpy as np
import torch

from src.network import Network
from src.utils import cost
from tests.utils import backprop_torch, average_loss_torch

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

    def test_bad_type(self):
        """Create a network using a layers array of invalid type"""

        with pytest.raises(TypeError):
            layers = np.array([])
            nw = Network(layers)


class TestFeedForward:

    def test_bad_type(self):
        """Try feedforward using an input array of invalid type"""

        layers = [3,4,2]
        nw = Network(layers)
        X = [1,2,3]

        with pytest.raises(TypeError):
            nw.feedforward(X)

    def test_diff_len(self):
        """Try feedforward using an input array of invalid length"""

        layers = [3,4,2]
        nw = Network(layers)
        X = np.array([1,2,3,4])

        with pytest.raises(ValueError):
            nw.feedforward(X)
    
    def test_output(self):
        """Try feedforward and check accuracy of the result"""

        layers = [3,4,2]
        nw = Network(layers)
        X = np.array([0.1, 0.1, 0.2]).reshape(3,1)

        nw.weights[0] = np.array([[0.1, 0.2, 0.3], [0.5, 0.4, 0.2], 
                               [0.1, 0.1, 0.2], [0.3, 0.2 , 0.1]])
        nw.weights[1] = np.array([[0.2, 0.1, 0.2, 0.2], [0.3, 0.1, 0.3, 0.1]])

        
        nw.biases[0] = np.array([0.1, 0.2, -0.1, 0.3]).reshape(4,1)
        nw.biases[1] = np.array([0.1, 0.2]).reshape(2,1)

        (Z,A) = nw.feedforward(X)        

        assert len(Z) == 3
        assert len(A) == 3

        assert np.array_equal(A[0], X)
        assert np.array_equal(np.round(Z[1],3), np.array([0.190, 0.330, -0.040, 0.370]).reshape(4,1))
        assert np.array_equal(np.round(A[1],3), np.array([0.547, 0.582, 0.490, 0.591]).reshape(4,1))
        assert np.array_equal(np.round(Z[2],3), np.array([0.484, 0.629]).reshape(2,1))
        assert np.array_equal(np.round(A[2],3), np.array([0.619, 0.652]).reshape(2,1))


class TestAverageLoss:
    def test_diff_len(self):
        """Case where X and Y have a different
        number of training examples"""

        layers = [3,4,2]
        nw = Network(layers)
        X = np.random.randn(100,3)
        Y = np.random.randn(99,2)

        with pytest.raises(ValueError):
            avg_loss = nw.average_loss(X,Y)


    def test_x_shape(self):
        """Case where number of rows in X
        differs from size of input layer"""

        layers = [3,4,2]
        nw = Network(layers)
        X = np.random.randn(100,7)
        Y = np.random.randn(100,2)

        with pytest.raises(ValueError):
            avg_loss = nw.average_loss(X,Y)


    def test_y_shape(self):
        """Case where number of rows in Y
        differs from size of output layer"""

        layers = [3,4,2]
        nw = Network(layers)
        X = np.random.randn(100,3)
        Y = np.random.randn(100,4)

        with pytest.raises(ValueError):
            avg_loss = nw.average_loss(X,Y)
    

    def test_output(self):
        """Test accuracty of output"""

        layers = [3,4,2]
        nw = Network(layers)
        X = np.random.randn(100,3)
        Y = np.random.randn(100,2)

        torch_loss = average_loss_torch(X,Y, nw)
        my_loss = nw.average_loss(X,Y)

        assert np.isclose(torch_loss, my_loss, atol=1e-5)


class TestBackProp:

    def test_output(self):
        """Test accuracy of backprop method"""

        layers = [3,4,2]
        nw = Network(layers)
        X = np.array([0.1, 0.1, 0.2]).reshape(3,1)

        nw.weights[0] = np.random.randn(4,3)
        nw.weights[1] = np.random.randn(2,4)

        
        nw.biases[0] = np.random.randn(4,1)
        nw.biases[1] = np.random.randn(2,1)

        Y = np.array([1, 0]).reshape((2,1))

        # compute grad_w and grad_b using our function
        (grad_w, grad_b) = nw.backprop(X,Y)

        # test against grad_w and grad_b 
        # computed using pytorch
        (grad_w_torch,grad_b_torch) = backprop_torch(X, Y, nw)

        assert len(nw.weights) == len(grad_w)
        assert len(nw.biases) == len(grad_b)

        assert np.shape(nw.weights[0]) == (4,3)
        assert np.shape(nw.weights[1]) == (2,4)
        
        assert np.shape(nw.biases[0]) == (4,1)
        assert np.shape(nw.biases[1]) == (2,1)

        assert np.allclose(grad_w[0], grad_w_torch[0])
        assert np.allclose(grad_w[1], grad_w_torch[1])
        assert np.allclose(grad_b[0], grad_b_torch[0].reshape(-1,1))
        assert np.allclose(grad_b[1], grad_b_torch[1].reshape(-1,1))

    def test_output_big(self):
        """Test accuracy of backprop method
            for a bigger network"""

        layers = [20,50,40,30,20,10]
        nw = Network(layers)
        X = np.random.rand(20,1)

        nw.weights = [np.random.rand(layers[i+1],layers[i]) for i in range(nw.size - 1)]
        nw.biases = [np.random.rand(layers[i],1) for i in range(1, nw.size)]

        Y = np.random.rand(10,1)

        # compute grad_w and grad_b using our function
        (grad_w, grad_b) = nw.backprop(X,Y)

        # test against grad_w and grad_b 
        # computed using pytorch
        (grad_w_torch,grad_b_torch) = backprop_torch(X, Y, nw)

        assert len(nw.weights) == len(grad_w)
        assert len(nw.biases) == len(grad_b)

        for i in range(nw.size - 1):
            assert np.shape(nw.weights[i]) == (layers[i+1],layers[i])
            assert np.shape(nw.biases[i]) == (layers[i+1],1)
            assert np.allclose(grad_w[i], grad_w_torch[i])
            assert np.allclose(grad_b[i], grad_b_torch[i].reshape(-1,1))