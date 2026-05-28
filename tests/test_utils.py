import pytest
import numpy as np
import math
import os

from src.network import Network
from src.utils import sigmoid, shuffle_and_batch, accuracy

class TestSigmoid:
    def test_output_int(self):
        z = 2
        out = sigmoid(z)

        assert round(out,2) == 0.88
    
    def test_output_vec(self):
        z = np.array([1,2,3])
        out = sigmoid(z)

        expected_out = np.array([0.73, 0.88, 0.95])

        assert np.array_equal(np.round(out,2), expected_out)

class TestMakeBatch:
    def test_diff_len(self):
        rows = 1000
        batch_size = 25
        x_cols = 20
        y_cols = 5

        X = np.random.rand(rows,x_cols)
        Y = np.random.rand(rows + 1,y_cols)

        with pytest.raises(ValueError):
            (X_batches, Y_batches) = shuffle_and_batch(X, Y, batch_size)


    def test_batch_no_remainder(self):
        cols = 1000
        batch_size = 25
        x_rows = 20
        y_rows = 5

        X = np.random.rand(x_rows, cols)
        Y = np.random.rand(y_rows, cols)

        (X_batches,Y_batches) = shuffle_and_batch(X, Y, batch_size)

        for (xb, yb) in zip(X_batches,Y_batches):
            assert np.shape(xb) == (x_rows, batch_size)
            assert np.shape(yb) == (y_rows, batch_size)

        assert len(X_batches) == cols/batch_size
        assert len(Y_batches) == cols/batch_size

    def test_batch_with_remainder(self):
        cols = 1000
        batch_size = 24
        x_rows = 20
        y_rows = 5

        X = np.random.rand(x_rows, cols)
        Y = np.random.rand(y_rows, cols)

        (X_batches,Y_batches) = shuffle_and_batch(X, Y, batch_size)

        for (xb, yb) in zip(X_batches[:-1], Y_batches[:-1]):
            assert np.shape(xb) == (x_rows, batch_size)
            assert np.shape(yb) == (y_rows, batch_size)
        
        assert np.shape(X_batches[-1]) == (x_rows, cols % batch_size)
        assert np.shape(Y_batches[-1]) == (y_rows, cols % batch_size)

        assert len(X_batches) == math.ceil(cols/batch_size)
        assert len(Y_batches) == math.ceil(cols/batch_size)

class TestAccuracy:
    def test_output(self):
        y_actual = np.zeros((10,100))
        y_predicted = np.zeros((10,100))

        # Suppose that label 0
        # is predicted for every training
        # example in y_actual
        y_actual[0, :] = 1

        # suppose that label 9
        # is predicted for every training
        # example in y
        y_predicted[9, :] = 0.1

        for i in range(10):
            # change prediction to label 0
            # for the first i * 10 examples
            acc = accuracy(y_actual, y_predicted)
            assert np.allclose(acc, i * 0.1)

            y_predicted[0, i: (i+1)*10] = 1


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
