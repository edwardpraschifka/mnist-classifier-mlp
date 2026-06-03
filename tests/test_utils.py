import subprocess
import os
import pytest
import numpy as np
import math

from src.network import Network
from src.main import eval_mnist
from src.utils import shuffle_and_batch, accuracy

class TestMakeBatch:
    def test_diff_len(self):
        cols = 1000
        batch_size = 25
        x_rows = 20
        y_rows = 5

        X = np.random.rand(x_rows, cols)
        Y = np.random.rand(y_rows, cols + 1)

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

            y_predicted[0, i*10: (i+1)*10] = 1