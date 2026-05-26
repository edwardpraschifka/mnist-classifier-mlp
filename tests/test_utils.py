import pytest
import numpy as np
import math

from src.utils import cost, sigmoid, shuffle_and_batch

class TestCostFunction:
    def test_output_pos(self):
        y1 = np.array([1,2,3]).reshape(-1,1)
        y2 = np.array([4,5,6]).reshape(-1,1)

        out = cost(y1,y2)
        assert out == 27

    def test_output_neg(self):
        y1 = np.array([-1,2,-3]).reshape(-1,1)
        y2 = np.array([4,-5,6]).reshape(-1,1)

        out = cost(y1,y2)
        assert out == 155

    def test_bad_type(self):
        y1 = list([])
        y2 = np.array([])

        with pytest.raises(TypeError):
            out = cost(y1,y2)
    
    def test_bad_shape(self):
        y1 = np.array([1,2,3])
        y2 = np.array([1,2,3])

        with pytest.raises(ValueError):
            out = cost(y1,y2)

    def test_diff_len(self):
        y1 = np.array([1,2,3]).reshape(-1,1)
        y2 = np.array([1,2]).reshape(-1,1)

        with pytest.raises(ValueError):
            out = cost(y1,y2)


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
        rows = 1000
        batch_size = 25
        x_cols = 20
        y_cols = 5

        X = np.random.rand(rows,x_cols)
        Y = np.random.rand(rows,y_cols)

        (X_batches,Y_batches) = shuffle_and_batch(X, Y, batch_size)

        for (xb, yb) in zip(X_batches,Y_batches):
            assert np.shape(xb) == (batch_size,x_cols)
            assert np.shape(yb) == (batch_size,y_cols)

        assert len(X_batches) == rows/batch_size
        assert len(Y_batches) == rows/batch_size

    def test_batch_with_remainder(self):
        rows = 1000
        batch_size = 24
        x_cols = 20
        y_cols = 5

        X = np.random.rand(rows,x_cols)
        Y = np.random.rand(rows,y_cols)

        (X_batches,Y_batches) = shuffle_and_batch(X, Y, batch_size)

        for (xb, yb) in zip(X_batches[:-1], Y_batches[:-1]):
            assert np.shape(xb) == (batch_size,x_cols)
            assert np.shape(yb) == (batch_size,y_cols)
        
        assert np.shape(X_batches[-1]) == (rows % batch_size, x_cols)
        assert np.shape(Y_batches[-1]) == (rows % batch_size, y_cols)

        assert len(X_batches) == math.ceil(rows/batch_size)
        assert len(Y_batches) == math.ceil(rows/batch_size)