import pytest
import numpy as np

from src.utils import cost, sigmoid, shuffle_and_batch

class TestCostFunction:
    def test_output_pos(self):
        y1 = np.array([1,2,3])
        y2 = np.array([4,5,6])

        out = cost(y1,y2)
        assert out == 27

    def test_output_neg(self):
        y1 = np.array([-1,2,-3])
        y2 = np.array([4,-5,6])

        out = cost(y1,y2)
        assert out == 155

    def test_bad_type(self):
        y1 = list([])
        y2 = np.array([])

        with pytest.raises(TypeError):
            out = cost(y1,y2)
        

    def test_diff_len(self):
        y1 = np.array([1,2,3])
        y2 = np.array([1,2])

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
    def test_batch_no_remainder(self):
        X = np.random.rand(1000,20)
        batches = shuffle_and_batch(X, batch_size=25)

        for batch in batches:
            assert np.shape(batch) == (25,20)

        assert len(batches) == 40

    def test_batch_with_remainder(self):
        X = np.random.rand(452,10)
        batches = shuffle_and_batch(X, batch_size=32)

        for batch in batches[:-1]:
            assert np.shape(batch) == (32,10)
        
        assert np.shape(batches[-1]) == (4,10)

        assert len(batches) == 15