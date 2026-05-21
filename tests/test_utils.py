import pytest
import numpy as np

from src.utils import cost

def test_cost_1():
    y1 = np.array([1,2,3])
    y2 = np.array([4,5,6])

    out = cost(y1,y2)
    assert out == 27

def test_cost_2():
    y1 = np.array([-1,2,-3])
    y2 = np.array([4,-5,6])

    out = cost(y1,y2)
    assert out == 155

def test_bad_type():
    y1 = list([])
    y2 = np.array([])

    with pytest.raises(TypeError):
        out = cost(y1,y2)
    

def test_diff_len():
    y1 = np.array([1,2,3])
    y2 = np.array([1,2])

    with pytest.raises(ValueError):
        out = cost(y1,y2)

