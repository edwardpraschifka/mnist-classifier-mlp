import pytest
import numpy as np

from src.network import Network

def test_network_dims():
    """Create a 3-layer network with layers of size 3,4,2"""

    layers = np.array([3,4,2])
    nw = Network(layers)

    assert nw.size == 3

    assert len(nw.weights) == 3
    assert nw.weights[0].shape == (3,4)
    assert nw.weights[1].shape == (4,2)
    
    assert len(nw.biases) == 3
    assert len(nw.biases[0]) == 3
    assert len(nw.biases[1]) == 4
    assert len(nw.biases[2]) == 2

def test_bad_type():
    with pytest.raises(TypeError):
        layers = list([])
        nw = Network(layers)
    