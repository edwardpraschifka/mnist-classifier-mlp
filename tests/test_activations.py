from src.activations import *

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