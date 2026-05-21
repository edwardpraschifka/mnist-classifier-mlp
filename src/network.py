import numpy as np

class Network:
    def __init__(self, layers: np.ndarray):
        if type(layers) != np.ndarray: 
            raise TypeError("layers expected <np.ndarray>" 
                        f", got {type(layers)}")
    
        self.layers = np.array(int)
        self.size = len(layers)
        self.weights = [0] * self.size
        self.biases = [0] * self.size

        for i in range(self.size-1):
            self.weights[i] = np.random.rand(layers[i], layers[i+1])
        
        for i in range(self.size):
            self.biases[i] = np.random.rand(layers[i])
