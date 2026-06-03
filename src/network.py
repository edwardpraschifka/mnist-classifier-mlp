import numpy as np
import pickle

from .utils import Sigmoid, shuffle_and_batch

class Network:
    def __init__(self, layers: np.ndarray, activation_fn = Sigmoid):
        """Creates a new network, where layer i has layers[i] nodes"""
    
        self.layers = layers
        self.size = len(layers)

        # size of weights and biases are both size(layers) - 1
        self.weights = [np.random.randn(layers[i], layers[i-1]) * np.sqrt(1/layers[i-1]) for i in range(1, self.size)]
        self.biases = [np.zeros((layers[i], 1)) for i in range(1, self.size)]

        self._activation_fn = activation_fn()
             
    
    def save(self, filename):
        """Saves network to file"""

        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    def load(self, filename: str):
        """Loads parameters from existing network"""

        with open(filename, 'rb') as f:
            nw = pickle.load(f)
            self.layers = nw.layers
            self.size = nw.size
            self.weights = nw.weights
            self.biases = nw.biases
    
    def _validate_inputs(self, X: np.ndarray, Y = None):
        """Check X and Y inputs against size of input
        and output layers"""

        # Check if dimension of X is 2
        if np.ndim(X) != 2:
             raise ValueError(f"X has dimension {np.ndim(X)}, but expected dim=2")

        # Check if number of features in each
        # example matches size of input layer
        if np.shape(X)[0] != self.layers[0]:
                raise ValueError(f"Input layer expected {self.layers[0]} rows"
                                 f" (X has shape {np.shape(X)})")

        if Y is not None:

            # Check if dimension of Y is 2
            if np.ndim(Y) != 2:
                 raise ValueError(f"Y has dimension {np.ndim(Y)}, but expected dim=2")
        
            # Check if number of training examples
            # in X matches number of training examples in Y
            if np.shape(X)[1] != np.shape(Y)[1]: 
                raise ValueError("X has "
                                f"{np.shape(X)[1]} columns, "
                                f"but Y has {np.shape(Y)[1]} columns")
            
            # Check if number of entries in Y
            # matches size of output layer
            if np.shape(Y)[0] != self.layers[-1]:
                    raise ValueError(f"Output layer expected {self.layers[-1]} rows"
                                     f" (Y has shape {np.shape(Y)}")

    
    def feedforward(self, X: np.ndarray):
        """Feeds a matrix X whose columns are training
        examples into the network, returns corresponding Y
        whose columns are the predicted values"""

        self._validate_inputs(X)
         
        Z = [0] * self.size
        A = [0] * self.size

        A[0] = X

        for i in range(1, self.size):
            Z[i] = (self.weights[i-1] @ A[i-1]) + self.biases[i-1]
            A[i] = self._activation_fn.forward(Z[i])
        
        return (Z,A)
    

    def backprop(self, X: np.ndarray, Y: np.ndarray):
            """Returns partial derivatives of cost function
                with respect to all weights and biases for matrices
                X and Y, whose columns are training examples"""

            self._validate_inputs(X)
            
            grad_w = self.weights.copy()
            grad_b = self.biases.copy()

            (Z, A) = self.feedforward(X)        

            # derivative of cost function with respect
            # to activations of output layer
            dcda = (A[-1] - Y)/(A[-1] * (1 - A[-1]))

            for i in range(self.size - 1, 0, -1):
                # c = cost function
                # a = activation value vector for current layer
                # z = pre-activation value vector for current layer
                # w = weight vector for current layer
                # b = bias vector for current layer.
                # a0 = activation value vector for next layer
                
                dadz = A[i] * (1 - A[i])
                dcdz = (dcda * dadz)

                dzdw = A[i-1]
              
                
                dcdw = dcdz @ dzdw.T
                dcdb = np.sum(dcdz, axis=1)

                grad_w[i-1] = dcdw
                grad_b[i-1] = dcdb.reshape(-1,1)

                dzda0 = self.weights[i-1]

                # update dcda for next layer
                dcda = dzda0.T @ dcdz 
            
            return (grad_w, grad_b)

            
    def gradient_descent(self, X: np.ndarray, Y: np.ndarray, batch_size=32, 
                         step=0.1, epochs=10, display=False):
        """Performs gradient descent, modifying self.weights
            and self.biases in-place"""

        self._validate_inputs(X)
        
        for e in range(epochs):
            (X_batches, Y_batches) = shuffle_and_batch(X, Y, batch_size)

            for xb, yb, i in zip(X_batches, Y_batches, range(len(X_batches))):

                if display==True:
                    print(f"[epoch {e}] processed {i}/{len(X_batches)}")

                avg_grad_w = [np.zeros(np.shape(self.weights[i])) for i in range(self.size - 1)]
                avg_grad_b = [np.zeros(np.shape(self.biases[i])) for i in range(self.size - 1)]

                # run forward pass & backprop
                (grad_w, grad_b) = self.backprop(xb, yb)

                for i in range(self.size - 1):
                    avg_grad_w[i] += grad_w[i]/np.shape(xb)[1]
                    avg_grad_b[i] += grad_b[i]/np.shape(xb)[1]
                
                for i in range(self.size - 1):
                    self.weights[i] -= step * avg_grad_w[i]
                    self.biases[i] -= step * avg_grad_b[i]