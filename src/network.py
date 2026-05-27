import numpy as np

from .utils import cost, sigmoid, shuffle_and_batch

class Network:
    def __init__(self, layers: np.ndarray):
        """Creates a new network, where layer i has layers[i] nodes"""

        if type(layers) != list: 
            raise TypeError("layers expected <class 'list'>" 
                        f", got {type(layers)}")
    
        self.layers = layers
        self.size = len(layers)

        # size of weights and biases are both size(layers) - 1
        self.weights = [np.random.rand(layers[i], layers[i-1]) for i in range(1, self.size)]
        self.biases = [np.random.rand(layers[i],1) for i in range(1, self.size)]

    def feedforward(self, X: np.ndarray):
        """Feeds a single training example
        into the network, returns corresponding Y"""

        if type(X) != np.ndarray: 
            raise TypeError("X expected <np.ndarray>" 
                        f", got {type(X)}")
        
        # check that the shape of X matches
        # the number of nodes in the input layer
        if np.shape(X) != (self.layers[0],1):
            raise ValueError("Expected X to have shape"
                             f" ({self.layers[0]}, 1), but "
                             f"got shape {np.shape(X)}")
        
        Z = [None] * (self.size)
        A = [None] * self.size
        Z[0] = X
        A[0] = X

        for i in range(1, self.size):
            Z[i] = (self.weights[i-1] @ A[i-1]) + self.biases[i-1]
            A[i] = sigmoid(Z[i])


        return (Z,A)
    
    def feedforward_vec(self, X: np.ndarray):
        """Feeds a matrix X whose columns are training
        examples into the network, returns corresponding Y
        whose columns are the predicted values"""

        if type(X) != np.ndarray: 
            raise TypeError("X expected <np.ndarray>" 
                        f", got {type(X)}")

         
        Z = [0] * self.size
        A = [0] * self.size

        A[0] = X

        for i in range(1, self.size):
            Z[i] = (self.weights[i-1] @ A[i-1]) + self.biases[i-1]
            A[i] = sigmoid(Z[i])
        
        return A[-1]


    
    def average_loss(self, X: np.ndarray, Y: np.ndarray):
        """Takes a set of training examples, and returns
        the network's average loss across the set"""

        if np.ndim(X) != 2:
             raise ValueError("Expected X to be a 2D array, "
                              f"but got shape {np.shape(X)}")
        
        if np.ndim(Y) != 2:
             raise ValueError("Expected Y to be a 2D array, "
                              f"but got shape {np.shape(Y)}")
             

        if np.shape(X)[0] != np.shape(Y)[0]: 
            raise ValueError("Shape mismatch: X has "
                            f"{np.shape(X)[0]} rows, "
                            f"but Y has {np.shape(Y)[0]} rows")
        
        if np.shape(X)[1] != self.layers[0]:
                raise ValueError("Expected X to have shape"
                                f" ({np.shape(X)[0]}, {self.layers[0]}), "
                                f"but got shape {np.shape(X)}")
            
        if np.shape(Y)[1] != self.layers[-1]:
                raise ValueError("Expected Y to have shape"
                                f" ({np.shape(Y)[0]}, {self.layers[-1]}), "
                                f"but got shape {np.shape(Y)}")
        
        result = 0
        rows = np.shape(X)[0]
        
        for x,y in zip(X,Y):
            (Z,A) = self.feedforward(x.reshape(-1,1))
            result += cost(y.reshape(-1,1), A[-1])
        
        return result/rows


    def backprop(self, X, Y):
            """Returns partial derivatives of cost function
                with respect to all weights and biases for a
                given training example"""
            
            if type(X) != np.ndarray:
                raise TypeError("X expected <np.ndarray>" 
                            f", got {type(X)}")
            
            if type(Y) != np.ndarray:
                raise TypeError("X expected <np.ndarray>" 
                            f", got {type(Y)}")

            if np.shape(X) != (self.layers[0],1):
                raise ValueError("Expected X to have shape"
                                f" ({self.layers[0]}, 1), but "
                                f"got shape {np.shape(X)}")
            
            if np.shape(Y) != (self.layers[-1],1):
                raise ValueError("Expected Y to have shape"
                                f" ({self.layers[-1]}, 1), but "
                                f"got shape {np.shape(Y)}")
            
            grad_w = self.weights.copy()
            grad_b = self.biases.copy()

            (Z, A) = self.feedforward(X)

            # derivative of cost function with respect
            # to activations of output layer
            dcda = 2 * (A[-1] - Y)

            for i in range(self.size - 1, 0, -1):
                # c = cost function
                # a = activation value vector for current layer
                # z = pre-activation value vector for current layer
                # w = weight vector for current layer
                # b = bias vector for current layer.
                # a0 = activation value vector for next layer
                
                dadz = sigmoid(Z[i]) * (1 - sigmoid(Z[i]))
                dcdz = (dcda * dadz).reshape(-1,1)

                dzdw = A[i-1].reshape(-1,1)
                dzdb = 1
              
                
                dcdw = dcdz @ dzdw.T
                dcdb = dcdz * dzdb

                grad_w[i-1] = dcdw
                grad_b[i-1] = dcdb

                dzda0 = self.weights[i-1]

                # update dcda for next layer
                dcda = dzda0.T @ dcdz 
            
            return (grad_w, grad_b)
    

    def gradient_descent(self, X: np.ndarray, Y: np.ndarray, batch_size=32, 
                         step=0.1, epochs=10, display=False):
        """Performs gradient descent, modifying self.weights
            and self.biases in-place"""
        
        for e in range(epochs):
            (X_batches, Y_batches) = shuffle_and_batch(X, Y, batch_size)

            for xb, yb, i in zip(X_batches, Y_batches, range(len(X_batches))):

                if display==True:
                    print(f"[epoch {e}] processed {i}/{len(X_batches)}")

                avg_grad_w = [np.zeros(np.shape(self.weights[i])) for i in range(self.size - 1)]
                avg_grad_b = [np.zeros(np.shape(self.biases[i])) for i in range(self.size - 1)]

                # for each training example in our batch,
                # run forward pass & backprop
                for i in range(np.shape(xb)[0]):
    
                    (grad_w, grad_b) = self.backprop(xb[i].reshape(-1,1), yb[i].reshape(-1,1))

                    for i in range(self.size - 1):
                        avg_grad_w[i] += grad_w[i]/batch_size
                        avg_grad_b[i] += grad_b[i]/batch_size
                
                for i in range(self.size - 1):
                    self.weights[i] -= step * avg_grad_w[i]
                    self.biases[i] -= step * avg_grad_b[i]