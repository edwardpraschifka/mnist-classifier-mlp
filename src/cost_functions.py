import numpy as np

class QuadraticCost:

    def forward(self, a, y):
        return (a-y)**2
    
    def derivative(self, a, y):
        return 2*(a-y)