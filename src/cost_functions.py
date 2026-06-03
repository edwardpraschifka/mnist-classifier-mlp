import numpy as np

class QuadraticCost:

    def forward(self, a, y):
        return (a-y)**2
    
    def derivative(self, a, y):
        return 2*(a-y)


class CrossEntropyCost:

    def forward(self, a, y):
        return -( y * np.log(a) + (1 - y) * np.log(1 - a))
        
    
    def derivative(self, a, y):
        return (a - y)/(a * (1 - a))
    