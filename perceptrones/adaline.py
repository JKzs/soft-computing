import numpy as np

class AdalineGD(object):

    def __init__(self, eta:float, epochs:int):
        self.eta = eta
        self.epochs = epochs
    
    def train(self, X, y):

        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_=[]

        for i in range(self.epochs):
            output = self.net_input(X)
            errors = (y - output)
            self.weights += self.eta * X.T.dot(errors)
            self.bias += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        return self.net_input(X)
    
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
