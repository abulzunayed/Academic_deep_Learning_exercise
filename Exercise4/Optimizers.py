import numpy as np
import copy

class Optimizer:
    
    def __init__(self):
        self.regularizer = None
        self.sub_gradient = None
    
    def add_regularizer(self, regularizer):
        self.regularizer = copy.deepcopy(regularizer)

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super(Sgd, self).__init__()
        self.learning_rate = float(learning_rate)
        self.weight_tensor = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            self.sub_gradient = self.regularizer.calculate_gradient(weight_tensor)
            self.weight_tensor = weight_tensor - self.learning_rate*(gradient_tensor + self.sub_gradient)
        else:
            self.weight_tensor = weight_tensor - self.learning_rate*gradient_tensor

        return self.weight_tensor

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super(SgdWithMomentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.weight_tensor = None
        self.previous_gradient_tensor = None

    def calculate_update(self, weight_tensor, gradient_tensor):

        if self.previous_gradient_tensor is None:
            self.previous_gradient_tensor = np.zeros_like(gradient_tensor)
        gradient_tensor = self.momentum_rate*self.previous_gradient_tensor - (self.learning_rate*gradient_tensor)
        if self.regularizer is not None:
            self.sub_gradient = self.regularizer.calculate_gradient(weight_tensor)
            self.weight_tensor = weight_tensor + gradient_tensor - self.learning_rate*self.sub_gradient
        else:
            self.weight_tensor = weight_tensor + gradient_tensor
        self.previous_gradient_tensor = gradient_tensor
        return self.weight_tensor

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super(Adam, self).__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.weight_tensor = None
        self.v_tensor = None
        self.r_tensor = None
        self.itr=0

    def calculate_update(self, weight_tensor, gradient_tensor):

        if self.itr == 0:
            self.v_tensor = np.zeros_like(gradient_tensor)
            self.r_tensor = np.zeros_like(gradient_tensor)

        self.itr = self.itr + 1
        self.v_tensor = self.mu*self.v_tensor + (1 - self.mu)*gradient_tensor
        self.r_tensor = self.rho * self.r_tensor + (1 - self.rho) * (gradient_tensor**2)

        self.v_hat = self.v_tensor/(1 - np.power(self.mu, self.itr))
        self.r_hat = self.r_tensor / (1 - np.power(self.rho, self.itr))

        if self.regularizer is not None:
            self.sub_gradient = self.regularizer.calculate_gradient(weight_tensor)
            self.weight_tensor = weight_tensor - self.learning_rate*self.sub_gradient - self.learning_rate * (self.v_hat / (np.sqrt(self.r_hat) + np.finfo(float).eps))
        else:
            self.weight_tensor = weight_tensor - self.learning_rate*(self.v_hat/(np.sqrt(self.r_hat) + np.finfo(float).eps))

        return self.weight_tensor