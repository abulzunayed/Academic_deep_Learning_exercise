import numpy as np
from .Optimizers import Optimizer

class L2_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        sub_gradient = self.alpha*weights
        return sub_gradient

    def norm(self, weights):
        norm = self.alpha*(np.linalg.norm(weights)**2)
        return norm


class L1_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        sub_gradient = self.alpha * np.sign(weights)
        return sub_gradient

    def norm(self, weights):
        norm = self.alpha*np.sum(np.abs(weights))
        return norm
