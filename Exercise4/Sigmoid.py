import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.output_tensor = None
        self.previous_error_tensor = None

    def forward(self, input_tensor):
        self.output_tensor = 1/(1 + np.exp(-input_tensor))
        return self.output_tensor

    def backward(self, error_tensor):
        self.previous_error_tensor = self.output_tensor*(1 - self.output_tensor)*error_tensor
        return self.previous_error_tensor

    # @property
    # def activation(self):
    #     return self.output_tensor
    #
    # @activation.setter
    # def activation(self, output_tensor):
    #     self.output_tensor = output_tensor