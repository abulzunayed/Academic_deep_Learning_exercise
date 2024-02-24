import numpy as np
from .Base import BaseLayer
class SoftMax(BaseLayer):

    def __init__(self):
        super(SoftMax, self).__init__()
        self.output_tensor = None
        self.error_tensor = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_tensor = input_tensor - np.max(input_tensor)
        transit = np.exp(input_tensor)
        self.output_tensor = transit/(np.sum(transit, axis=1).reshape(-1,1))
        return self.output_tensor

    def backward(self, error_tensor):
        self.error_tensor = np.multiply(self.output_tensor, (error_tensor - np.sum(error_tensor * self.output_tensor, axis=1).reshape(-1,1)))
        return self.error_tensor