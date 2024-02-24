import numpy as np
from .Base import BaseLayer

class Flatten(BaseLayer):

    def __init__(self):
        super(Flatten, self).__init__()
        self.output_tensor = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        shape = np.shape(self.input_tensor) #shape is a tuple of integers give lengths of dimensions.
        self.output_tensor = self.input_tensor.reshape(shape[0], np.prod(shape[1:]))
        return self.output_tensor

    def backward(self, error_tensor):
        error_tensor = error_tensor.reshape(np.shape(self.input_tensor))
        return error_tensor