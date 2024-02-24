import numpy as np
class CrossEntropyLoss:

    def __init__(self):
        self.output_tensor = None
        self.error_tensor = None
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = -np.sum(np.log((input_tensor + np.finfo(float).eps))*label_tensor)
        return self.output_tensor

    def backward(self, label_tensor):
        self.error_tensor = -(label_tensor/self.input_tensor)
        return self.error_tensor