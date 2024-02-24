from .Base import BaseLayer
class ReLU(BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()
        self.input_tensor = None
        self.output_tensor = None
        self.error_tensor = None


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = input_tensor
        self.output_tensor[input_tensor < 0] = 0
        return self.output_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.error_tensor[self.input_tensor <= 0] = 0

        return self.error_tensor
