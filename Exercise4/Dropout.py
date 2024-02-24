import numpy as np
from .Base import BaseLayer
class Dropout(BaseLayer):

    def __init__(self, probability):
        super(Dropout, self).__init__()
        self.probability = probability
        self.output_tensor = None
        self.previous_error_tensor = None

    def forward(self, input_tensor):
        self.rng = np.random.default_rng(13)
        if self.testing_phase:
            self.output_tensor = input_tensor
        else:
            self.output_tensor = np.zeros_like(input_tensor)
            random_choice = self.rng.uniform(0, 1, np.shape(input_tensor))
            random_choice[random_choice > self.probability] = 0
            random_choice[random_choice > 0] = 1
            self.output_tensor = (input_tensor*random_choice)/self.probability
        return self.output_tensor
    
    def backward(self, error_tensor):
        self.rng = np.random.default_rng(13)
        if self.testing_phase:
            self.previous_error_tensor = error_tensor
        else:
            self.output_tensor = np.zeros_like(error_tensor)
            random_choice = self.rng.uniform(0, 1, np.shape(error_tensor))
            random_choice[random_choice > self.probability] = 0
            random_choice[random_choice > 0] = 1
            self.previous_error_tensor = (error_tensor*random_choice)/self.probability
        return self.previous_error_tensor