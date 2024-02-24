import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.weight_tensor = None
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weight_tensor = np.ones(weights_shape)*self.value
        return self.weight_tensor

class UniformRandom:
    def __init__(self):
        self.weight_tensor = None

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weight_tensor = np.random.uniform(0,1,weights_shape)
        return self.weight_tensor


class Xavier:
    def __init__(self):
        self.weight_tensor = None
        self.sigma = 0.0

    def initialize(self, weights_shape, fan_in, fan_out):
        self.sigma = np.sqrt(2 / (fan_in + fan_out))
        self.weight_tensor = np.random.normal(0, self.sigma, weights_shape)
        return self.weight_tensor


class He:
    def __init__(self):
        self.weight_tensor = None
        self.sigma = 0.0

    def initialize(self, weights_shape, fan_in, fan_out):
        self.sigma = np.sqrt(2 / (fan_in))
        self.weight_tensor = np.random.normal(0, self.sigma, weights_shape)
        return self.weight_tensor