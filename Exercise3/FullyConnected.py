from .Base import BaseLayer
from Optimization import Optimizers
import numpy as np
class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = None
        self.weights = np.random.uniform(low=0, high=1, size=[output_size, input_size])
        self.biases = None
        self.biases = np.random.uniform(low=0, high=1, size=[output_size, 1])
        self.output_tensor = None
        self._optimizer = None
        self.error_tensor = None
        self.gradient_tensor = None
        self.input_tensor = None
        self.optflag = False
        self.weights = np.concatenate((self.weights, self.biases), axis=1)  # size - outputsize*inputsize + 1

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.input_size
        fan_out = self.output_size
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        #self.biases = bias_initializer.initialize(self.biases.shape, fan_in, fan_out)
        #self.weights = np.concatenate((self.weights, self.biases), axis=1)  # size - outputsize*inputsize + 1


    def forward(self, input_tensor):
        self.input_tensor = np.concatenate((input_tensor, np.ones([input_tensor.shape[0], 1])), axis=1)
        self.output_tensor = np.dot(self.input_tensor, self.weights.T)
        return self.output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.optflag = True

    def backward(self, error_tensor):
        self.error_tensor = np.dot(error_tensor, self.weights[:,0:self.weights.shape[1]-1]) #size - batchsize*inputsize
        self.gradient_tensor = np.dot(error_tensor.T, self.input_tensor)  # size - outputsize*inputsize + 1
        if self.optflag:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_tensor)
        return self.error_tensor

    @property
    def gradient_weights(self):
        return self.gradient_tensor