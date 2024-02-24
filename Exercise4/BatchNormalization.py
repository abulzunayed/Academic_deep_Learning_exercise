from .Base import BaseLayer
import numpy as np
import copy
from Optimization import Optimizers
from .Helpers import *
class BatchNormalization(BaseLayer):

    def __init__(self, channels):
        super(BatchNormalization, self).__init__()
        self.trainable = True
        self.channels = channels
        self.bias = None
        self.weights = None
        #self.initialize(self.channels)
        self.mean = None
        self.sigma = None
        self.mu_tilde = 0
        self.sigma_tilde = 0
        self.input_tensor_tilde = None
        self.output_tensor = None
        self.alpha = 0.8
        self.gradient_bn_weights = None
        self.gradient_tensor = None
        self.gradient_bn_bias = None
        self.optflag = False
        self.bias = np.zeros([1, self.channels])
        self.weights = np.ones([1, self.channels])


    def initialize(self, weights_initializer, bias_initializer):
        pass

    def forward(self, input_tensor):
        if len(input_tensor.shape)==4:
            self.reformat_in_tensor = self.reformat(input_tensor)
        else:
            self.reformat_in_tensor = input_tensor

        if not self.testing_phase:
            self.mean = np.mean(self.reformat_in_tensor, axis=0)
            self.sigma = np.std(self.reformat_in_tensor, axis=0)
            self.mu_tilde = self.alpha * self.mu_tilde + (1 - self.alpha) * self.mean
            self.sigma_tilde = self.alpha * self.sigma_tilde + (1 - self.alpha) * self.sigma
            self.input_tensor_tilde = (self.reformat_in_tensor - self.mean)/(np.sqrt(np.square(self.sigma) + np.finfo(float).eps))
            self.output_tensor = np.multiply(self.input_tensor_tilde, self.weights) + self.bias
        else:
            self.input_tensor_tilde = (self.reformat_in_tensor - self.mu_tilde)/(np.sqrt(np.square(self.sigma_tilde) + np.finfo(float).eps))
            self.output_tensor = np.multiply(self.input_tensor_tilde, self.weights) + self.bias
        if len(input_tensor.shape) == 4:
            self.output_tensor = self.reformat(self.output_tensor)

        return self.output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._weights_optimizer = copy.deepcopy(optimizer)
        self._bias_optimizer = copy.deepcopy(optimizer)
        self.optflag = True

    def backward(self, error_tensor):
        if len(error_tensor.shape)==4:
            error_tensor_copy = self.reformat(error_tensor)
        else:
            error_tensor_copy = error_tensor

        self.gradient_tensor = compute_bn_gradients(error_tensor_copy, self.reformat_in_tensor, self.weights, self.mean, np.square(self.sigma))
        self.gradient_bn_bias = np.sum(error_tensor_copy, axis=0).reshape([1, self.channels])
        self.gradient_bn_weights = np.sum(np.multiply(error_tensor_copy, self.input_tensor_tilde), axis=0).reshape([1, self.channels])

        if self.optflag:
            self. weights = self._weights_optimizer.calculate_update(self.weights, self.gradient_bn_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bn_bias)

        if len(error_tensor.shape)==4:
            self.gradient_tensor = self.reformat(self.gradient_tensor)

        return self.gradient_tensor

    @property
    def gradient_weights(self):
        return self.gradient_bn_weights

    @property
    def gradient_bias(self):
        return self.gradient_bn_bias

    def reformat(self, input_tensor):
        channels = input_tensor.shape[1]
        if len(input_tensor.shape)==4:
            self.new_reformat_tensor = input_tensor
            reformat_input_tensor = np.zeros([int(np.prod(input_tensor.shape)/channels), channels])
            for channel in range(channels):
                reformat_input_tensor[:, channel] = input_tensor[:,channel].reshape(int(np.prod(input_tensor.shape)/channels))
        else:
            reformat_input_tensor = np.zeros_like(self.new_reformat_tensor)
            for channel in range(channels):
                reformat_input_tensor[:, channel, :, :] = input_tensor[:, channel].reshape(self.new_reformat_tensor.shape[0],self.new_reformat_tensor.shape[2], self.new_reformat_tensor.shape[3])
        return reformat_input_tensor
