from .Base import BaseLayer
from Optimization import Optimizers
import numpy as np
from scipy.signal import correlate, convolve
import copy
class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super(Conv, self).__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = None
        self.weights = np.random.uniform(0,1, (self.num_kernels, *self.convolution_shape))
        self.bias = None
        self.bias = np.random.uniform(0,1, (self.num_kernels, 1))
        self.batch_size = None
        self.strided_input_tensor = None
        self._optimizer = None
        self._weights_optimizer = None
        self._bias_optimizer = None
        self.optflag = False
        self.new_kernels = None
        self.prev_err_tensor = None
        self.bias_gradient_tensor = None
        self.weights_gradient_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor # (2,3,10,14)
        self.batch_size = self.input_tensor.shape[0]
        if len(self.stride_shape) == 2:
            self.output_tensor = np.zeros([input_tensor.shape[0], self.num_kernels, int(np.ceil(input_tensor.shape[2] / self.stride_shape[0])), int(np.ceil(input_tensor.shape[3] / self.stride_shape[1]))])
            for i in range(self.batch_size):
                for kernel in range(self.num_kernels):
                    output = correlate(self.input_tensor[i], self.weights[kernel], mode="same")  # zero padding
                    self.output_tensor[i, kernel] = output[int(self.weights.shape[1] / 2), ::self.stride_shape[0], ::self.stride_shape[1]] + self.bias[kernel]

        else:
            self.output_tensor = np.zeros([self.batch_size, self.num_kernels, int(np.ceil(self.input_tensor.shape[2]/self.stride_shape[0]))])
            for i in range(self.batch_size):
                for kernel in range(self.num_kernels):
                    output = correlate(self.input_tensor[i], self.weights[kernel], mode="same")
                    self.output_tensor[i, kernel] = output[int(self.weights.shape[1] / 2), ::self.stride_shape[0]] + self.bias[kernel]

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
        self.error_tensor = error_tensor # dimensions: batch size, num_kernels, self.output_tensor.shape[2:]

        # calculation of bias gradient

        self.bias_gradient_tensor = np.zeros([self.num_kernels, 1])
        for kernel in range(self.num_kernels):
            self.bias_gradient_tensor[kernel] += np.sum(self.error_tensor[:,kernel])

        # calculation of previous error tensor

        self.prev_err_tensor = np.zeros_like(self.input_tensor)

        self.upsampled_error_tensor = np.zeros([self.error_tensor.shape[0], self.error_tensor.shape[1], *self.input_tensor.shape[2:]])
        if len(self.stride_shape) == 2:
            self.upsampled_error_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = self.error_tensor
            self.new_kernels = np.zeros([self.weights.shape[1], self.num_kernels, self.convolution_shape[1], self.convolution_shape[2]]) # channels, no. of kernels, shape of kernel
            for i in range(self.weights.shape[1]): # new kernels creation
                self.new_kernels[i] = np.flip(self.weights[:,i,:,:], axis=0)
        else:
            self.upsampled_error_tensor[:, :, ::self.stride_shape[0]] = self.error_tensor
            self.new_kernels = np.zeros([self.weights.shape[1], self.num_kernels, self.convolution_shape[1]]) #channels, no. of kernels, shape of kernel
            for i in range(self.weights.shape[1]): # new kernels creation
                self.new_kernels[i] = np.flip(self.weights[:,i,:], axis=0)

        for i in range(self.error_tensor.shape[0]):
            for kernel in range(self.new_kernels.shape[0]):
                output = convolve(self.upsampled_error_tensor[i], self.new_kernels[kernel], mode="same")
                self.prev_err_tensor[i, kernel] = output[int(self.new_kernels.shape[1] / 2)]

        # calculation of weights gradient tensor
        self.weights_gradient_tensor = np.zeros_like(self.weights)
        if (len(self.convolution_shape) == 3) and (self.convolution_shape[1] != self.convolution_shape[2]):
                kernel_height = self.convolution_shape[1]
                height_padded = int(np.floor(kernel_height/2))
                height_padded_uneven = int(np.floor(kernel_height/2 - 0.5))
                kernel_width = self.convolution_shape[2]
                width_padded = int(np.floor(kernel_width/2))
                width_padded_uneven = int(np.floor(kernel_width/2 - 0.5))
                padded_input_tensor = np.pad(self.input_tensor, ((0, 0), (0, 0), (height_padded, height_padded_uneven), (width_padded, width_padded_uneven)), mode="constant")

        elif len(self.convolution_shape) == 3:
                kernel_height = self.convolution_shape[1]
                height_padded = int(np.floor(kernel_height/2))
                kernel_width = self.convolution_shape[2]
                width_padded = int(np.floor(kernel_width/2))
                padded_input_tensor = np.pad(self.input_tensor, ((0, 0), (0, 0), (height_padded, height_padded), (width_padded, width_padded)), mode="constant")

        else:
            kernel_height = self.convolution_shape[1]
            height_padded = int(np.floor(kernel_height/2))
            padded_input_tensor = np.pad(self.input_tensor, ((0, 0), (0, 0), (height_padded, height_padded)), mode="constant")

        for batch in range(self.error_tensor.shape[0]):
            for channel in range(self.input_tensor.shape[1]):
                for kernel in range(self.num_kernels):
                    self.weights_gradient_tensor[kernel, channel] += correlate(padded_input_tensor[batch, channel], self.upsampled_error_tensor[batch, kernel], mode="valid")

        if self.optflag:
            self.weights = self._weights_optimizer.calculate_update(self.weights, self.weights_gradient_tensor)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.bias_gradient_tensor)

        return self.prev_err_tensor

    @property
    def gradient_weights(self):
        return self.weights_gradient_tensor

    @property
    def gradient_bias(self):
        return self.bias_gradient_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_out = self.num_kernels*np.prod(self.convolution_shape[1:])
        fan_in  = np.prod(self.convolution_shape)
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, 1)
