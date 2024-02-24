import numpy as np
from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .Sigmoid import Sigmoid
from .TanH import TanH
class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._memorize = False
        self.fully_connected_I = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fully_connected_II = FullyConnected(self.hidden_size, self.output_size)
        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        self.hidden_state = np.zeros([1, self.hidden_size])
        self._optimizer = None
        self.optflag = False
        self.output_tensor = None
        self.gradient_weights_tensor = None
        self.regularization_loss = 0
        self.weights = np.zeros((self.hidden_size, self.input_size+self.hidden_size+1))

    @property
    def memorize(self):
        return self._memorize
    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    def forward(self, input_tensor):
        self.output_tensor = np.zeros((input_tensor.shape[0], self.output_size))
        self.fc_input_tensor = np.zeros((input_tensor.shape[0], self.hidden_size + self.input_size))# 9,20
        self.fc_tanh = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.fc_sigmoid = np.zeros((input_tensor.shape[0], self.output_size))
        self.batch_hidden_state = np.zeros((input_tensor.shape[0], self.hidden_size))
        for batch in range(input_tensor.shape[0]):
            self.batch_hidden_state[batch] = self.hidden_state
            self.fc_input_tensor[batch] = np.concatenate((self.hidden_state, input_tensor[batch].reshape(1,-1)), axis=1)#1,20 and reshape(1,-1) means columnn is unknown, row is 1.
            self.fc_tanh[batch] = self.fully_connected_I.forward(self.fc_input_tensor[batch].reshape(1,-1))
            self.hidden_state = self.tanh.forward(self.fc_tanh[batch].reshape(1,-1))
            self.fc_sigmoid[batch] = self.fully_connected_II.forward(self.hidden_state)
            self.output_tensor[batch] = self.sigmoid.forward(self.fc_sigmoid[batch].reshape(1,-1))
        return self.output_tensor
    @property
    def gradient_weights(self):
        return self.gradient_weights_tensor
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.optflag = True
    def calculate_regularization_loss(self):
        self.regularization_loss += self._optimizer.regularizer.norm(self.weights)
        return self.regularization_loss
    @property
    def rnn_weights(self):
        self.weights = self.fully_connected_I.weights
        return self.weights
    @rnn_weights.setter
    def rnn_weights(self, weights):
        self.weights = weights
    def backward(self, error_tensor):
        self.previous_error_tensor = np.zeros((error_tensor.shape[0], self.input_size))
        self.sigmoid_err_tensor = np.zeros((error_tensor.shape[0], self.output_size))
        self.previous_error_tensor_II = np.zeros((error_tensor.shape[0], self.hidden_size))
        self.previous_error_tensor_I = np.zeros((error_tensor.shape[0], self.hidden_size + self.input_size))
        self.gradient_weights_tensor_II = np.zeros((self.output_size, self.hidden_size + 1))
        self.gradient_weights_tensor_I = np.zeros((self.hidden_size, self.input_size + self.hidden_size + 1))
        self.tanh_err_tensor = np.zeros((error_tensor.shape[0], self.hidden_size))
        self.hidden_error_tensor = np.zeros((error_tensor.shape[0], self.hidden_size))
        for batch in range(error_tensor.shape[0]):
            self.sigmoid_err_tensor[batch] = self.sigmoid.backward(error_tensor[batch])#1,5
            if batch != 0:
                self.previous_error_tensor_II[batch] = np.dot(self.sigmoid_err_tensor[batch], self.fully_connected_II.weights[:,0:self.fully_connected_II.weights.shape[1]-1]) + self.hidden_error_tensor[batch-1] #1,7
            else:
                self.previous_error_tensor_II[batch] = np.dot(self.sigmoid_err_tensor[batch],self.fully_connected_II.weights[:,0:self.fully_connected_II.weights.shape[1] - 1]) # 1,7
            self.tanh_err_tensor[batch] = self.tanh.backward(self.previous_error_tensor_II[batch]) #1,7
            self.previous_error_tensor_I[batch] = np.dot(self.tanh_err_tensor[batch], self.fully_connected_I.weights[:,0:self.fully_connected_I.weights.shape[1]-1]) #1,20
            self.hidden_error_tensor[batch] = self.previous_error_tensor_I[batch,self.input_size:self.input_size + self.hidden_size]
        self.previous_error_tensor = self.previous_error_tensor_I[:,0:self.input_size]
        self.gradient_weights_tensor_II = np.dot(self.sigmoid_err_tensor.T, np.concatenate((self.batch_hidden_state, np.ones([self.batch_hidden_state.shape[0], 1])), axis=1))  # 5,8
        self.gradient_weights_tensor_I = np.dot(self.tanh_err_tensor.T, np.concatenate((self.fc_input_tensor, np.ones([self.fc_input_tensor.shape[0], 1])), axis=1))  # 7,21
        return self.previous_error_tensor
    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.input_size+self.hidden_size
        fan_out = self.hidden_size
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)