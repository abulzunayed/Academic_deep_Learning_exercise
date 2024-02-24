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
        self._gradient_weights = None
        self.regularization_loss = 0
        self.fcI_weights = np.zeros((self.hidden_size, self.input_size+self.hidden_size+1))


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

        if self.memorize is False:
            self.hidden_state = np.zeros((1, self.hidden_size))

        for batch in range(input_tensor.shape[0]):
            self.fc_input_tensor[batch] = np.concatenate((input_tensor[batch].reshape(1,-1), self.hidden_state), axis=1)#1,20
            self.fc_tanh[batch] = self.fully_connected_I.forward(self.fc_input_tensor[batch].reshape(1,-1))
            self.hidden_state = self.tanh.forward(self.fc_tanh[batch].reshape(1,-1))
            self.batch_hidden_state[batch] = self.hidden_state
            self.fc_sigmoid[batch] = self.fully_connected_II.forward(self.hidden_state)
            self.output_tensor[batch] = self.sigmoid.forward(self.fc_sigmoid[batch].reshape(1,-1))
        return self.output_tensor

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, weights):
        self._gradient_weights = weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.optflag = True

    def calculate_regularization_loss(self):
        self.regularization_loss = self.fully_connected_I.optimizer.regularizer.norm(self.fully_connected_I.weights) + self.fully_connected_II.optimizer.regularizer.norm(self.fully_connected_II.weights)
        return self.regularization_loss

    @property
    def weights(self):
        #self.fc1_weights = self.fully_connected_I.weights
        return self.fully_connected_I.weights

    @weights.setter
    def weights(self, weights):
        self.fully_connected_I.weights = weights

    def backward(self, error_tensor):
        self.previous_error_tensor = np.zeros((error_tensor.shape[0], self.input_size))
        self.gradient_weights_tensor_II = np.zeros((self.output_size, self.hidden_size + 1))
        self.gradient_weights_tensor_I = np.zeros((self.hidden_size, self.input_size + self.hidden_size + 1))
        hidden_gradient_tensor = np.zeros((1, self.hidden_size))
        for batch in reversed(range(error_tensor.shape[0])):

            self.sigmoid.output_tensor = self.output_tensor[batch]
            sigmoid_err_tensor = self.sigmoid.backward(error_tensor[batch]).reshape(1,-1)#1.5
            self.fully_connected_II.input_tensor = np.concatenate((self.batch_hidden_state[batch].reshape(1,-1), np.ones([1,1])), axis=1)#1,8
            previous_err_tensor_II = self.fully_connected_II.backward(sigmoid_err_tensor) + hidden_gradient_tensor#1,7
            self.gradient_weights_tensor_II += self.fully_connected_II.gradient_weights#5,8
            self.tanh.output_tensor = self.batch_hidden_state[batch].reshape(1,-1)#1,7
            tanh_err_tensor = self.tanh.backward(previous_err_tensor_II)#1,7
            self.fully_connected_I.input_tensor = np.concatenate((self.fc_input_tensor[batch].reshape(1,-1), np.ones([1,1])), axis=1)#1,21
            previous_err_tensor_I = self.fully_connected_I.backward(tanh_err_tensor)#1,20
            self.gradient_weights_tensor_I += self.fully_connected_I.gradient_tensor#7,21
            self.previous_error_tensor[batch] = previous_err_tensor_I[:,:self.input_size]
            hidden_gradient_tensor = previous_err_tensor_I[:,self.input_size:]

        self._gradient_weights = self.gradient_weights_tensor_I
        if self.optflag:
            self.fully_connected_I.weights = self.optimizer.calculate_update(self.fully_connected_I.weights, self.gradient_weights_tensor_I)
            self.fully_connected_II.weights = self.optimizer.calculate_update(self.fully_connected_II.weights, self.gradient_weights_tensor_II)

        return self.previous_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.fully_connected_I.initialize(weights_initializer, bias_initializer)
        self.fully_connected_II.initialize(weights_initializer, bias_initializer)



