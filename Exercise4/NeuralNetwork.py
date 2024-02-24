import copy
from Layers.Base import BaseLayer
class NeuralNetwork(BaseLayer):

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.output_tensor = None
        self.error_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = False
        self.regularization_loss = 0

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        self.output_tensor = input_tensor
        for i in self.layers:
            self.output_tensor = i.forward(self.output_tensor)
            if i.trainable:
                if self.optimizer.regularizer is not None:
                    if i.weights is not None:
                        self.regularization_loss += self.optimizer.regularizer.norm(i.weights)
        loss = self.loss_layer.forward(self.output_tensor, self.label_tensor) + self.regularization_loss
        #loss = self.loss_layer.forward(self.output_tensor, self.label_tensor)
        return loss

    def backward(self):
        self.error_tensor = self.label_tensor
        self.error_tensor = self.loss_layer.backward(self.error_tensor)#last layer
        for i in reversed(self.layers):
            self.error_tensor = i.backward(self.error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            opt = copy.deepcopy(self.optimizer)
            layer.optimizer = opt
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase
        for i in self.layers:
            i.testing_phase = phase

    def train(self, iterations):
        self.phase = False
        self.testing_phase = False
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()


    def test(self, input_tensor):
        self.output_tensor = input_tensor
        self.phase = True
        self.testing_phase = True
        for i in self.layers:
            self.output_tensor = i.forward(self.output_tensor)
        return self.output_tensor
