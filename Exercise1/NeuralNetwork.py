import copy
class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.output_tensor = None
        self.error_tensor = None
    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        self.output_tensor = input_tensor
        for i in self.layers:
            self.output_tensor = i.forward(self.output_tensor)
        loss = self.loss_layer.forward(self.output_tensor, self.label_tensor)
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
        self.layers.append(layer)


    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()


    def test(self, input_tensor):
        self.output_tensor = input_tensor
        for i in self.layers:
            self.output_tensor = i.forward(self.output_tensor)

        return self.output_tensor
