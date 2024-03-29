class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.weight_tensor = weight_tensor - self.learning_rate*gradient_tensor
        return self.weight_tensor
