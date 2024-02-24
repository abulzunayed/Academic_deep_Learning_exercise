import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super(Pooling, self).__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.prev_error_tensor = None
        self.indices_list = []

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        row = int(np.ceil((self.input_tensor.shape[2]-self.pooling_shape[0]+1)/self.stride_shape[0]))
        col = int(np.ceil((self.input_tensor.shape[3] - self.pooling_shape[1] + 1) / self.stride_shape[1]))
        self.output_tensor = np.zeros([self.input_tensor.shape[0], self.input_tensor.shape[1], row, col])
        self.indices_list = []
        for batch in range(self.input_tensor.shape[0]):
            for channel in range(self.input_tensor.shape[1]):
                select = self.input_tensor[batch, channel]
                for i in range(row):
                    for j in range(col):
                        output = select[i * self.stride_shape[0]:i * self.stride_shape[0] + self.pooling_shape[0], j * self.stride_shape[1]:j * self.stride_shape[1] + self.pooling_shape[1]]
                        indices = np.argwhere(output == output.max())
                        stored_indices = np.zeros([1, 4], dtype=int)
                        stored_indices[0,0] = batch
                        stored_indices[0,1] = channel
                        stored_indices[0,2] = indices[0,0] + i * (self.stride_shape[0])
                        stored_indices[0, 3] = indices[0, 1] + j * (self.stride_shape[1])
                        self.indices_list.append(stored_indices[0])
                        self.output_tensor[batch, channel, i, j] = output[indices[0,0], indices[0,1]]

        return self.output_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor.reshape(-1,1)
        self.prev_error_tensor = np.zeros_like(self.input_tensor)
        for i in range(len(self.indices_list)):
            self.prev_error_tensor[self.indices_list[i][0], self.indices_list[i][1], self.indices_list[i][2], self.indices_list[i][3]] += self.error_tensor[i]

        return self.prev_error_tensor