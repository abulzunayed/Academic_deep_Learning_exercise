import numpy as np
import matplotlib.pyplot as plt
import copy


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):

        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        circle = np.sqrt((xx - self.position[0])**2 + (yy - self.position[1])**2)
        self.output = np.zeros([self.resolution, self.resolution])
        self.output[circle<=self.radius] = 1
        output = copy.deepcopy(self.output)
        return output


    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        if (self.resolution % (2 * self.tile_size)) == 0:
            white_tile = np.ones([self.tile_size, self.tile_size])
            black_tile = np.zeros([self.tile_size, self.tile_size])
            first_line_block = np.concatenate((black_tile, white_tile), axis=1)
            second_line_block = np.concatenate((white_tile, black_tile), axis=1)
            no_repetitions = int(self.resolution / (2 * self.tile_size))
            first_line = np.tile(first_line_block, no_repetitions)
            second_line = np.tile(second_line_block, no_repetitions)
            block = np.concatenate((first_line, second_line), axis=0)
            self.output = np.tile(block, (no_repetitions,1))
            output = copy.deepcopy(self.output)
            return output
        else:
            raise Exception("resolution should be integral multiple of tile size")

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.ndarray((self.resolution, self.resolution, 3))

    def draw(self):
        r_line = np.linspace(0,1,self.resolution)
        r_channel = np.tile(r_line, (self.resolution,1))
        g_line = np.linspace(0, 1, self.resolution)
        g_line = g_line.reshape(self.resolution,1)
        g_channel = np.tile(g_line, self.resolution)
        b_line = np.linspace(1, 0, self.resolution)
        b_channel = np.tile(b_line, (self.resolution, 1))

        self.output[:, :, 0] = r_channel
        self.output[:, :, 1] = g_channel
        self.output[:, :, 2] = b_channel

        output = copy.deepcopy(self.output)
        return output

    def show(self):
        plt.imshow(self.output)
        plt.show()
