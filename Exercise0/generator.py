import os.path
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import json

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        #TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.images = {}
        self.labels = {}
        self.next_call = 0
        self.count=0
        for filename in os.listdir(self.file_path):
            self.images[filename] = np.load(os.path.join(self.file_path, filename))
            self.images[filename] = np.resize(self.images[filename], self.image_size)
            self.count += 1
        with open(self.label_path) as json_file:
            self.labels = json.load(json_file)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        images = np.ndarray((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))
        labels = np.ndarray((self.batch_size, 1))
        shuffledimages = {}
        shuffledlabels = {}
        image_datasize = len(self.images)
        if (image_datasize % self.batch_size) != 0:
            no_images_tobe_added = self.batch_size - (image_datasize % self.batch_size)
            for filename in os.listdir(self.file_path):
                imagepath_added = str(self.count)+".npy"
                self.images[imagepath_added] = np.load(os.path.join(self.file_path, filename))
                self.images[imagepath_added] = np.resize(self.images[imagepath_added], self.image_size)
                self.labels[str(self.count)] = self.labels[filename.replace(".npy","")]
                self.count += 1
                if self.count == (image_datasize + no_images_tobe_added):
                    break

        if self.mirroring:
            for i in self.images:
                self.images[i] = self.augment(self.images[i])

        elif self.rotation:
            for i in self.images:
                self.images[i] = self.augment(self.images[i])

        if self.shuffle:
            images_list = list(self.images.keys())
            np.random.shuffle(images_list)

            for i in images_list:
                shuffledimages[i] = self.images[i]
                shuffledlabels[i.replace(".npy","")] = self.labels[i.replace(".npy","")]
            idx = 0
            for i in list(shuffledimages.keys())[self.next_call*self.batch_size:(self.next_call+1)*self.batch_size]:
                images[idx] = shuffledimages[i]
                labels[idx] = shuffledlabels[i.replace(".npy","")]
                idx += 1
        else:
            idx = 0
            for i in list(self.images.keys())[self.next_call*self.batch_size:(self.next_call+1)*self.batch_size]:
                images[idx] = self.images[i]
                labels[idx] = self.labels[i.replace(".npy","")]
                idx += 1
        self.next_call += 1
        return images.astype(int), labels.astype(int)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        if self.mirroring:
            mirror_choice = np.random.randint(0, 3)
            if mirror_choice == 0:
                img = np.fliplr(img)
            elif mirror_choice == 1:
                img = np.flipud(img)
            else:
                img = np.flipud(np.fliplr(img))

        elif self.rotation:
            img = np.rot90(img, np.random.randint(1, 4))
        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function

        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        for i in range(2, 10):
            if self.batch_size % i == 0:
                height = i
                width = int(self.batch_size/i)
                break
        fig, ax = plt.subplots(height, width)
        ax = ax.reshape(height, width)
        plot_count=0
        for i in range(height):
            for j in range(width):
                ax[i, j].imshow(images[plot_count].astype(int))
                ax[i, j].set_title(self.class_name(int(labels[plot_count])))
                plot_count += 1
        plt.show()