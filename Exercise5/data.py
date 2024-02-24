from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        super(ChallengeDataset, self).__init__()
        self.data = data
        self.mode = mode
        if self.mode == 'train':
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                     tv.transforms.ToTensor(),
                                                     tv.transforms.RandomHorizontalFlip(),
                                                     tv.transforms.Normalize(train_mean, train_std)])

        else:
            self._transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                                     tv.transforms.RandomVerticalFlip(),
                                                     tv.transforms.Normalize(train_mean, train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = imread(self.data.iloc[index]['filename'])
        image = gray2rgb(image)
        image = self._transform(image)
        label = self.data.iloc[index][1:]
        label = torch.from_numpy(np.array([label['crack'], label['inactive']]))
        return image, label