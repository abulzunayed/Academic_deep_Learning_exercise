#team id : id: ByspYqp0eZ

import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from torch.nn.modules import Module
from model import ResNet
import pandas as pd
from sklearn.model_selection import train_test_split

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

data = pd.read_csv('data.csv', sep=';')
train_data, val_data = train_test_split(data, test_size=0.30)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train_data_loader = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=64)
val_data_loader = t.utils.data.DataLoader(ChallengeDataset(val_data, 'val'), batch_size=64)

# create an instance of our ResNet model
# TODO
ResNet_model = ResNet()
if t.cuda.is_available():
    ResNet_model.cuda()
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
#loss = t.nn.BCEWithLogitsLoss(pos_weight=t.squeeze(ChallengeDataset(train_data, 'train').pos_weight()))
loss = t.nn.BCELoss()
optim = t.optim.Adam(ResNet_model.parameters(), lr=0.002, weight_decay=0.000008) #weight_decay=1e-6)
early_stopping_criterion=35
trainer = Trainer(ResNet_model, loss, optim, train_data_loader, val_data_loader, early_stopping_patience=early_stopping_criterion)
# go, go, go... call fit on trainer
res = trainer.fit(60)
onnx_file_path = "./Mymodel.onnx"
# save the model
with open(onnx_file_path, "wb+") as path:
    trainer.save_onnx(path)

# go, go, go... call fit on trainer
#TODO
# res = trainer.fit(100)
# epoch = 56

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
#plt.imshow('losses.png')