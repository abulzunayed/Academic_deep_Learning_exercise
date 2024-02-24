import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #self.drop = nn.Dropout2d()
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        y = self.conv1(input)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        #y = self.drop(y)
        y_in = self.conv3(input)
        y_in = self.bn3(y_in)
        output = self.relu2(y + y_in)
        #output = self.relu2(y)
        return output


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.block1 = ResBlock(in_channels=64, out_channels=64, stride=1)
        self.block2 = ResBlock(in_channels=64, out_channels=128, stride=2)
        self.block3 = ResBlock(in_channels=128, out_channels=256, stride=2)
        self.block4 = ResBlock(in_channels=256, out_channels=512, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=512, out_features=2)
        self.bn2 = nn.BatchNorm1d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        y = self.relu1(self.bn1(self.conv1(input)))
        y = self.mp1(y)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = F.avg_pool2d(y, kernel_size=10)
        y = self.flatten(y)
        y = self.fc(y)
        y = self.bn2(y)
        output = self.sigmoid(y)
        return output

