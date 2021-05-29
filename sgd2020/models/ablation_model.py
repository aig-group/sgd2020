# Identical copies of two AlexNet models
import torch
import torch.nn as nn
import copy 

__all__ = ['mlp', 'cnn']

class FullyConnected(nn.Module):

    def __init__(self, input_dim=32*32 , width=256, depth=3, num_classes=10, relu=False, dropout=False, bn=False):
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.relu = relu
        self.dropout = dropout
        self.bn = bn
        layers = self.get_layers()

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.width, self.num_classes, bias=False)

    def get_layers(self):
        layers = []
        layers.append(nn.Linear(self.input_dim, self.width, bias=False))
        if self.bn:
            layers.append(nn.BatchNorm1d(self.width))
        if self.relu:
            layers.append(nn.ReLU(inplace=True))
        if self.dropout:
            layers.append(nn.Dropout())
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            if self.bn:
                layers.append(nn.BatchNorm1d(self.width))
            if self.relu:
                layers.append(nn.ReLU(inplace=True))
            if self.dropout:
                layers.append(nn.Dropout())
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.features(x)
        return self.classifier(x)


# This is a copy from online repositories 
class AlexNet(nn.Module):

    def __init__(self, input_height=32, input_width=32, input_channels=3, ch=64, num_classes=10, relu=False, dropout=False, bn=False):
        # ch is the scale factor for number of channels
        super(AlexNet, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.relu = relu
        self.dropout = dropout
        self.bn = bn
        layers = []
        layers.append(nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2))
        if self.bn:
            layers.append(nn.BatchNorm2d(ch))
        if self.relu:
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(nn.Conv2d(ch, ch, kernel_size=5, padding=2))
        if self.bn:
            layers.append(nn.BatchNorm2d(ch))
        if self.relu:
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1))
        if self.bn:
            layers.append(nn.BatchNorm2d(ch))
        if self.relu:
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1))
        if self.bn:
            layers.append(nn.BatchNorm2d(ch))
        if self.relu:
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1))
        if self.bn:
            layers.append(nn.BatchNorm2d(ch))
        if self.relu:
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        self.features = nn.Sequential(*layers)

        # self.features = nn.Sequential(
        #     nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2),
        #     nn.BatchNorm2d(ch),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(ch, ch, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(ch),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(ch, ch, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(ch),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(ch, ch, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(ch),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(ch, ch, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(ch),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        self.size = self.get_size()
        # print(self.size)
        a = torch.tensor(self.size).float()
        b = torch.tensor(2).float()
        self.width = int(a) * int(1 + torch.log(a) / torch.log(b))

        layers = []
        if self.dropout:
            layers.append(nn.Dropout())
        layers.append(nn.Linear(self.size, self.width))
        if self.relu:
            layers.append(nn.ReLU(inplace=True))
        if self.dropout:
            layers.append(nn.Dropout())
        layers.append(nn.Linear(self.width, self.width))
        if self.relu:
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(self.width, num_classes))
        self.classifier = nn.Sequential(*layers)

        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(self.size, self.width),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(self.width, self.width),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.width, num_classes),
        # )

    def get_size(self):
        # hack to get the size for the FC layer...
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        # print(y.size())
        return y.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def cnn(**kwargs):
    return AlexNet(**kwargs)


def mlp(**kwargs):
    return FullyConnected(**kwargs)
