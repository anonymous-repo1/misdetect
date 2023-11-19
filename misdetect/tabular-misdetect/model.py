import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from collections import OrderedDict
import numpy as np
import pandas
from get_data_tabular import columns,classes
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, loaded_data):
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

# k = 768
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.net1 = Sequential(OrderedDict([
#             ('batch norm1', nn.BatchNorm1d(len(columns) - 1, momentum=0.99)),
#             ('linear1', nn.Linear(len(columns) - 1, k)),
#             ('relu1', nn.ReLU())
#         ]))
#         self.net2 = Sequential(OrderedDict([
#             ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
#             ('linear2', nn.Linear(k, k)),
#             ('relu2', nn.ReLU())
#         ]))
#         self.net3 = Sequential(OrderedDict([
#             ('batch norm3', nn.BatchNorm1d(k, momentum=0.99)),
#             ('linear3', nn.Linear(k, 2)),
#             ('sigmoid', nn.Sigmoid())
#         ]))
#
#     def forward(self, x):
#         a = self.net1(x)
#         b = self.net2(a) + a
#         c = self.net3(b)
#         # c = self.net3(a)
#         return c
k = 320
k2 = 384

class Model(nn.Module):
    def __init__(self, input_len):
        super(Model, self).__init__()
        self.net1 = Sequential(OrderedDict([
            ('batch norm1', nn.BatchNorm1d(input_len, momentum=0.99)),
            ('linear1', nn.Linear(input_len, k)),
            ('relu1', nn.ReLU())
        ]))
        self.net2 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear2', nn.Linear(k, k)),
            ('relu2', nn.ReLU())
        ]))
        self.net10 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear2', nn.Linear(k, k2)),
            ('relu2', nn.ReLU())
        ]))
        self.net11 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k2, momentum=0.99)),
            ('linear2', nn.Linear(k2, k2)),
            ('relu2', nn.ReLU())
        ]))
        self.net12 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k2, momentum=0.99)),
            ('linear2', nn.Linear(k2, k)),
            ('relu2', nn.ReLU())
        ]))
        self.net13 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear2', nn.Linear(k, k)),
            ('relu2', nn.ReLU())
        ]))
        self.net3 = Sequential(OrderedDict([
            ('batch norm3', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear3', nn.Linear(k, 7)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        a = self.net1(x)
        b = self.net2(a) + a
        h1 = self.net10(b)
        h2 = self.net11(h1)
        h3 = self.net12(h2)
        h4 = self.net13(h3)
        c = self.net3(h4)
        c = c.squeeze(-1)
        return c

kk = 768

class Model_retrieval(nn.Module):
    def __init__(self, input_len):
        super(Model_retrieval, self).__init__()
        self.net1 = Sequential(OrderedDict([
            ('batch norm1', nn.BatchNorm1d(input_len, momentum=0.99)),
            ('linear1', nn.Linear(input_len, kk)),
            ('relu1', nn.ReLU())
        ]))
        self.net2 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(kk, momentum=0.99)),
            ('linear2', nn.Linear(kk, kk)),
            ('relu2', nn.ReLU())
        ]))
        self.net3 = Sequential(OrderedDict([
            ('batch norm3', nn.BatchNorm1d(kk, momentum=0.99)),
            ('linear3', nn.Linear(kk, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        a = self.net1(x)
        b = self.net2(a) + a
        c = self.net3(b)
        return c


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, 3, padding=2), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
#             nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
#             nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(2048, 4096), nn.ReLU(), nn.Dropout(0.5),
#             nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
#             nn.Linear(4096, 100)
#         )
#
#     def forward(self, x):
#         x = self.feature(x)
#         output = self.classifier(x)
#         return output


# model = Model()
# check_input = torch.ones(7)
# check = model(check_input)
# print(check.shape)
