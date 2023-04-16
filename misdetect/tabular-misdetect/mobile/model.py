import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from collections import OrderedDict
import numpy as np
import pandas
import torch.nn.functional as F



k = 320
k2 = 1024

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net1 = Sequential(OrderedDict([
            ('batch norm1', nn.BatchNorm1d(20, momentum=0.99)),
            ('linear1', nn.Linear(20, k)),
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
            ('linear3', nn.Linear(k, 4)),
            ('sigmoid', nn.Sigmoid(),
             # nn.Softmax(dim=-1)
             )
        ]))

    def forward(self, x):
        a = self.net1(x)
        b = self.net2(a) + a
        h1 = self.net10(b)
        h2 = self.net11(h1)
        h3 = self.net12(h2)
        # h4 = self.net11(h3)
        c = self.net3(h3)
        return c


kk = 768

class Model_retrieval(nn.Module):
    def __init__(self):
        super(Model_retrieval, self).__init__()
        self.net1 = Sequential(OrderedDict([
            ('batch norm1', nn.BatchNorm1d(63, momentum=0.99)),
            ('linear1', nn.Linear(63, kk)),
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
#         self.net1 = Sequential(OrderedDict([
#             ('batch norm1', nn.BatchNorm1d(20, momentum=0.99)),
#             ('linear1', nn.Linear(20, k)),
#             ('relu1', nn.ReLU())
#         ]))
#         self.net2 = Sequential(OrderedDict([
#             ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
#             ('linear2', nn.Linear(k, k)),
#             ('relu2', nn.ReLU())
#         ]))
#         self.net3 = Sequential(OrderedDict([
#             ('batch norm3', nn.BatchNorm1d(k, momentum=0.99)),
#             ('linear3', nn.Linear(k, 4)),
#             ('sigmoid', nn.Sigmoid())
#         ]))
#
#     def forward(self, x):
#         a = self.net1(x)
#         b = self.net2(a) + a
#         c = self.net3(b)
#         return c

def normalize(dataframe, column):
    for i in dataframe:
        dic = {}
        num = 0
        if isinstance(dataframe.loc[0, i], str):
            for j in dataframe.index:
                if dataframe.loc[j, i] not in dic:
                    dic[dataframe.loc[j, i]] = num
                    num += 1
                dataframe.loc[j, i] = dic[dataframe.loc[j, i]]
            dataframe[i] = pandas.to_numeric(dataframe[i])
    for i in dataframe:
        if i != column:
            dataframe[i] = (dataframe[i] - dataframe[i][np.isfinite(dataframe[i])].mean()) / \
                           dataframe[i][np.isfinite(dataframe[i])].std()

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


# model = Model(7)
# check_input = torch.ones(7,1)
# check = model(check_input)
# print(check.shape)
