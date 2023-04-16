import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from collections import OrderedDict

#定义卷积神经网络
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, 3, 1, 1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, 3, 1, 1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2, 2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10),
                                         nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


class PNN(nn.Module):
    def __init__(self, feature_size):
        super(PNN, self).__init__()

        self.embedding = nn.Linear(feature_size, feature_size)
        self.interaction = nn.Linear(feature_size, feature_size, bias=False)
        self.product = nn.Linear(feature_size, 1, bias=False)

    def forward(self, x, y):
        x = self.embedding(x)
        #y = self.embedding(y)

        xy_interaction = self.interaction(x * y)
        xy_product = self.product(x * y)

        z = xy_interaction + xy_product

        return z

kk = 768

class Model_retrieval(nn.Module):
    def __init__(self):
        super(Model_retrieval, self).__init__()
        self.net1 = Sequential(OrderedDict([
            ('batch norm1', nn.BatchNorm1d(384, momentum=0.99)),
            ('linear1', nn.Linear(384, kk)),
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

class ClassifierModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 10),
            # nn.Softmax(dim=-1),
        )

    def forward(self, X):
        X = self.cnn(X)
        X = self.out(X)
        return X

model = Model()
c = ClassifierModule()
check_input = torch.ones((64, 1, 28, 28))
check = model(check_input)
print(check.shape)
