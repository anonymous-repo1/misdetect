import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from collections import OrderedDict

k = 64
k2 = 128
k3 = 2048

def call_bn(bn, x):
    return bn(x)

class Model(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(Model, self).__init__()
        self.c1=nn.Conv2d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)
        self.l_c1=nn.Linear(128,n_outputs)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)
        self.bn7=nn.BatchNorm2d(512)
        self.bn8=nn.BatchNorm2d(256)
        self.bn9=nn.BatchNorm2d(128)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h=self.c9(h)
        h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit=self.l_c1(h)
        if self.top_bn:
            logit=call_bn(self.bn_c1, logit)
        return logit

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.net1 = Sequential(
#             Conv2d(3, k, 5, padding=2),
#             MaxPool2d(2)
#         )
#         self.net2 = Sequential(
#             Conv2d(k, k, 5, padding=2),
#             MaxPool2d(2)
#         )
#         self.net3 = Sequential(
#             Conv2d(k, k2, 5, padding=2),
#             MaxPool2d(2),
#             Flatten(),
#             Linear(k3, k2),
#             Linear(k2, 10)
#         )
#
#     def forward(self, x):
#         a = self.net1(x)
#         b = self.net2(a)
#         c = self.net3(b)
#         return c


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        accu = accuracy(out, labels)
        return loss, accu

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'Loss': loss.detach(), 'Accuracy': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['Loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['Accuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'Loss': epoch_loss.item(), 'Accuracy': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch :", epoch + 1)
        print(
            f'Train Accuracy:{result["train_accuracy"] * 100:.2f}% Validation Accuracy:{result["Accuracy"] * 100:.2f}%')
        print(f'Train Loss:{result["train_loss"]:.4f} Validation Loss:{result["Loss"]:.4f}')


# class Model(ImageClassificationBase):
#     def __init__(self):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
#             nn.BatchNorm2d(64),
#
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
#             nn.BatchNorm2d(128),
#
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
#             nn.BatchNorm2d(256),
#
#             nn.Flatten(),
#             nn.Linear(256 * 4 * 4, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#             # nn.Softmax(dim=-1)
#             )
#
#     def forward(self, xb):
#         return self.network(xb)

class AttentionCNN(nn.Module):
    def __init__(self):
        super(AttentionCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = nn.functional.relu(x)

        # Self-attention mechanism
        a = self.attention(x.view(-1, 256))
        a = torch.softmax(a, dim=0)
        a = a.view(-1, 1, 8, 8)
        x = x * a

        x = x.view(-1, 256)
        x = self.fc(x)

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

model = Model()
check_input = torch.ones((64, 3, 32, 32))
check = model(check_input)
print(check.shape)
