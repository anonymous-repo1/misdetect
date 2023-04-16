import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
# from torchvision import transforms

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# clean数据占比
good_sample_ratio = 0.9


class MyDataSet(Dataset):
    def __init__(self, loaded_data):
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], index


transform_train = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train_set_tmp = torchvision.datasets.FashionMNIST(root="data", train=True, transform=transform_train, download=True)
test_set = torchvision.datasets.FashionMNIST(root="data", train=False, transform=transform_test, download=True)

train_set = []
for i in train_set_tmp:
    train_set.append(list(i))

cnt_label = {}
for idx, tensor in enumerate(train_set):
    cnt_label[tensor[1]] = cnt_label.get(tensor[1], 0) + 1

cnt_good_label_tgt = {}
for k, v in cnt_label.items():
    cnt_good_label_tgt[k] = int(v * good_sample_ratio)

manipulate_label = {}
good_idx_set = []
for idx, tensor in enumerate(train_set):
    manipulate_label[tensor[1]] = manipulate_label.get(tensor[1], 0) + 1
    if manipulate_label[tensor[1]] > cnt_good_label_tgt[tensor[1]]:
        p = np.random.randint(0, len(cnt_label))
        while True:
            if p != tensor[1]:
                train_set[idx][1] = p
                break
            p = np.random.randint(0, 10)
    else:
        good_idx_set.append(idx)

good_idx_array = np.array(good_idx_set)
all_idx_array = np.arange(len(train_set))
bad_idx_array = np.setdiff1d(all_idx_array, good_idx_array)
train_clean_dataset = []
for i in good_idx_array:
    train_clean_dataset.append(train_set[i])
    if train_set[i][1] != train_set_tmp[i][1]:
        print("--------------------------------")
train_bad_dataset = []
for i in bad_idx_array:
    train_bad_dataset.append(train_set[i])
    if train_set[i][1] == train_set_tmp[i][1]:
        print("--------------------------------")
train_bad_dataset2 = []
for i in bad_idx_array:
    train_bad_dataset2.append(train_set_tmp[i])

train_clean_bad_set_ground_truth = train_clean_dataset + train_bad_dataset2
train_clean_bad_outlier = train_clean_dataset + train_bad_dataset
print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_outlier))

# =============================================================================================
# import torch.nn as nn
# import torch.optim as optim
#
# # Define the feature embedding model
# class FeatureEmbedding(nn.Module):
#     def __init__(self, input_size, embedding_size):
#         super(FeatureEmbedding, self).__init__()
#         self.fc1 = nn.Linear(input_size, embedding_size)
#         self.fc2 = nn.Linear(embedding_size, input_size)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# feature_embedding_model = FeatureEmbedding(input_size=784, embedding_size=32)
# trainloader = torch.utils.data.DataLoader(train_clean_bad_outlier, batch_size=128, shuffle=False)
# # Define the feature embedding model and optimizer
# model = FeatureEmbedding(input_size=784, embedding_size=32)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Train the feature embedding model on the MNIST training set
# for epoch in range(10):
#     for batch_idx, (data, target) in enumerate(trainloader):
#         data = data.view(-1, 784)
#         optimizer.zero_grad()
#         embedding = model(data)
#         loss = nn.MSELoss()(data, embedding)
#         loss.backward()
#         optimizer.step()
#
# # Evaluate the model on the MNIST test set and calculate the Mahalanobis distance for each sample
# mean_embedding = np.mean(model(torch.cat([data.view(-1, 784) for data, target in trainloader], axis=0)).detach().numpy(), axis=0)
#
# covariance_embedding = np.cov(model(torch.cat([data.view(-1, 784) for data, target in trainloader], axis=0)).detach().numpy().T)
# mahalanobis = np.zeros(mnist_test.data.shape[0])
# for i, (data, target) in enumerate(test_loader):
#     data = data.view(-1, 784)
#     embedding = model(data)
#     mahalanobis[i*64:i*64+len(data)] = np.sqrt(np.sum(np.power((embedding.detach().numpy() - mean_embedding), 2) / np.diag(covariance_embedding), axis=1))
#
# # Determine the threshold for outliers
# threshold = np.percentile(mahalanobis, 99)
#
# # Find the indices of the outliers
# outlier_indices = np.where(mahalanobis > threshold)
#
# # Print the indices of the outliers
# print("Outlier indices: ", outlier_indices)


# train_clean_bad_set = []
# for i in range(len(train_clean_bad_outlier)):
#     train_clean_bad_set.append(train_clean_bad_outlier[i])
# print(len(train_clean_bad_set))

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np

X_train = []
for i in train_clean_bad_outlier:
    X_train.append(i[0].numpy().flatten())
X_train = np.array(X_train)


# Reduce the dimensionality of the dataset using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Calculate the Mahalanobis distance for each sample
mean = np.mean(X_pca, axis=0)
covariance = np.cov(X_pca.T)
mahalanobis = np.zeros(X_pca.shape[0])
for i in range(X_pca.shape[0]):
    mahalanobis[i] = np.sqrt(np.dot(np.dot((X_pca[i]-mean), np.linalg.inv(covariance)), (X_pca[i]-mean).T))

# Determine the threshold for outliers
threshold = np.percentile(mahalanobis, 99)

# Find the indices of the outliers
outlier_index = np.where(mahalanobis > threshold)

outlier_index = outlier_index[0].tolist()

# Print the number of potential outliers and their indices
print("Number of potential outliers:", len(outlier_index))
print("Indices of potential outliers:", outlier_index)
print(type(outlier_index))

outlier_verify = []
bad_outlier = 0
clean_outlier = 0
for i in outlier_index:
    outlier_verify.append(train_clean_bad_outlier[i])
    if len(train_clean_bad_outlier) > i >= len(train_clean_dataset):
        bad_outlier += 1
    else:
        clean_outlier += 1
print("bad_outlier:{}, clean_outlier:{}".format(bad_outlier, clean_outlier))

train_clean_bad_set = []
for i in range(len(train_clean_bad_outlier)):
    if i not in outlier_index:
        train_clean_bad_set.append(train_clean_bad_outlier[i])
print(len(train_clean_bad_set))
# print(train_clean_bad_set[0])

# tmp_pca = train_clean_bad_set[0][0].view(784).tolist()
# tmp_pca.append(train_clean_bad_set[0][1])
# tmp_pca = [torch.tensor(tmp_pca), 2]
# tmp_pca.append(0)
# =============================================================================================

# print(type(train_clean_bad_set[0][0]), type(train_clean_bad_set[0][1]), type(train_clean_bad_set[0]))

# 随机制造脏数据，而不是每个类取固定比例制造脏数据，得到的结果是，两者early loss准确率差不多，均为0.73左右
# 但flip label的准确率下降了，为0.26左右
# 同样，如果把训练的batch size改成1，则early loss=0.45 and flip label = 0.10
# train_clean_size = int(good_sample_ratio * len(train_set_tmp))
# train_bad_size = len(train_set_tmp) - train_clean_size
# train_clean_set, train_bad_set = torch.utils.data.random_split(train_set_tmp, [train_clean_size, train_bad_size])
#
# train_set = []
# for i in train_set_tmp:
#     train_set.append(list(i))
#
# train_clean_dataset = []
# train_bad_dataset = []
# for i in train_bad_set:
#     train_bad_dataset.append(list(i))
#
# for i in train_clean_set:
#     train_clean_dataset.append(list(i))
#
# for i in train_bad_dataset:
#     p = np.random.randint(0, len(classes))
#     while True:
#         if p != i[1]:
#             i[1] = p
#             break
#         p = np.random.randint(0, 10)
#
# # train_clean_dataset = train_clean_dataset[:9000]
# # train_bad_dataset = train_bad_dataset[:6000]
#
# train_clean_bad_set = train_clean_dataset + train_bad_dataset
# print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_set))
