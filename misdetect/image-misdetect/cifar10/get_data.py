import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# clean数据占比
good_sample_ratio = 0.8


class MyDataSet(Dataset):
    def __init__(self, loaded_data):
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], index


transform_train = torchvision.transforms.Compose([  # Compose将这些变换按照顺序连接起来
    # 将图片放大成高和宽各为 40 像素的正方形。
    torchvision.transforms.Resize(40),
    # 随机对高和宽各为 40 像素的正方形图片裁剪出面积为原图片面积 0.64 到 1 倍之间的小正方
    # 形，再放缩为高和宽各为 32 像素的正方形。
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                             ratio=(1.0, 1.0)),
    # 将图片像素值按比例缩小到 0 和 1 之间，并将数据格式从“高 * 宽 * 通道”改为“通道 * 高 * 宽”。
    torchvision.transforms.ToTensor(),
    # 对图片的每个通道做标准化。
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])
])

# 测试时，无需对图像做标准化以外的增强数据处理。
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])
])

train_set_tmp = torchvision.datasets.CIFAR10(root="data", train=True, transform=transform_train, download=True)
test_set = torchvision.datasets.CIFAR10(root="data", train=False, transform=transform_test, download=True)

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
# train_clean_bad_set = []
# for i in range(len(train_clean_bad_outlier)):
#     train_clean_bad_set.append(train_clean_bad_outlier[i])

# import torch
# # Load the pre-trained ResNet-18 model
# feature_embedding_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
#
# # Set the model to evaluation mode
# feature_embedding_model.eval()
# trainloader = torch.utils.data.DataLoader(train_clean_bad_outlier, batch_size=128, shuffle=False)
#
# # Generate feature embeddings for all test images
# embeddings = []
# with torch.no_grad():
#     for data in trainloader:
#         images, _ = data
#         features = feature_embedding_model(images)
#         embeddings.append(features.view(features.size(0), -1))
# embeddings = torch.cat(embeddings, dim=0)
#
# # Calculate the distance between each test image embedding and the mean embedding
# mean_embedding = torch.mean(embeddings, dim=0)
# distances = torch.norm(embeddings - mean_embedding, dim=1)
#
# # Identify potential outliers as images with high distance from the mean embedding
# threshold = np.percentile(distances.numpy(), 99.5)
# outlier_index = torch.where(distances > threshold)[0]
#
# # Print the number of potential outliers and their indices
# print("Number of potential outliers:", len(outlier_index))
# print("Indices of potential outliers:", outlier_index)

outlier_verify = []
# bad_outlier = 0
# clean_outlier = 0
# for i in range(len(outlier_index)):
#     outlier_verify.append(train_clean_bad_outlier[i])
#     if len(train_clean_bad_outlier) > i >= len(train_clean_dataset):
#         bad_outlier += 1
#     else:
#         clean_outlier += 1
# print("bad_outlier:{}, clean_outlier:{}".format(bad_outlier, clean_outlier))
#
train_clean_bad_set = []
for i in range(len(train_clean_bad_outlier)):
    # if i not in outlier_index:
    train_clean_bad_set.append(train_clean_bad_outlier[i])
print(len(train_clean_bad_set))
# print(train_clean_bad_set[0])

# tmp_pca = train_clean_bad_set[0][0].view(3072).tolist()
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
