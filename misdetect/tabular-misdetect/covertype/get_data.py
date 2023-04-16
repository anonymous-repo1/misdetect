import torch
from model import normalize
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import copy


columns = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
    'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
    'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
    'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
    'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
    'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',
    'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type'
]

classes = ['0', '1', '2', '3', '4', '5', '6']

label_index = columns.index('Cover_Type')


# clean数据占比
good_sample_ratio = 1.0

train_set_before = pd.read_csv('data/covtype_normalize.csv')


# 首先将pandas读取的数据转化为array
train_set_before = np.array(train_set_before)

# 把label改成从0开始
for i in train_set_before:
    i[label_index] -= 1

label = np.array(train_set_before).T[label_index]
train_set_before = np.delete(train_set_before, label_index, axis=1)

# 将train_set转化为特定list形式
train_set = []
for i in range(len(train_set_before)):
    tmp = [torch.tensor(train_set_before[i].tolist()), int(label[i])]
    train_set.append(tmp)

train_set_tmp = copy.deepcopy(train_set)

class MyDataSet(Dataset):
    def __init__(self, loaded_data):
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

test_set = copy.deepcopy(train_set_tmp[int(0.8*len(train_set_tmp)):])
train_set_tmp = copy.deepcopy(train_set_tmp[:int(0.8*len(train_set_tmp))])
# cnt_label = {}
# for idx, tensor in enumerate(train_set):
#     cnt_label[tensor[1]] = cnt_label.get(tensor[1], 0) + 1
# print(len(cnt_label))
#
# cnt_good_label_tgt = {}
# for k, v in cnt_label.items():
#     cnt_good_label_tgt[k] = int(v * good_sample_ratio)
#
# manipulate_label = {}
# good_idx_set = []
# for idx, tensor in enumerate(train_set):
#     manipulate_label[tensor[1]] = manipulate_label.get(tensor[1], 0) + 1
#     if manipulate_label[tensor[1]] > cnt_good_label_tgt[tensor[1]]:
#         # train_set[idx][1] = (train_set[idx][1] + 2) % len(cnt_label)
#         p = np.random.randint(0, len(cnt_label))
#         while True:
#             if p != tensor[1]:
#                 train_set[idx][1] = p
#                 break
#             p = np.random.randint(0, len(cnt_label))
#     else:
#         good_idx_set.append(idx)
#
# good_idx_array = np.array(good_idx_set)
# all_idx_array = np.arange(len(train_set))
# bad_idx_array = np.setdiff1d(all_idx_array, good_idx_array)
# train_clean_dataset = []
# for i in good_idx_array:
#     train_clean_dataset.append(train_set[i])
#     if train_set[i][1] != train_set_tmp[i][1]:
#         print("--------------------------------")
# train_bad_dataset = []
# for i in bad_idx_array:
#     train_bad_dataset.append(train_set[i])
#     if train_set[i][1] == train_set_tmp[i][1]:
#         print("--------------------------------")
# train_bad_dataset2 = []
# for i in bad_idx_array:
#     train_bad_dataset2.append(train_set_tmp[i])
#
# train_clean_bad_set = train_clean_dataset + train_bad_dataset
# train_clean_bad_set_outlier = train_clean_dataset + train_bad_dataset2
# print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_set))

# 随机制造脏数据，而不是每个类取固定比例制造脏数据，得到的结果是，两者early loss准确率差不多，均为0.73左右
# 但flip label的准确率下降了，为0.26左右
# 同样，如果把训练的batch size改成1，则early loss=0.45 and flip label = 0.10
train_clean_size = int(good_sample_ratio * len(train_set_tmp))
train_bad_size = len(train_set_tmp) - train_clean_size
train_clean_set, train_bad_set = torch.utils.data.random_split(train_set_tmp, [train_clean_size, train_bad_size])

train_set = []
for i in train_set_tmp:
    train_set.append(list(i))

train_clean_dataset = []
train_bad_dataset = []
for i in train_bad_set:
    train_bad_dataset.append(list(i))

for i in train_clean_set:
    train_clean_dataset.append(list(i))

train_clean_bad_set_ground_truth = train_clean_dataset + train_bad_dataset

for i in train_bad_dataset:
    p = np.random.randint(0, len(classes))
    while True:
        if p != i[1]:
            i[1] = p
            break
        p = np.random.randint(0, len(classes))

# train_clean_bad_set = train_clean_dataset + train_bad_dataset
train_clean_bad_outlier = train_clean_dataset + train_bad_dataset
print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_outlier))

# =============================================================================================
# train_clean_bad_set = []
# for i in range(len(train_clean_bad_outlier)):
#     train_clean_bad_set.append(train_clean_bad_outlier[i])

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
#
# X_train = []
# for i in train_clean_bad_outlier:
#     X_train.append(i[0].numpy())
# X_train = np.array(X_train)
#
# # print(X_train.shape)
#
# sc = StandardScaler()
# X_train_std = sc.fit_transform(X_train)
#
# pca = PCA(n_components=50)
# X_train_pca = pca.fit_transform(X_train_std)
# # print(X_train_pca[0])
# # print(X_train_pca)
# # print(X_train_std.shape)
outlier_verify = []
#
# from sklearn.ensemble import IsolationForest
# clf = IsolationForest().fit(X_train_pca)
# outlier_predict = clf.predict(X_train_pca)
# outlier_index = []
# bad_outlier = 0
# clean_outlier = 0
# for i in range(len(outlier_predict)):
#     if outlier_predict[i] == -1:
#         outlier_index.append(i)
#         outlier_verify.append(train_clean_bad_outlier[i])
#         if len(train_clean_bad_outlier) > i >= len(train_clean_dataset):
#             bad_outlier += 1
#         else:
#             clean_outlier += 1
# print("bad_outlier:{}, clean_outlier:{}".format(bad_outlier, clean_outlier))

train_clean_bad_set = []
for i in range(len(train_clean_bad_outlier)):
    # if i not in outlier_index:
    train_clean_bad_set.append(train_clean_bad_outlier[i])
print(len(train_clean_bad_set))
