import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import copy


columns = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
    'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width',
    'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi', 'price_range'
]

classes = ['0', '1', '2', '3']

label_index = columns.index('price_range')

# clean数据占比
good_sample_ratio = 0.7

train_set_before = pd.read_csv('data/train_normalize.csv')

# 首先将pandas读取的数据转化为array
train_set_before = np.array(train_set_before)

# 把label改成从0开始
# for i in train_set_before:
#     if i[label_index] == 1:
#         i[label_index] = 0
#     else:
#         i[label_index] = 1

label = np.array(train_set_before).T[label_index]
train_set_before = np.delete(train_set_before, label_index, axis=1)

# 将train_set转化为特定list形式
train_set = []
for i in range(len(train_set_before)):
    tmp = [torch.tensor(train_set_before[i].tolist()), int(label[i])]
    train_set.append(tmp)

train_set_tmp = copy.deepcopy(train_set)


# #--------------------------------------------------------------------
# from sklearn.neighbors import LocalOutlierFactor as LOF
#
# clf0 = LOF(n_neighbors=20)
# clf1 = LOF(n_neighbors=20)
# clf2 = LOF(n_neighbors=20)
# clf3 = LOF(n_neighbors=20)
#
# normal = 0
# abnormal = 0
# X_train0 = []
# X_train1 = []
# X_train2 = []
# X_train3 = []
#
# train_new = []
#
# for idx, tensor in enumerate(train_set):
#     if tensor[1] == 0:
#         X_train0.append(tensor[0].numpy())
#     elif tensor[1] == 1:
#         X_train1.append(tensor[0].numpy())
#     elif tensor[1] == 2:
#         X_train2.append(tensor[0].numpy())
#     elif tensor[1] == 3:
#         X_train3.append(tensor[0].numpy())
#
# X_train0 = np.array(X_train0).reshape(-1, 1)
# X_train1 = np.array(X_train1).reshape(-1, 1)
# X_train2 = np.array(X_train2).reshape(-1, 1)
# X_train3 = np.array(X_train3).reshape(-1, 1)
#
# predict0 = clf0.fit_predict(X_train0)
# predict1 = clf1.fit_predict(X_train1)
# predict2 = clf2.fit_predict(X_train2)
# predict3 = clf3.fit_predict(X_train3)
#
# for i in range(len(train_set)):
#     if train_set[i][1] == 0:
#         if predict0[i] == 1:
#             normal += 1
#             train_new.append(train_set[i])
#         else:
#             abnormal += 1
#     elif train_set[i][1] == 1:
#         if predict1[i] == 1:
#             normal += 1
#             train_new.append(train_set[i])
#         else:
#             abnormal += 1
#     elif train_set[i][1] == 2:
#         if predict2[i] == 1:
#             normal += 1
#             train_new.append(train_set[i])
#         else:
#             abnormal += 1
#     elif train_set[i][1] == 3:
#         if predict3[i] == 1:
#             normal += 1
#             train_new.append(train_set[i])
#         else:
#             abnormal += 1
#
#
# print(normal, abnormal)
# train_set = copy.deepcopy(train_new)
# train_set_tmp = copy.deepcopy(train_new)
# #--------------------------------------------------------------------



class MyDataSet(Dataset):
    def __init__(self, loaded_data):
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], index


cnt_label = {}
for idx, tensor in enumerate(train_set):
    cnt_label[tensor[1]] = cnt_label.get(tensor[1], 0) + 1
print(len(cnt_label))

cnt_good_label_tgt = {}
for k, v in cnt_label.items():
    cnt_good_label_tgt[k] = int(v * good_sample_ratio)

manipulate_label = {}
good_idx_set = []
for idx, tensor in enumerate(train_set):
    manipulate_label[tensor[1]] = manipulate_label.get(tensor[1], 0) + 1
    if manipulate_label[tensor[1]] > cnt_good_label_tgt[tensor[1]]:
        # train_set[idx][1] = (train_set[idx][1] + 2) % len(cnt_label)
        p = np.random.randint(0, len(cnt_label))
        while True:
            if p != tensor[1]:
                train_set[idx][1] = p
                break
            p = np.random.randint(0, len(cnt_label))
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

# train_clean_bad_set = train_clean_dataset + train_bad_dataset
train_clean_bad_outlier = train_clean_dataset + train_bad_dataset
print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_outlier))

# =============================================================================================
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
#
# X_train = []
# for i in train_clean_bad_outlier:
#     X_train.append(i[0].numpy())
# X_train = np.array(X_train)
#
# print(X_train.shape)
#
# sc = StandardScaler()
# X_train_std = sc.fit_transform(X_train)
#
# pca = PCA(n_components=15)
# X_train_pca = pca.fit_transform(X_train_std)
# # print(X_train_pca[0])
# # print(X_train_pca)
# print(X_train_std.shape)
outlier_verify = []

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
# =============================================================================================


#--------------------------------------------------------------------
# from sklearn.neighbors import LocalOutlierFactor as LOF
#
# clf0 = LOF(n_neighbors=20)
# clf1 = LOF(n_neighbors=20)
# clf2 = LOF(n_neighbors=20)
# clf3 = LOF(n_neighbors=20)
#
# normal = 0
# abnormal = 0
# X_train0 = []
# X_train1 = []
# X_train2 = []
# X_train3 = []
#
# train_new = []
#
# for idx, tensor in enumerate(train_clean_bad_set_outlier):
#     if tensor[1] == 0:
#         X_train0.append(tensor[0].numpy())
#     elif tensor[1] == 1:
#         X_train1.append(tensor[0].numpy())
#     elif tensor[1] == 2:
#         X_train2.append(tensor[0].numpy())
#     elif tensor[1] == 3:
#         X_train3.append(tensor[0].numpy())
#
# X_train0 = np.array(X_train0).reshape(-1, 1)
# X_train1 = np.array(X_train1).reshape(-1, 1)
# X_train2 = np.array(X_train2).reshape(-1, 1)
# X_train3 = np.array(X_train3).reshape(-1, 1)
#
# predict0 = clf0.fit_predict(X_train0)
# predict1 = clf1.fit_predict(X_train1)
# predict2 = clf2.fit_predict(X_train2)
# predict3 = clf3.fit_predict(X_train3)
#
# for i in range(len(train_set)):
#     if train_set[i][1] == 0:
#         if predict0[i] == 1:
#             normal += 1
#             train_new.append(train_set[i])
#         else:
#             abnormal += 1
#     elif train_set[i][1] == 1:
#         if predict1[i] == 1:
#             normal += 1
#             train_new.append(train_set[i])
#         else:
#             abnormal += 1
#     elif train_set[i][1] == 2:
#         if predict2[i] == 1:
#             normal += 1
#             train_new.append(train_set[i])
#         else:
#             abnormal += 1
#     elif train_set[i][1] == 3:
#         if predict3[i] == 1:
#             normal += 1
#             train_new.append(train_set[i])
#         else:
#             abnormal += 1
#
#
# print(normal, abnormal)
# train_set = copy.deepcopy(train_new)
# train_set_tmp = copy.deepcopy(train_new)
#--------------------------------------------------------------


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
#         p = np.random.randint(0, len(classes))
#
# # train_clean_dataset = train_clean_dataset[:9000]
# # train_bad_dataset = train_bad_dataset[:6000]
#
# train_clean_bad_set = train_clean_dataset + train_bad_dataset
# print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_set))

