import time
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import pandas
import torch
from get_data_image import *
from model import *
import operator
import math
# from pyod.models.knn import KNN

# GPU
device = torch.device("mps")

def aggregation_feature(input_features):
    # Define the attention mechanism
    # print(input_features.shape)
    attention = nn.Linear(input_features.size(1), 1, bias=False)
    # Compute the attention weights
    weights = attention(input_features)
    weights = torch.softmax(weights, dim=0)
    # Compute the weighted sum of the input features
    weighted_sum = torch.sum(weights * input_features, dim=0)
    # Print the output
    return weighted_sum

def interactions_feature_label(neighbor_features, neighbor_labels):
    # Define the interaction layer
    interaction = nn.Linear(len(neighbor_features[0]), len(neighbor_features[0]))
    # print(label.unsqueeze(1).float())
    # Concatenate the embedded feature and label along the feature dimension

    tmp_label = np.expand_dims(neighbor_labels[0], 0)
    tmp_label = torch.tensor(tmp_label)
    x = 1
    while x < len(neighbor_features[0]):
        tmp = np.expand_dims(neighbor_labels[0], 0)
        tmp_label = torch.cat((tmp_label, torch.tensor(tmp)), 0)
        x += 1
    concatenated = torch.cat((neighbor_features[0].unsqueeze(1).float(), tmp_label.unsqueeze(1).float()), dim=1)
    # Compute the element-wise product of the embedded feature and label
    product = torch.prod(concatenated, dim=1)
    # Project the concatenated tensor using the interaction layer
    ans = interaction(product)
    ans = ans.reshape([1, len(ans)])
    # print("ans:{}".format(ans))
    # 利用interaction layer将retrieval_pool样本的特征和标签进行聚合
    # Define the embedding layer to reduce the dimensionality of the feature
    i = 1
    while i < len(neighbor_features):
        # Define the interaction layer
        interaction = nn.Linear(len(neighbor_features[i]), len(neighbor_features[i]))
        # print(label.unsqueeze(1).float())
        # Concatenate the embedded feature and label along the feature dimension
        tmp_label = np.expand_dims(neighbor_labels[i], 0)
        tmp_label = torch.tensor(tmp_label)
        x = 1
        while x < len(neighbor_features[i]):
            tmp = np.expand_dims(neighbor_labels[i], 0)
            tmp_label = torch.cat((tmp_label, torch.tensor(tmp)), 0)
            # tmp_label = torch.cat((tmp_label, torch.tensor(neighbor_labels[i])), 0)
            x += 1
        concatenated = torch.cat((neighbor_features[i].unsqueeze(1).float(), tmp_label.unsqueeze(1).float()), dim=1)

        # Compute the element-wise product of the embedded feature and label
        product = torch.prod(concatenated, dim=1)

        # Project the concatenated tensor using the interaction layer
        projected = interaction(product)

        ans = torch.cat((ans, projected.reshape([1, len(projected)])), 0)
        # Print the output
        i += 1
    return aggregation_feature(ans)
    # print(projected.squeeze(0))

# 测试influence detect的效果
def misdetect(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset, train_clean_bad_set_ground_truth):
    total_len = len(train_clean_bad_set)
    clean_len = len(train_clean_dataset)
    bad_len = len(train_bad_dataset)

    new_train_clean_bad_set = copy.deepcopy(train_clean_bad_set)
    train_clean_bad_set_copy = copy.deepcopy(train_clean_bad_set)
    ground_truth = copy.deepcopy(train_clean_bad_set_ground_truth)

    train_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)

    epoch = 1

    detect_num = int(len(train_bad_dataset) / 4)
    clean_pool_len = int(len(train_bad_dataset) / 8)
    bad_pool_len = int(len(train_bad_dataset) / 4)

    # 总共的correct个数
    total_correct_num = 0
    final_dirty_set = []

    print(Models[dataset])
    # 第一轮，选出干净池子和脏池子
    model = Models[dataset]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)

    model.train()
    early_loss = np.zeros(total_len, dtype=np.float64)

    for i in range(epoch):
        for data in train_loader:
            # GPU加速
            train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            train_label_predict = model(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function(train_label_predict, train_label)
            train_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            num = 0
            for data in test_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                test_label_predict = test_label_predict.to(device)
                loss = loss_function(test_label_predict, test_label)
                early_loss[num] += loss
                num += 1

    loss_clean = sorted(enumerate(list(early_loss)), key=lambda x: x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
    clean_pool_idx = [x[0] for x in loss_clean]  # 获取排序好后b坐标,下标在第0位

    model_clean = Models[dataset]
    model_clean = model_clean.to(device)
    optimizer_clean = torch.optim.Adam(model_clean.parameters(), lr=0.001)
    loss_function_clean = nn.CrossEntropyLoss()
    loss_function_clean = loss_function_clean.to(device)

    model_clean.train()
    clean_pool = []
    for indeX in range(clean_pool_len):
        clean_pool.append(train_clean_bad_set[clean_pool_idx[indeX]])
    train_loader_clean = DataLoader(dataset=MyDataSet(clean_pool), batch_size=64, shuffle=True)

    # 训练干净数据集到收敛
    for epo in range(40):
        for data in train_loader_clean:
            # GPU加速
            train_feature_clean, train_label_clean = data
            train_feature_clean = train_feature_clean.to(device)
            train_label_clean = train_label_clean.to(device)
            optimizer_clean.zero_grad()
            train_label_predict_clean = model_clean(train_feature_clean)

            # GPU加速
            train_label_predict_clean = train_label_predict_clean.to(device)
            train_loss_clean = loss_function_clean(train_label_predict_clean, train_label_clean)
            train_loss_clean.backward()
            optimizer_clean.step()

    loss_bad = sorted(enumerate(list(early_loss)), key=lambda x: -x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
    bad_pool_idx = [x[0] for x in loss_bad]  # 获取排序好后b坐标,下标在第0位

    bad_pool_idx = bad_pool_idx[:bad_pool_len]
    influence_bad = []
    model_clean.eval()
    for indeX in bad_pool_idx:
        bad_test = [train_clean_bad_set[indeX]]
        test_loader_bad = DataLoader(dataset=MyDataSet(bad_test), batch_size=1, shuffle=False)
        for data in test_loader_bad:
            train_feature_bad, train_label_bad = data
            train_feature_bad = train_feature_bad.to(device)
            train_label_bad = train_label_bad.to(device)
            train_label_predict_bad = model_clean(train_feature_bad)

            # GPU加速
            train_label_predict_bad = train_label_predict_bad.to(device)
            train_loss_bad = loss_function_clean(train_label_predict_bad, train_label_bad)

            train_loss_bad_gradient = torch.autograd.grad(train_loss_bad, model_clean.parameters())

            grad = 0.0
            for item in train_loss_bad_gradient:
                item = torch.norm(item)
                grad += item

            tmp = [indeX, grad]
            influence_bad.append(tmp)

    influence_bad_sorted = sorted(influence_bad, key=lambda x: -x[1])
    influence_bad_idx = [x[0] for x in influence_bad_sorted]  # 获取排序好后b坐标,下标在第0位

    correct_num = 0
    true_bad_detected_idx = []
    detect_idx_50 = []
    early_loss_predicted = np.zeros((total_len, 1), dtype=np.int32)
    early_loss_actual = np.ones((total_len, 1), dtype=np.int32)
    for i in range(clean_len):
        early_loss_actual[i][0] = 0

    # 每轮detect500个
    for i in range(detect_num):
        detect_idx_50.append(influence_bad_idx[i])
        early_loss_predicted[influence_bad_idx[i]][0] = 1

        if (total_len - 1) >= influence_bad_idx[i] >= clean_len:
            correct_num += 1
            true_bad_detected_idx.append(influence_bad_idx[i])

    total_correct_num += correct_num

    # 计算总的精度
    acc = accuracy_score(early_loss_actual, early_loss_predicted)
    precision = precision_score(early_loss_actual, early_loss_predicted)
    recall = recall_score(early_loss_actual, early_loss_predicted)
    print("1:precision:{},recall:{},f1:{}".format(precision, recall, 2*precision*recall/(precision+recall)))

    ground_truth_tmp = []
    new_train_clean_bad_set = []
    for i in range(total_len):
        if i not in detect_idx_50:
            new_train_clean_bad_set.append(train_clean_bad_set_copy[i])
            ground_truth_tmp.append(ground_truth[i])
        else:
            final_dirty_set.append(ground_truth[i])
    # for i in range(total_len):
    #     if i not in detect_idx_50:
    #         train_clean_bad_set_tmp.append(train_clean_bad_set_copy[i])
    train_clean_bad_set_copy = copy.deepcopy(new_train_clean_bad_set)
    ground_truth = copy.deepcopy(ground_truth_tmp)

    total_len = len(new_train_clean_bad_set)
    bad_len = bad_len - correct_num
    clean_len = total_len - bad_len

    train_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)


    # 800 / 50 = 16
    for times in range(3):
        model = Models[dataset]
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(device)

        model.train()
        early_loss = np.zeros(total_len, dtype=np.float64)
        if times == 2:
            detect_num = int(len(train_bad_dataset) / 5)
        if (times * detect_num) >= 0.3 * len(train_bad_dataset):
            # print("------------------------------------kaishi----------------------------------------")
            epoch = 3

        for i in range(epoch):
            for data in train_loader:
                # GPU加速
                train_feature, train_label = data
                train_feature = train_feature.to(device)
                train_label = train_label.to(device)
                optimizer.zero_grad()
                train_label_predict = model(train_feature)

                # GPU加速
                train_label_predict = train_label_predict.to(device)
                train_loss = loss_function(train_label_predict, train_label)
                train_loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                num = 0
                for data in test_loader:
                    test_feature, test_label = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model(test_feature)
                    test_label_predict = test_label_predict.to(device)
                    loss = loss_function(test_label_predict, test_label)
                    early_loss[num] += loss
                    num += 1

        loss_bad = sorted(enumerate(list(early_loss)),
                          key=lambda x: -x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
        bad_pool_idx = [x[0] for x in loss_bad]  # 获取排序好后b坐标,下标在第0位

        bad_pool_idx = bad_pool_idx[:bad_pool_len]
        influence_bad = []
        model_clean.eval()
        for indeX in bad_pool_idx:
            bad_test = [train_clean_bad_set[indeX]]
            test_loader_bad = DataLoader(dataset=MyDataSet(bad_test), batch_size=1, shuffle=False)
            for data in test_loader_bad:
                train_feature_bad, train_label_bad = data
                train_feature_bad = train_feature_bad.to(device)
                train_label_bad = train_label_bad.to(device)
                train_label_predict_bad = model_clean(train_feature_bad)

                # GPU加速
                train_label_predict_bad = train_label_predict_bad.to(device)
                train_loss_bad = loss_function_clean(train_label_predict_bad, train_label_bad)

                train_loss_bad_gradient = torch.autograd.grad(train_loss_bad, model_clean.parameters())

                grad = 0.0
                for item in train_loss_bad_gradient:
                    item = torch.norm(item)
                    grad += item

                tmp = [indeX, grad]
                influence_bad.append(tmp)

        influence_bad_sorted = sorted(influence_bad, key=lambda x: -x[1])
        influence_bad_idx = [x[0] for x in influence_bad_sorted]  # 获取排序好后b坐标,下标在第0位

        if times == 2:
            detect_num = int(len(train_bad_dataset) / 5)
        # print(len(influence_bad_idx))

        correct_num = 0
        true_bad_detected_idx = []
        detect_idx_50 = []
        early_loss_predicted = np.zeros((total_len, 1), dtype=np.int32)
        early_loss_actual = np.ones((total_len, 1), dtype=np.int32)
        for i in range(clean_len):
            early_loss_actual[i][0] = 0

        # 每轮detect500个
        for i in range(detect_num):
            detect_idx_50.append(influence_bad_idx[i])
            early_loss_predicted[influence_bad_idx[i]][0] = 1

            if (total_len - 1) >= influence_bad_idx[i] >= clean_len:
                correct_num += 1
                true_bad_detected_idx.append(influence_bad_idx[i])

        # print("loss最高的脏数据占比为:{}".format(correct_num / detect_num))
        total_correct_num += correct_num

        # 计算总的精度
        pre = total_correct_num / (detect_num * (times + 2))
        recall = total_correct_num / len(train_bad_dataset)
        f1 = 2 * recall * pre / (pre + recall)
        # print("early loss detection: acc:{},precision:{},recall:{}".format(acc, precision, recall))
        print("After {}round, precision{},recall{},f1 score{}".format(times + 2, pre, recall, f1))

        ground_truth_tmp = []
        new_train_clean_bad_set = []
        for i in range(total_len):
            if i not in detect_idx_50:
                new_train_clean_bad_set.append(train_clean_bad_set_copy[i])
                ground_truth_tmp.append(ground_truth[i])
            else:
                final_dirty_set.append(ground_truth[i])
        # for i in range(total_len):
        #     if i not in detect_idx_50:
        #         train_clean_bad_set_tmp.append(train_clean_bad_set_copy[i])
        train_clean_bad_set_copy = copy.deepcopy(new_train_clean_bad_set)
        ground_truth = copy.deepcopy(ground_truth_tmp)

        total_len = len(new_train_clean_bad_set)
        bad_len = bad_len - correct_num
        clean_len = total_len - bad_len
        # print(clean_len, bad_len, total_len)

        train_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True)
        test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)

    precision = total_correct_num / (4 * detect_num)
    recall = total_correct_num / len(train_bad_dataset)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("detect_iterate * detect_num:{}".format(4 * detect_num))
    print("detect precision：{}".format(precision))
    print("detect recall：{}".format(recall))
    print("detect f1 score：{}".format(f1))

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def knn(train_clean_bad_set, train_clean_dataset, train_bad_dataset):
    features = []
    labels = []
    for item in train_clean_bad_set:
        features.append(item[0])
        labels.append(item[1])
    features = pandas.DataFrame(features)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(features)

    detect_num = 0
    correct_num = 0
    for i in range(len(train_clean_bad_set)):
        distance, index = nbrs.kneighbors(train_clean_bad_set[i][0].reshape(1, -1), 3)
        neighbors = []
        for x in np.array(index.tolist()[0]):
            neighbors.append(train_clean_bad_set[x])
        result = getResponse(neighbors)
        if train_clean_bad_set[i][1] != result:
            detect_num += 1
            if (len(train_clean_bad_set) - 1) >= i >= len(train_clean_dataset):
                correct_num += 1

    precision = correct_num / detect_num
    recall = correct_num / len(train_bad_dataset)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("correct_num{}, detect_num:{}".format(correct_num, detect_num))
    print("detect precision：{}".format(precision))
    print("detect recall：{}".format(recall))
    print("detect f1 score：{}".format(f1))

def clean_pool(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset):
    total_len = len(train_clean_bad_set)
    clean_len = len(train_clean_dataset)
    bad_len = len(train_bad_dataset)

    train_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=1, shuffle=False)

    epoch = 3

    # 总共的correct个数
    total_correct_num = 0
    total_detect_num = 0

    # 第一轮，选出干净池子和脏池子
    model = Models[dataset]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)

    # 800 / 50 = 16
    for times in range(1):
        model.train()
        early_loss = np.zeros(total_len, dtype=np.float64)
        for i in range(epoch):
            for data in train_loader:
                # GPU加速
                train_feature, train_label = data
                train_feature = train_feature.to(device)
                train_label = train_label.to(device)
                optimizer.zero_grad()
                train_label_predict = model(train_feature)

                # GPU加速
                train_label_predict = train_label_predict.to(device)
                train_loss = loss_function(train_label_predict, train_label)
                train_loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                num = 0
                for data in test_loader:
                    test_feature, test_label = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model(test_feature)
                    test_label_predict = test_label_predict.to(device)
                    loss = loss_function(test_label_predict, test_label)
                    early_loss[num] += loss
                    num += 1

        loss_clean = sorted(enumerate(list(early_loss)),
                            key=lambda x: x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
        clean_pool_idx = [x[0] for x in loss_clean]  # 获取排序好后b坐标,下标在第0位

        model = Models[dataset]
        model_clean = model_clean.to(device)
        optimizer_clean = torch.optim.Adam(model_clean.parameters(), lr=0.001)
        loss_function_clean = nn.CrossEntropyLoss()
        loss_function_clean = loss_function_clean.to(device)

        model_clean.train()
        clean_pool = []
        for indeX in range(int(len(train_clean_dataset) / 2)):
            clean_pool.append(train_clean_bad_set[clean_pool_idx[indeX]])
        train_loader_clean = DataLoader(dataset=MyDataSet(clean_pool), batch_size=128, shuffle=True)

        # 训练干净数据集到收敛
        for epo in range(40):
            for data in train_loader_clean:
                # GPU加速
                train_feature_clean, train_label_clean = data
                train_feature_clean = train_feature_clean.to(device)
                train_label_clean = train_label_clean.to(device)
                optimizer_clean.zero_grad()
                train_label_predict_clean = model_clean(train_feature_clean)

                # GPU加速
                train_label_predict_clean = train_label_predict_clean.to(device)
                train_loss_clean = loss_function_clean(train_label_predict_clean, train_label_clean)
                train_loss_clean.backward()
                optimizer_clean.step()
            model_clean.eval()
            with torch.no_grad():
                test_accuracy_total = 0
                for data in train_loader_clean:
                    test_feature_total, test_label_total = data
                    test_feature_total = test_feature_total.to(device)
                    test_label_total = test_label_total.to(device)
                    test_label_predict_total = model(test_feature_total)
                    test_accuracy_num_total = (test_label_predict_total.argmax(1) == test_label_total).sum()
                    test_accuracy_total += test_accuracy_num_total
                if (test_accuracy_total / (int(len(train_clean_dataset) / 2))) >= 0.9:
                    break

        num = 0
        model_clean.eval()
        with torch.no_grad():
            for data in test_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model_clean(test_feature)
                if test_label_predict.argmax(1) != test_label:
                    total_detect_num += 1
                    if total_len > num >= (clean_len - 1):
                        total_correct_num += 1
                num += 1
    precision = total_correct_num / total_detect_num
    recall = total_correct_num / len(train_bad_dataset)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("detect precision：{}".format(precision))
    print("detect recall：{}".format(recall))
    print("detect f1 score：{}".format(f1))

def cleanlab(dataset ,train_clean_bad_set, train_clean_dataset, train_bad_dataset):
    features = []
    labels = []
    for i in train_clean_bad_set:
        features.append(i[0].numpy())
        labels.append(i[1])
    features = np.array(features)
    labels = np.array(labels)

    from skorch import NeuralNetClassifier
    from sklearn.model_selection import cross_val_predict

    model_skorch = NeuralNetClassifier(Models[dataset])

    from cleanlab.filter import find_label_issues
    pred_probs = cross_val_predict(
        model_skorch,
        features,
        labels,
        cv=10,
        method="predict_proba",
    )
    predicted_labels = pred_probs.argmax(axis=1)
    ranked_label_issues = find_label_issues(
        labels,
        pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    detect_num = len(ranked_label_issues)
    correct_num = 0
    for i in ranked_label_issues:
        if (len(train_clean_bad_set) - 1) >= i >= len(train_clean_dataset):
            correct_num += 1

    precision = correct_num / detect_num
    recall = correct_num / len(train_bad_dataset)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("correct_num{}, detect_num:{}".format(correct_num, detect_num))
    print("detect precision：{}".format(precision))
    print("detect recall：{}".format(recall))
    print("detect f1 score：{}".format(f1))

# DUTI
def forget_event(train_clean_bad_set, train_clean_dataset, train_bad_dataset):
    clean_pool = train_clean_bad_set[:3]

    features_vali = []
    labels_vali = []
    for i in clean_pool:
        features_vali.append(i[0].numpy())
        labels_vali.append(i[1])
    features_vali = np.array(features_vali)
    labels_vali = np.array(labels_vali)


    new_train_clean_bad_set = copy.deepcopy(train_clean_bad_set[3:])
    train_clean_bad_set_copy = copy.deepcopy(train_clean_bad_set[3:])

    total_len = len(new_train_clean_bad_set)
    clean_len = len(train_clean_dataset) - 3
    bad_len = len(train_bad_dataset)

    features = []
    labels = []
    for i in new_train_clean_bad_set:
        features.append(i[0].numpy())
        labels.append(i[1])
    features = np.array(features)
    labels = np.array(labels)

    from sklearn.neighbors import NearestCentroid
    NCN = NearestCentroid("manhattan")
    NCN.fit(features, labels)
    predict_vali = NCN.predict(features_vali)

    # 总共的correct个数
    total_correct_num = 0
    total_detect_num = 0
    times = 0
    while((predict_vali == labels_vali).sum() != 3):
        predict_logistic = NCN.predict(features)
        detect_idx_50 = []
        correct_num = 0
        for i in range(len(predict_logistic)):
            if predict_logistic[i] != labels[i]:
                total_detect_num += 1
                detect_idx_50.append(i)
                if (total_len - 1) >= i >= clean_len:
                    correct_num += 1

        new_train_clean_bad_set = []
        for i in range(len(predict_logistic)):
            if i not in detect_idx_50:
                new_train_clean_bad_set.append(train_clean_bad_set_copy[i])
        train_clean_bad_set_copy = copy.deepcopy(new_train_clean_bad_set)

        total_len = len(new_train_clean_bad_set)
        bad_len = bad_len - correct_num
        clean_len = total_len - bad_len
        total_correct_num += correct_num

        features = []
        labels = []
        for i in new_train_clean_bad_set:
            features.append(i[0].numpy())
            labels.append(i[1])
        features = np.array(features)
        labels = np.array(labels)

        # LogisticRegression
        from sklearn.linear_model import LogisticRegression
        logistic = LogisticRegression()
        logistic.fit(features, labels)
        predict_vali = logistic.predict(features_vali)
        if times >= 20:
            break
        times += 1

    precision = total_correct_num / total_detect_num
    recall = total_correct_num / len(train_bad_dataset)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("correct_num{}, detect_num:{}".format(total_correct_num, total_detect_num))
    print("detect precision：{}".format(precision))
    print("detect recall：{}".format(recall))
    print("detect f1 score：{}".format(f1))

# ensemble
def ensemble(train_clean_bad_set, train_clean_dataset, train_bad_dataset):
    features = []
    labels = []
    for i in train_clean_bad_set:
        features.append(i[0].numpy())
        labels.append(i[1])
    features = np.array(features)
    labels = np.array(labels)

    # 随机森林
    from sklearn.ensemble import RandomForestClassifier
    random_forest = RandomForestClassifier()
    random_forest.fit(features, labels)
    predict_random_tree = random_forest.predict(features)

    # KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree').fit(features, labels)
    knn.fit(features, labels)
    predict_KNN = knn.predict(features)

    #NCN
    from sklearn.neighbors import NearestCentroid
    NCN = NearestCentroid("manhattan")
    NCN.fit(features, labels)
    predict_NCN = NCN.predict(features)

    # AdaBoostClassifier
    from sklearn.ensemble import AdaBoostClassifier
    adaboost = AdaBoostClassifier()
    adaboost.fit(features, labels)
    predict_adaboost = adaboost.predict(features)

    # LogisticRegression
    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression()
    logistic.fit(features, labels)
    predict_logistic = logistic.predict(features)

    # QDA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    qda = QDA()
    qda.fit(features, labels)
    predict_qda = qda.predict(features)

    # LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda = LDA()
    lda.fit(features, labels)
    predict_lda = lda.predict(features)

    # 支持向量机
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(features, labels)
    predict_svm = svm.predict(features)

    # naive_bayes
    from sklearn.naive_bayes import GaussianNB
    naive_bayes = GaussianNB()
    naive_bayes.fit(features, labels)
    predict_naive_bayes = naive_bayes.predict(features)

    ## 决策树
    from sklearn import tree
    decision = tree.DecisionTreeClassifier()
    decision.fit(features, labels)
    predict_decision = decision.predict(features)

    predict_final = np.zeros((len(train_clean_bad_set), len(classes)), dtype=np.int32)
    predict_count = np.zeros(len(train_clean_bad_set), dtype=np.int32)
    for i in range(len(train_clean_bad_set)):
        predict_final[i][predict_NCN[i]] += 1
        predict_final[i][predict_KNN[i]] += 1
        predict_final[i][predict_adaboost[i]] += 1
        # predict_final[i][predict_random_tree[i]] += 1
        # predict_final[i][predict_qda[i]] += 1
        # predict_final[i][predict_logistic[i]] += 1
        # predict_final[i][predict_lda[i]] += 1
        # predict_final[i][predict_svm[i]] += 1
        # predict_final[i][predict_naive_bayes[i]] += 1
        # predict_final[i][predict_decision[i]] += 1


    detect_num_consensus_filters = 0
    correct_num_consensus_filters = 0
    for i in range(len(train_clean_bad_set)):
        if predict_final[i][labels[i]] == 0:
            detect_num_consensus_filters += 1
            if (len(train_clean_bad_set) - 1) >= i >= len(train_clean_dataset):
                correct_num_consensus_filters += 1
    # for i in range(total_len):
    #     num = 0
    #     for j in range(len(classes)):
    #         if predict_final[i][j] != 0:
    #             num += 1
    #     predict_count[i] = num
    #
    # print(predict_count)
    #
    #
    # for i in range(len(predict_count)):
    #     if predict_count[i] != 1:
    #         detect_num += 1
    #         if (total_len - 1) >= i >= clean_len:
    #             correct_num += 1

    precision_consensus_filters = correct_num_consensus_filters / detect_num_consensus_filters
    recall_consensus_filters = correct_num_consensus_filters / len(train_bad_dataset)
    f1_consensus_filters = 2 * (precision_consensus_filters * recall_consensus_filters) / (
            precision_consensus_filters + recall_consensus_filters)
    print("==================Ensemble method with consensus filters====================")
    print("correct_num{}, detect_num:{}".format(correct_num_consensus_filters, detect_num_consensus_filters))
    print("detect precision：{}".format(precision_consensus_filters))
    print("detect recall：{}".format(recall_consensus_filters))
    print("detect f1 score：{}".format(f1_consensus_filters))

    detect_num_majority_vote = 0
    correct_num_majority_vote = 0
    for i in range(len(train_clean_bad_set)):
        if predict_final[i][labels[i]] <= 2:
            detect_num_majority_vote += 1
            if (len(train_clean_bad_set) - 1) >= i >= len(train_clean_dataset):
                correct_num_majority_vote += 1
    precision_majority_vote = correct_num_majority_vote / detect_num_majority_vote
    recall_majority_vote = correct_num_majority_vote / len(train_bad_dataset)
    f1_majority_vote = 2 * (precision_majority_vote * recall_majority_vote) / (
                precision_majority_vote + recall_majority_vote)
    print("==================Ensemble method with majority vote====================")
    print("correct_num{}, detect_num:{}".format(correct_num_majority_vote, detect_num_majority_vote))
    print("detect precision：{}".format(precision_majority_vote))
    print("detect recall：{}".format(recall_majority_vote))
    print("detect f1 score：{}".format(f1_majority_vote))

def mentornet(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset):
    new_train_clean_bad_set = []
    for i in range(len(train_clean_bad_set)):
        tmp = [train_clean_bad_set[i], i]
        new_train_clean_bad_set.append(tmp)

    train_clean_size = int(0.5 * len(train_clean_bad_set))
    train_bad_size = len(train_clean_bad_set) - train_clean_size
    train_set_split1, train_set_split2 = torch.utils.data.random_split(train_clean_bad_set,
                                                                       [train_clean_size, train_bad_size])
    total_detect_num = 0
    total_correct_num = 0
    train_set1 = []
    train_set2 = []
    for i in train_set_split1:
        train_set1.append(i[0])
    for i in train_set_split2:
        train_set2.append(i[0])

    train_loader = DataLoader(dataset=MyDataSet(train_set1), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(train_set2), batch_size=1, shuffle=False)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)
    model = Models[dataset]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for i in range(5):
        for data in train_loader:
            # GPU加速
            train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            train_label_predict = model(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function(train_label_predict, train_label)
            train_loss.backward()
            optimizer.step()

    model.eval()
    num = 0
    with torch.no_grad():
        for data in test_loader:
            test_feature, test_label = data
            test_feature = test_feature.to(device)
            test_label = test_label.to(device)
            test_label_predict = model(test_feature)
            test_label_predict = test_label_predict.to(device)
            if test_label_predict.argmax(1) != test_label:
                total_detect_num += 1
                if len(train_clean_bad_set) > train_set_split2[num][1] >= len(train_clean_dataset) - 1:
                    total_correct_num += 1
            num += 1

    train_loader = DataLoader(dataset=MyDataSet(train_set2), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(train_set1), batch_size=1, shuffle=False)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)
    model = Models[dataset]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for i in range(5):
        for data in train_loader:
            # GPU加速
            train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            train_label_predict = model(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function(train_label_predict, train_label)
            train_loss.backward()
            optimizer.step()

    model.eval()
    num = 0
    with torch.no_grad():
        for data in test_loader:
            test_feature, test_label = data
            test_feature = test_feature.to(device)
            test_label = test_label.to(device)
            test_label_predict = model(test_feature)
            test_label_predict = test_label_predict.to(device)
            if test_label_predict.argmax(1) != test_label:
                total_detect_num += 1
                if len(train_clean_bad_set) > train_set_split2[num][1] >= len(train_clean_dataset) - 1:
                    total_correct_num += 1
            num += 1

    precision = total_correct_num / total_detect_num
    recall = total_correct_num / len(train_bad_dataset)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("correct_num{}, detect_num:{}".format(total_correct_num, total_detect_num))
    print("detect precision：{}".format(precision))
    print("detect recall：{}".format(recall))
    print("detect f1 score：{}".format(f1))


def coteaching(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset):
    import torch.nn.functional as F

    model1 = Models[dataset]
    model1 = model1.to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    loss_function1 = nn.CrossEntropyLoss()
    loss_function1 = loss_function1.to(device)

    model2 = Models[dataset]
    model2 = model2.to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    loss_function2 = nn.CrossEntropyLoss()
    loss_function2 = loss_function2.to(device)

    train_loader1 = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=1, shuffle=False)
    train_loader2 = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=True)

    model1.train()
    model2.train()
    for epo in range(200):
        early_loss1 = np.zeros(len(train_clean_bad_set), dtype=np.float64)
        for data in train_loader1:
            # GPU加速
            train_feature, train_label, idx = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer1.zero_grad()
            train_label_predict = model1(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            # train_loss = loss_function1(train_label_predict, train_label)
            for i in range(len(idx)):
                early_loss1[idx[i].item()] = F.cross_entropy(train_label_predict[i].view(1,-1), train_label[i].view(-1),reduce = False)

        early_loss2 = np.zeros(len(train_clean_bad_set), dtype=np.float64)
        for data in train_loader2:
            # GPU加速
            train_feature, train_label, idx = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer2.zero_grad()
            train_label_predict = model2(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            for i in range(len(idx)):
                early_loss2[idx[i].item()] = F.cross_entropy(train_label_predict[i].view(1,-1), train_label[i].view(-1),reduce = False)

        early_loss1 = sorted(enumerate(list(early_loss1)),
                            key=lambda x: x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
        early_loss1_idx = [x[0] for x in early_loss1]  # 获取排序好后b坐标,下标在第0位

        early_loss2 = sorted(enumerate(list(early_loss2)),
                             key=lambda x: x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
        early_loss2_idx = [x[0] for x in early_loss2]  # 获取排序好后b坐标,下标在第0位

        update1 = []
        for index in range(int(0.5*len(early_loss1))):
            update1.append(train_clean_bad_set[early_loss1_idx[index]])

        update2 = []
        for index in range(int(0.5 * len(early_loss2))):
            update2.append(train_clean_bad_set[early_loss2_idx[index]])

        train_loader_loss1 = DataLoader(dataset=MyDataSet(update1), batch_size=128, shuffle=True)
        train_loader_loss2 = DataLoader(dataset=MyDataSet(update2), batch_size=128, shuffle=True)

        for data in train_loader_loss2:
            # GPU加速
            train_feature, train_label, idx = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer1.zero_grad()
            train_label_predict = model1(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function1(train_label_predict, train_label)
            train_loss.backward()
            optimizer1.step()

        for data in train_loader_loss1:
            # GPU加速
            train_feature, train_label, idx = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer2.zero_grad()
            train_label_predict = model2(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function2(train_label_predict, train_label)
            train_loss.backward()
            optimizer2.step()


        detect_num = 0
        correct_num = 0

        if epo % 10 == 0:
            model1.eval()
            with torch.no_grad():
                num = 0
                for data in test_loader:
                    test_feature, test_label, index = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model1(test_feature)
                    test_label_predict = test_label_predict.to(device)
                    if test_label_predict.argmax(1) != test_label:
                        detect_num += 1
                        if (len(train_clean_bad_set) - 1) >= num >= len(train_clean_dataset):
                            correct_num += 1
                    num += 1

                precision = correct_num / detect_num
                recall = correct_num / len(train_bad_dataset)
                f1 = 2 * (precision * recall) / (precision + recall)
                print("第{}轮: precision:{},recall:{},f1:{}".format(epo + 1, precision, recall, f1))



def non_iter(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset):
    train_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=6400, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=1, shuffle=False)
    model = Models[dataset]
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    early_loss = np.zeros(len(train_clean_bad_set), dtype=float)
    for i in range(2):
        # batch_num = 0
        for data in train_loader:
            # GPU加速
            train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            train_label_predict = model(train_feature)
            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function(train_label_predict, train_label)
            train_loss.backward()
            optimizer.step()

        model.eval()
        num = 0
        with torch.no_grad():
            for data in test_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                test_label_predict = test_label_predict.to(device)
                # loss = abs(max(test_label_predict[0][0].item(), test_label_predict[0][1].item()) - test_label.item())
                # loss = (test_label_predict.argmax(1).double() - test_label.double())**2
                loss = loss_function(test_label_predict, test_label)
                early_loss[num] += loss
                num += 1

    b = sorted(enumerate(early_loss), key=lambda x: -x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
    c = [x[0] for x in b]  # 获取排序好后b坐标,下标在第0位
    correct_num = 0
    bad_detected_idx = []
    early_loss_predicted = np.zeros((len(train_clean_bad_set), 1), dtype=np.int)
    early_loss_actual = np.ones((len(train_clean_bad_set), 1), dtype=np.int)
    for i in range(len(train_clean_dataset)):
        early_loss_actual[i][0] = 0

    for i in range(len(train_bad_dataset)):
            # if i == 50:
            #     print(correct_num, "correct_num_50")
            # if i == 100:
            #     print(correct_num, "correct_num_100")
            #     exit()
        early_loss_predicted[c[i]][0] = 1
        if (len(train_clean_bad_set) - 1) >= c[i] >= (len(train_clean_dataset) - 1):
            correct_num += 1
            bad_detected_idx.append(c[i])

        # 计算总的精度
    acc = accuracy_score(early_loss_actual, early_loss_predicted)
    precision = precision_score(early_loss_actual, early_loss_predicted)
    recall = recall_score(early_loss_actual, early_loss_predicted)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("acc:{},precision:{},recall:{},f1:{}".format(acc, precision, recall, f1))

def M_W_IM(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset):
    total_len = len(train_clean_bad_set)
    clean_len = len(train_clean_dataset)
    bad_len = len(train_bad_dataset)

    new_train_clean_bad_set = copy.deepcopy(train_clean_bad_set)
    train_clean_bad_set_copy = copy.deepcopy(train_clean_bad_set)
    ground_truth = copy.deepcopy(train_clean_bad_set)

    train_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)

    epoch = 1

    # 总共的correct个数
    total_correct_num = 0
    final_dirty_set = []


    # 第一轮，选出干净池子和脏池子
    model = Models[dataset]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)

    model.train()
    early_loss = np.zeros(total_len, dtype=np.float64)

    for i in range(epoch):
        for data in train_loader:
            # GPU加速
            train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            train_label_predict = model(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function(train_label_predict, train_label)
            train_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            num = 0
            for data in test_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                test_label_predict = test_label_predict.to(device)
                loss = loss_function(test_label_predict, test_label)
                early_loss[num] += loss
                num += 1

    loss_clean = sorted(enumerate(list(early_loss)), key=lambda x: x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
    clean_pool_idx = [x[0] for x in loss_clean]  # 获取排序好后b坐标,下标在第0位

    model_clean = Models[dataset]
    model_clean = model_clean.to(device)
    optimizer_clean = torch.optim.Adam(model_clean.parameters(), lr=0.001)
    loss_function_clean = nn.CrossEntropyLoss()
    loss_function_clean = loss_function_clean.to(device)

    model_clean.train()
    clean_pool = []
    for indeX in range(1000):
        clean_pool.append(train_clean_bad_set[clean_pool_idx[indeX]])
    train_loader_clean = DataLoader(dataset=MyDataSet(clean_pool), batch_size=64, shuffle=True)

    # 训练干净数据集到收敛
    for epo in range(40):
        for data in train_loader_clean:
            # GPU加速
            train_feature_clean, train_label_clean = data
            train_feature_clean = train_feature_clean.to(device)
            train_label_clean = train_label_clean.to(device)
            optimizer_clean.zero_grad()
            train_label_predict_clean = model_clean(train_feature_clean)

            # GPU加速
            train_label_predict_clean = train_label_predict_clean.to(device)
            train_loss_clean = loss_function_clean(train_label_predict_clean, train_label_clean)
            train_loss_clean.backward()
            optimizer_clean.step()


    influence_bad = []
    model_clean.eval()
    for indeX in range(len(train_clean_bad_set)):
        bad_test = [train_clean_bad_set[indeX]]
        test_loader_bad = DataLoader(dataset=MyDataSet(bad_test), batch_size=1, shuffle=False)
        for data in test_loader_bad:
            train_feature_bad, train_label_bad = data
            train_feature_bad = train_feature_bad.to(device)
            train_label_bad = train_label_bad.to(device)
            train_label_predict_bad = model_clean(train_feature_bad)

            # GPU加速
            train_label_predict_bad = train_label_predict_bad.to(device)
            train_loss_bad = loss_function_clean(train_label_predict_bad, train_label_bad)

            train_loss_bad_gradient = torch.autograd.grad(train_loss_bad, model_clean.parameters(), allow_unused=True)

            grad = 0.0
            for item in train_loss_bad_gradient:
                if item == None:
                    continue
                item = torch.norm(item)
                grad += item

            tmp = [indeX, grad]
            influence_bad.append(tmp)

    influence_bad_sorted = sorted(influence_bad, key=lambda x: -x[1])
    influence_bad_idx = [x[0] for x in influence_bad_sorted]  # 获取排序好后b坐标,下标在第0位


    correct_num = 0
    true_bad_detected_idx = []
    detect_idx_50 = []
    early_loss_predicted = np.zeros((total_len, 1), dtype=np.int32)
    early_loss_actual = np.ones((total_len, 1), dtype=np.int32)
    for i in range(clean_len):
        early_loss_actual[i][0] = 0

    # 每轮detect500个
    for i in range(len(train_bad_dataset)):
        detect_idx_50.append(influence_bad_idx[i])
        early_loss_predicted[influence_bad_idx[i]][0] = 1

        if (total_len - 1) >= influence_bad_idx[i] >= clean_len:
            correct_num += 1
            true_bad_detected_idx.append(influence_bad_idx[i])

    total_correct_num += correct_num

    # 计算总的精度
    acc = accuracy_score(early_loss_actual, early_loss_predicted)
    precision = precision_score(early_loss_actual, early_loss_predicted)
    recall = recall_score(early_loss_actual, early_loss_predicted)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("precision:{},recall:{},f1:{}".format(acc, precision, recall, f1))

def M_W_M(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset):
    total_len = len(train_clean_bad_set)
    clean_len = len(train_clean_dataset)
    bad_len = len(train_bad_dataset)

    new_train_clean_bad_set = copy.deepcopy(train_clean_bad_set)
    train_clean_bad_set_copy = copy.deepcopy(train_clean_bad_set)
    ground_truth = copy.deepcopy(train_clean_bad_set)

    train_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)

    epoch = 3

    # 总共的correct个数
    total_correct_num = 0
    final_dirty_set = []


    # 第一轮，选出干净池子和脏池子
    model = Models[dataset]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)

    model.train()
    early_loss = np.zeros(total_len, dtype=np.float64)

    for i in range(epoch):
        for data in train_loader:
            # GPU加速
            train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            train_label_predict = model(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function(train_label_predict, train_label)
            train_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            num = 0
            for data in test_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                test_label_predict = test_label_predict.to(device)
                loss = loss_function(test_label_predict, test_label)
                early_loss[num] += loss
                num += 1

    loss_clean = sorted(enumerate(list(early_loss)), key=lambda x: x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
    clean_pool_idx = [x[0] for x in loss_clean]  # 获取排序好后b坐标,下标在第0位

    model_clean = Models[dataset]
    model_clean = model_clean.to(device)
    optimizer_clean = torch.optim.Adam(model_clean.parameters(), lr=0.001)
    loss_function_clean = nn.CrossEntropyLoss()
    loss_function_clean = loss_function_clean.to(device)

    model_clean.train()
    clean_pool = []
    for indeX in range(int(len(train_bad_dataset) / 4)):
        clean_pool.append(train_clean_bad_set[clean_pool_idx[indeX]])
    train_loader_clean = DataLoader(dataset=MyDataSet(clean_pool), batch_size=64, shuffle=True)

    # 训练干净数据集到收敛
    for epo in range(40):
        for data in train_loader_clean:
            # GPU加速
            train_feature_clean, train_label_clean = data
            train_feature_clean = train_feature_clean.to(device)
            train_label_clean = train_label_clean.to(device)
            optimizer_clean.zero_grad()
            train_label_predict_clean = model_clean(train_feature_clean)

            # GPU加速
            train_label_predict_clean = train_label_predict_clean.to(device)
            train_loss_clean = loss_function_clean(train_label_predict_clean, train_label_clean)
            train_loss_clean.backward()
            optimizer_clean.step()

    loss_bad = sorted(enumerate(list(early_loss)), key=lambda x: -x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
    bad_pool_idx = [x[0] for x in loss_bad]  # 获取排序好后b坐标,下标在第0位

    bad_pool_idx = bad_pool_idx[:int(len(train_bad_dataset) / 4) + 10]
    influence_bad = []
    model_clean.eval()
    for indeX in bad_pool_idx:
        bad_test = [train_clean_bad_set[indeX]]
        test_loader_bad = DataLoader(dataset=MyDataSet(bad_test), batch_size=1, shuffle=False)
        for data in test_loader_bad:
            train_feature_bad, train_label_bad = data
            train_feature_bad = train_feature_bad.to(device)
            train_label_bad = train_label_bad.to(device)
            train_label_predict_bad = model_clean(train_feature_bad)

            # GPU加速
            train_label_predict_bad = train_label_predict_bad.to(device)
            train_loss_bad = loss_function_clean(train_label_predict_bad, train_label_bad)

            train_loss_bad_gradient = torch.autograd.grad(train_loss_bad, model_clean.parameters(), allow_unused=True)
            grad = 0.0
            for item in train_loss_bad_gradient:
                if item == None:
                    continue
                item = torch.norm(item)
                grad += item

            tmp = [indeX, grad]
            influence_bad.append(tmp)

    influence_bad_sorted = sorted(influence_bad, key=lambda x: -x[1])
    influence_bad_idx = [x[0] for x in influence_bad_sorted]  # 获取排序好后b坐标,下标在第0位

    correct_num = 0
    true_bad_detected_idx = []
    detect_idx_50 = []
    early_loss_predicted = np.zeros((total_len, 1), dtype=np.int32)
    early_loss_actual = np.ones((total_len, 1), dtype=np.int32)
    for i in range(clean_len):
        early_loss_actual[i][0] = 0

    # 每轮detect500个
    for i in range(int(len(train_bad_dataset) / 4)):
        detect_idx_50.append(influence_bad_idx[i])
        early_loss_predicted[influence_bad_idx[i]][0] = 1

        if (total_len - 1) >= influence_bad_idx[i] >= clean_len:
            correct_num += 1
            true_bad_detected_idx.append(influence_bad_idx[i])

    total_correct_num += correct_num

    # 计算总的精度
    acc = accuracy_score(early_loss_actual, early_loss_predicted)
    precision = precision_score(early_loss_actual, early_loss_predicted)
    recall = recall_score(early_loss_actual, early_loss_predicted)

    ground_truth_tmp = []
    new_train_clean_bad_set = []
    for i in range(total_len):
        if i not in detect_idx_50:
            new_train_clean_bad_set.append(train_clean_bad_set_copy[i])
            ground_truth_tmp.append(ground_truth[i])
        else:
            final_dirty_set.append(ground_truth[i])
    # for i in range(total_len):
    #     if i not in detect_idx_50:
    #         train_clean_bad_set_tmp.append(train_clean_bad_set_copy[i])
    train_clean_bad_set_copy = copy.deepcopy(new_train_clean_bad_set)
    ground_truth = copy.deepcopy(ground_truth_tmp)

    total_len = len(new_train_clean_bad_set)
    bad_len = bad_len - correct_num
    clean_len = total_len - bad_len

    train_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)


    # 800 / 50 = 16
    for times in range(3):
        model = Models[dataset]
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(device)

        model.train()
        early_loss = np.zeros(total_len, dtype=np.float64)

        if (times * int(len(train_bad_dataset) / 4)) >= 0.2 * len(train_bad_dataset):
            epoch = 8

        for i in range(epoch):
            for data in train_loader:
                # GPU加速
                train_feature, train_label = data
                train_feature = train_feature.to(device)
                train_label = train_label.to(device)
                optimizer.zero_grad()
                train_label_predict = model(train_feature)

                # GPU加速
                train_label_predict = train_label_predict.to(device)
                train_loss = loss_function(train_label_predict, train_label)
                train_loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                num = 0
                for data in test_loader:
                    test_feature, test_label = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model(test_feature)
                    test_label_predict = test_label_predict.to(device)
                    loss = loss_function(test_label_predict, test_label)
                    early_loss[num] += loss
                    num += 1

        loss_bad = sorted(enumerate(list(early_loss)),
                          key=lambda x: -x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
        bad_pool_idx = [x[0] for x in loss_bad]  # 获取排序好后b坐标,下标在第0位

        bad_pool_idx = bad_pool_idx[:int(len(train_bad_dataset) / 4) + 10]
        influence_bad = []
        model_clean.eval()
        for indeX in bad_pool_idx:
            bad_test = [train_clean_bad_set[indeX]]
            test_loader_bad = DataLoader(dataset=MyDataSet(bad_test), batch_size=1, shuffle=False)
            for data in test_loader_bad:
                train_feature_bad, train_label_bad = data
                train_feature_bad = train_feature_bad.to(device)
                train_label_bad = train_label_bad.to(device)
                train_label_predict_bad = model_clean(train_feature_bad)

                # GPU加速
                train_label_predict_bad = train_label_predict_bad.to(device)
                train_loss_bad = loss_function_clean(train_label_predict_bad, train_label_bad)

                train_loss_bad_gradient = torch.autograd.grad(train_loss_bad, model_clean.parameters(), allow_unused=True)

                grad = 0.0
                for item in train_loss_bad_gradient:
                    if item == None:
                        continue
                    item = torch.norm(item)
                    grad += item

                tmp = [indeX, grad]
                influence_bad.append(tmp)

        influence_bad_sorted = sorted(influence_bad, key=lambda x: -x[1])
        influence_bad_idx = [x[0] for x in influence_bad_sorted]  # 获取排序好后b坐标,下标在第0位


        correct_num = 0
        true_bad_detected_idx = []
        detect_idx_50 = []
        early_loss_predicted = np.zeros((total_len, 1), dtype=np.int32)
        early_loss_actual = np.ones((total_len, 1), dtype=np.int32)
        for i in range(clean_len):
            early_loss_actual[i][0] = 0

        # 每轮detect500个
        for i in range(int(len(train_bad_dataset) / 4)):
            detect_idx_50.append(influence_bad_idx[i])
            early_loss_predicted[influence_bad_idx[i]][0] = 1

            if (total_len - 1) >= influence_bad_idx[i] >= clean_len:
                correct_num += 1
                true_bad_detected_idx.append(influence_bad_idx[i])

        total_correct_num += correct_num

        # 计算总的精度
        acc = accuracy_score(early_loss_actual, early_loss_predicted)
        precision = precision_score(early_loss_actual, early_loss_predicted)
        recall = recall_score(early_loss_actual, early_loss_predicted)

        ground_truth_tmp = []
        new_train_clean_bad_set = []
        for i in range(total_len):
            if i not in detect_idx_50:
                new_train_clean_bad_set.append(train_clean_bad_set_copy[i])
                ground_truth_tmp.append(ground_truth[i])
            else:
                final_dirty_set.append(ground_truth[i])
        # for i in range(total_len):
        #     if i not in detect_idx_50:
        #         train_clean_bad_set_tmp.append(train_clean_bad_set_copy[i])
        train_clean_bad_set_copy = copy.deepcopy(new_train_clean_bad_set)
        ground_truth = copy.deepcopy(ground_truth_tmp)

        total_len = len(new_train_clean_bad_set)
        bad_len = bad_len - correct_num
        clean_len = total_len - bad_len

        train_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)

    precision = total_correct_num / (4 * int(len(train_bad_dataset) / 4))
    recall = total_correct_num / len(train_bad_dataset)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("detect_iterate * detect_num:{}".format(4 * int(len(train_bad_dataset) / 4)))
    print("detect precision：{}".format(precision))
    print("detect recall：{}".format(recall))
    print("detect f1 score：{}".format(f1))

def evaluate_all_methods(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset, train_clean_bad_set_ground_truth):
    print("=======================misdetect=========================")
    misdetect(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset, train_clean_bad_set_ground_truth)

    print("=======================knn=========================")
    knn(train_clean_bad_set, train_clean_dataset, train_bad_dataset)

    print("=======================clean_pool=========================")
    clean_pool(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset)

    print("=======================cleanlab=========================")
    cleanlab(train_clean_bad_set, train_clean_dataset, train_bad_dataset)

    print("=======================forget_event=========================")
    forget_event(train_clean_bad_set, train_clean_dataset, train_bad_dataset)

    print("=======================ensemble=========================")
    ensemble(train_clean_bad_set, train_clean_dataset, train_bad_dataset)

    print("=======================coteaching=========================")
    coteaching(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset)

    print("=======================mentornet=========================")
    mentornet(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset)

    print("=======================non_iter=========================")
    non_iter(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset)

    print("=======================M_W_IM=========================")
    M_W_IM(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset)

    print("=======================M_W_M=========================")
    M_W_M(dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset)
