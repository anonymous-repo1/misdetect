import torch
from torch.utils.tensorboard import SummaryWriter
from get_data import *
import numpy as np
from torch.utils.data import DataLoader
from model import Model
import decimal
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from pyod.models.knn import KNN
from sklearn.neighbors import LocalOutlierFactor as LOF

# GPU
device = torch.device("mps")


# 调整学习率
def adjust_learning_rate(optimizer, epoch):
    lr = 1e-3 * 0.1 ** (epoch // 4)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 对比clean和bad数据的准确率
def train_acc(model, train_loader, train_loader_clean, train_loader_bad, total_len,
              train_clean_len, train_bad_len, epoch, optimizer, loss_function, logs_path):
    writer = SummaryWriter(logs_path)
    model.train()

    for i in range(epoch):
        train_accuracy = 0
        for data in train_loader:
            train_feature, train_label = data
            # GPU加速
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)

            optimizer.zero_grad()
            train_label_predict = model(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function(train_label_predict, train_label)
            train_loss.backward()
            train_accuracy_num = (train_label_predict.argmax(1) == train_label).sum()
            optimizer.step()

            train_accuracy += train_accuracy_num
        # writer.add_scalar("train_acc", train_accuracy / len(train_dataset), i)
        # print("第{}轮准确率:{}".format(i, train_accuracy / len(train_dataset)))

        model.eval()
        with torch.no_grad():
            test_accuracy_clean = 0
            for data in train_loader_clean:
                test_feature_clean, test_label_clean = data
                test_feature_clean = test_feature_clean.to(device)
                test_label_clean = test_label_clean.to(device)
                test_label_predict_clean = model(test_feature_clean)
                test_accuracy_num_clean = (test_label_predict_clean.argmax(1) == test_label_clean).sum()
                test_accuracy_clean += test_accuracy_num_clean
        model.eval()
        with torch.no_grad():
            test_accuracy_bad = 0
            for data in train_loader_bad:
                test_feature_bad, test_label_bad = data
                test_feature_bad = test_feature_bad.to(device)
                test_label_bad = test_label_bad.to(device)
                test_label_predict_bad = model(test_feature_bad)
                test_accuracy_num_bad = (test_label_predict_bad.argmax(1) == test_label_bad).sum()
                test_accuracy_bad += test_accuracy_num_bad
        model.eval()
        with torch.no_grad():
            test_accuracy_total = 0
            for data in train_loader:
                test_feature_total, test_label_total = data
                test_feature_total = test_feature_total.to(device)
                test_label_total = test_label_total.to(device)
                test_label_predict_total = model(test_feature_total)
                test_accuracy_num_total = (test_label_predict_total.argmax(1) == test_label_total).sum()
                test_accuracy_total += test_accuracy_num_total
        writer.add_scalars("test_acc", {"clean": test_accuracy_clean / train_clean_len,
                                        "bad": test_accuracy_bad / train_bad_len,
                                        "total": test_accuracy_total / total_len}, i)

        print("第{}轮clean准确率:{},bad准确率:{},total准确率:{}".format(
            i, test_accuracy_clean / train_clean_len,
               test_accuracy_bad / train_bad_len,
               test_accuracy_total / total_len))
    writer.close()


# 求第一个epoch的loss
def train_early_loss(model, train_loader, test_loader, epoch, optimizer, loss_function, logs_path):
    writer = SummaryWriter(logs_path)
    model.train()

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
        num = 0
        early_loss = []
        with torch.no_grad():
            for data in test_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                test_label_predict = test_label_predict.to(device)
                loss = loss_function(test_label_predict, test_label)
                early_loss.append(loss)
                writer.add_scalar("test_loss", loss, num)
                num += 1
    writer.close()


# early loss准确率
def precision_and_recall(model, train_loader, test_loader, total_len, clean_len, bad_len,
                         epoch, optimizer, loss_function):
    model.train()
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
        early_loss = []
        with torch.no_grad():
            for data in test_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                test_label_predict = test_label_predict.to(device)
                loss = loss_function(test_label_predict, test_label)
                early_loss.append(loss)

        b = sorted(enumerate(early_loss), key=lambda x: -x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
        c = [x[0] for x in b]  # 获取排序好后b坐标,下标在第0位

        correct_num = 0
        bad_detected_idx = []
        early_loss_predicted = np.zeros((total_len, 1), dtype=np.int)
        early_loss_actual = np.ones((total_len, 1), dtype=np.int)
        for i in range(clean_len):
            early_loss_actual[i][0] = 0

        for i in range(bad_len):
            early_loss_predicted[c[i]][0] = 1
            if (total_len - 1) >= c[i] >= (clean_len - 1):
                correct_num += 1
                bad_detected_idx.append(c[i])
        print("loss最高的脏数据占比为:{}".format(correct_num / bad_len))

        # 计算总的精度
        acc = accuracy_score(early_loss_actual, early_loss_predicted)
        precision = precision_score(early_loss_actual, early_loss_predicted)
        recall = recall_score(early_loss_actual, early_loss_predicted)
        print("early loss detection: acc:{},precision:{},recall:{}".format(acc, precision, recall))

        len_bad = len(bad_detected_idx)
        print(len_bad)

        flip_test_set = []
        flip_test_set_after_flip = []
        index_rand = []
        # 求脏label和改成正确lebel之后的测试的dataloader
        for idx in range(len_bad):
            index_rand.append(bad_detected_idx[idx])
            flip_test_set.append(train_set[bad_detected_idx[idx]])
            flip_test_set_after_flip.append(train_set_tmp[bad_detected_idx[idx]])

        # index_rand.sort()
        # print(index_rand)
        # print("=============================================================")
        # bad_idx_array.sort()
        # print(bad_idx_array)
        # print(index_rand == bad_idx_array)
        # print(len(index_rand), len(bad_idx_array))
        # print(judge_repeated(bad_idx_array))
        # print(judge_repeated(index_rand))
        # exit()

        flip_loader = DataLoader(MyDataSet(flip_test_set), batch_size=1, shuffle=False)
        flip_loader_after_flip = DataLoader(MyDataSet(flip_test_set_after_flip), batch_size=1, shuffle=False)

        model.eval()
        num = 0
        flip_loss = np.zeros((len_bad, 2), dtype=np.float)
        with torch.no_grad():
            for data in flip_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                loss = loss_function(test_label_predict, test_label)
                flip_loss[num][0] = loss
                flip_loss[num][1] = test_label
                num += 1

        # 求循环flip label之后的dataloader
        loop_num = 0
        for sum in range(len(classes) - 2):
            flip_tmp_set = []
            for i in index_rand:
                if (train_set[i][1] + 1) % len(classes) != train_set_tmp[i][1]:
                    train_set[i][1] = (train_set[i][1] + 1) % len(classes)
                    flip_tmp_set.append(train_set[i])
                else:
                    train_set[i][1] = (train_set[i][1] + 2) % len(classes)
                    flip_tmp_set.append(train_set[i])
            flip_loader_tmp = DataLoader(MyDataSet(flip_tmp_set), batch_size=1, shuffle=False)
            num2 = 0
            with torch.no_grad():
                for data in flip_loader_tmp:
                    test_feature, test_label = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model(test_feature)
                    loss = loss_function(test_label_predict, test_label)
                    if flip_loss[num2][0] > loss:
                        flip_loss[num2][0] = loss
                        flip_loss[num2][1] = test_label
                    num2 += 1
            print(loop_num)
            loop_num += 1

        # for i in index_rand:
        #     train_set[i][1] = train_set_tmp[i][1]

        num2 = 0
        flip_correct_num = 0
        with torch.no_grad():
            for data in flip_loader_after_flip:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                loss = loss_function(test_label_predict, test_label)
                if flip_loss[num2][0] > loss:
                    flip_loss[num2][0] = loss
                    flip_loss[num2][1] = test_label
                    flip_correct_num += 1
                    if test_label_predict.argmax(1) != test_label:
                        print("///////////////////////////////////////////////////////////")
                num2 += 1
        print(len_bad, len(flip_loss), len(flip_test_set_after_flip), flip_correct_num)
        # flip_correct_num = 0
        # for i in range(len_random):
        #     if flip_loss[i][1] == flip_test_set_after_flip[i][1]:
        #         flip_correct_num += 1
        print("flip后loss下降最大的是正确的label的占比为:{}".format(flip_correct_num / len_bad))


# early loss准确率
def early_loss_with_multiple_epoch(model, train_loader, test_loader, total_len, clean_len, bad_len,
                                   epoch, optimizer, loss_function):
    model.train()
    early_loss = np.zeros(total_len, dtype=np.float)
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
    b = sorted(enumerate(list(early_loss)), key=lambda x: -x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
    c = [x[0] for x in b]  # 获取排序好后b坐标,下标在第0位

    correct_num = 0
    bad_detected_idx = []
    early_loss_predicted = np.zeros((total_len, 1), dtype=np.int)
    early_loss_actual = np.ones((total_len, 1), dtype=np.int)
    for i in range(clean_len):
        early_loss_actual[i][0] = 0

    for i in range(bad_len):
        early_loss_predicted[c[i]][0] = 1
        if (total_len - 1) >= c[i] >= (clean_len - 1):
            correct_num += 1
            bad_detected_idx.append(c[i])
    print("loss最高的脏数据占比为:{}".format(correct_num / bad_len))

    # 计算总的精度
    acc = accuracy_score(early_loss_actual, early_loss_predicted)
    precision = precision_score(early_loss_actual, early_loss_predicted)
    recall = recall_score(early_loss_actual, early_loss_predicted)
    print("early loss detection: acc:{},precision:{},recall:{}".format(acc, precision, recall))

    len_bad = len(bad_detected_idx)
    print(len_bad)


# 求early loss跟脏数据比例的关系
def train_early_loss_about_ratio(epoch, loss_function, logs_path):
    writer = SummaryWriter(logs_path)
    # 控制浮点数计算精度
    ratio = decimal.Decimal(0.0)
    interval = decimal.Decimal(0.1)
    while ratio <= 1.08:
        train_set2 = []
        for i in train_set_tmp:
            train_set2.append(list(i))

        cnt_label2 = {}
        for idx, tensor in enumerate(train_set2):
            cnt_label2[tensor[1]] = cnt_label2.get(tensor[1], 0) + 1

        cnt_good_label_tgt2 = {}
        for k, v in cnt_label2.items():
            cnt_good_label_tgt2[k] = int(v * ratio)

        manipulate_label2 = {}
        good_idx_set2 = []
        for idx, tensor in enumerate(train_set2):
            manipulate_label2[tensor[1]] = manipulate_label2.get(tensor[1], 0) + 1
            if manipulate_label2[tensor[1]] > cnt_good_label_tgt2[tensor[1]]:
                p = np.random.randint(0, len(classes))
                while True:
                    if p != tensor[1]:
                        train_set2[idx][1] = p
                        break
                    p = np.random.randint(0, len(classes))
            else:
                good_idx_set2.append(idx)
        good_idx_array2 = np.array(good_idx_set2)
        all_idx_array2 = np.arange(len(train_set2))
        bad_idx_array2 = np.setdiff1d(all_idx_array2, good_idx_array2)
        train_clean_dataset2 = []
        for i in good_idx_array2:
            train_clean_dataset2.append(train_set2[i])
            if train_set2[i][1] != train_set_tmp[i][1]:
                print("--------------------------------")
        train_bad_dataset2 = []
        for i in bad_idx_array2:
            train_bad_dataset2.append(train_set2[i])
            if train_set2[i][1] == train_set_tmp[i][1]:
                print("--------------------------------")

        train_clean_bad_set2 = train_clean_dataset2 + train_bad_dataset2
        print(len(train_clean_dataset2), len(train_bad_dataset2), len(train_clean_bad_set2))
        train_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set2), batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set2), batch_size=1, shuffle=False)

        model = Model()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for i in range(epoch):
            model.train()
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
                    loss = loss_function(test_label_predict, test_label)
                    writer.add_scalar("test_loss_{}".format(len(train_clean_dataset2)), loss, num)
                    num += 1
        ratio += interval
    writer.close()


# 看flip成正确label后在同个模型再训练一遍模型得到的loss变化,和不retrain得到的loss变化
def flip_label(model, train_loader, len_random, epoch, optimizer, loss_function, logs_path):
    writer = SummaryWriter(logs_path)
    model.train()

    flip_test_set = []
    flip_test_set_after_flip = []
    index_rand = []
    while len(index_rand) < len_random:
        rand_num = np.random.randint(0, len(bad_idx_array))
        if rand_num not in index_rand:
            index_rand.append(bad_idx_array[rand_num])
            flip_test_set.append(train_set[bad_idx_array[rand_num]])
            flip_test_set_after_flip.append(train_set_tmp[bad_idx_array[rand_num]])
            print("before:{}, after:{}".format(train_set[bad_idx_array[rand_num]][1],
                                               train_set_tmp[bad_idx_array[rand_num]][1]))

    flip_loader = DataLoader(MyDataSet(flip_test_set), batch_size=1, shuffle=False)
    flip_loader_after_flip = DataLoader(MyDataSet(flip_test_set_after_flip), batch_size=1, shuffle=False)
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
        num = 0
        with torch.no_grad():
            for data in flip_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                loss = loss_function(test_label_predict, test_label)
                writer.add_scalars("test_loss", {"flip_before": loss}, num)
                num += 1

        # flip成正确的label之后再测试一遍loss
        num2 = 0
        with torch.no_grad():
            for data in flip_loader_after_flip:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                loss = loss_function(test_label_predict, test_label)
                writer.add_scalars("test_loss", {"flip_after": loss}, num2)
                num2 += 1

    train_clean_dataset_tmp = []
    train_bad_dataset2 = []
    for i in bad_idx_array:
        if i not in index_rand:
            train_bad_dataset2.append(train_set[i])
        else:
            train_clean_dataset_tmp.append((train_set_tmp[i]))
    train_clean_dataset2 = train_clean_dataset + train_clean_dataset_tmp
    print(len(train_clean_dataset_tmp))
    print(len(train_bad_dataset2), len(train_clean_dataset2))
    train_loader_after_flip = DataLoader(MyDataSet(train_clean_dataset2 + train_bad_dataset2),
                                         batch_size=64, shuffle=True)
    model.train()
    for i in range(epoch):
        for data in train_loader_after_flip:
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
        num2 = 0
        with torch.no_grad():
            for data in flip_loader_after_flip:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                loss = loss_function(test_label_predict, test_label)
                writer.add_scalars("test_loss", {"flip_retrain": loss}, num2)
                num2 += 1

    writer.close()


# 看遍历所有flip成正确label后不再训练模型得到的loss变化
def flip_label_loop(model, train_loader, len_random, epoch, optimizer, loss_function, logs_path):
    writer = SummaryWriter(logs_path)
    model.train()

    flip_test_set = []
    flip_test_set_after_flip = []
    index_rand = []
    # 求脏label和改成正确lebel之后的测试的dataloader
    while len(index_rand) < len_random:
        rand_num = np.random.randint(0, len(bad_idx_array))
        if rand_num not in index_rand:
            index_rand.append(bad_idx_array[rand_num])
            flip_test_set.append(train_set[bad_idx_array[rand_num]])
            flip_test_set_after_flip.append(train_set_tmp[bad_idx_array[rand_num]])
    flip_loader = DataLoader(MyDataSet(flip_test_set), batch_size=1, shuffle=False)
    flip_loader_after_flip = DataLoader(MyDataSet(flip_test_set_after_flip), batch_size=1, shuffle=False)
    for index_num in range(epoch):
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
            for data in flip_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                loss = loss_function(test_label_predict, test_label)
                writer.add_scalars("test_loss", {"flip_before": loss}, num)
                num += 1

        # 求循环flip label之后的dataloader
        loop_num = 0
        for sum in range(len(classes) - 2):
            flip_tmp_set = []
            for i in index_rand:
                if (train_set[i][1] + 1) % len(classes) != train_set_tmp[i][1]:
                    print("before:{}, after:{}, correct:{}".format(train_set[i][1], (train_set[i][1] + 1) % len(classes),
                                                                   train_set_tmp[i][1]))
                    train_set[i][1] = (train_set[i][1] + 1) % len(classes)
                    flip_tmp_set.append(train_set[i])
                else:
                    print("before:{}, after:{}, correct:{}".format(train_set[i][1], (train_set[i][1] + 2) % len(classes),
                                                                   train_set_tmp[i][1]))
                    train_set[i][1] = (train_set[i][1] + 2) % len(classes)
                    flip_tmp_set.append(train_set[i])
            print("+++++++++++++++++++++++++++++++++")
            flip_loader_tmp = DataLoader(MyDataSet(flip_tmp_set), batch_size=1, shuffle=False)
            num2 = 0
            with torch.no_grad():
                for data in flip_loader_tmp:
                    test_feature, test_label = data
                    print(test_label.item())
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model(test_feature)
                    loss = loss_function(test_label_predict, test_label)
                    writer.add_scalars("test_loss", {"flip_{}".format(loop_num): loss}, num2)
                    num2 += 1
            print("---------------------------")
            print(loop_num)
            loop_num += 1

        for i in index_rand:
            print("before:{}, after:{}".format(train_set[i][1], train_set_tmp[i][1]))
            train_set[i][1] = train_set_tmp[i][1]

        num2 = 0
        with torch.no_grad():
            for data in flip_loader_after_flip:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                loss = loss_function(test_label_predict, test_label)
                writer.add_scalars("test_loss", {"flip_after": loss}, num2)
                num2 += 1
    writer.close()


def judge_repeated(array):
    if len(set(array)) == len(array):
        return False
    else:
        return True


# 看flip成正确label后loss变大的样本是不是outlier
def outlier_detection(model, train_loader, len_random, epoch, optimizer, loss_function, logs_path):
    writer = SummaryWriter(logs_path)
    model.train()

    flip_test_set = []
    flip_test_set_after_flip = []
    index_rand = []
    while len(index_rand) < len_random:
        rand_num = np.random.randint(0, len(bad_idx_array))
        if rand_num not in index_rand:
            index_rand.append(bad_idx_array[rand_num])
            flip_test_set.append(train_set[bad_idx_array[rand_num]])
            flip_test_set_after_flip.append(train_set_tmp[bad_idx_array[rand_num]])
            print("before:{}, after:{}".format(train_set[bad_idx_array[rand_num]][1],
                                               train_set_tmp[bad_idx_array[rand_num]][1]))

    flip_loader = DataLoader(MyDataSet(flip_test_set), batch_size=1, shuffle=False)
    flip_loader_after_flip = DataLoader(MyDataSet(flip_test_set_after_flip), batch_size=1, shuffle=False)
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
        num = 0
        loss_before = []
        with torch.no_grad():
            for data in flip_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                loss = loss_function(test_label_predict, test_label)
                loss_before.append(loss)
                writer.add_scalars("test_loss", {"flip_before": loss}, num)
                num += 1

        # flip成正确的label之后再测试一遍loss
        num2 = 0
        loss_after = []
        with torch.no_grad():
            for data in flip_loader_after_flip:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                loss = loss_function(test_label_predict, test_label)
                loss_after.append(loss)
                writer.add_scalars("test_loss", {"flip_after": loss}, num2)
                num2 += 1

        outlier_index = []
        print(len(loss_before), "-----", len(loss_after), "[][][][][][]", len_random)
        for i in range(len(loss_before)):
            if loss_before[i] < loss_after[i]:
                outlier_index.append(index_rand[i])
                judge_outlier(index_rand[i])
                print(index_rand[i])
                # writer.add_image("outlier", train_set_tmp[index_rand[i]][0], step)

    writer.close()


def judge_outlier(index):
    # 训练一个kNN检测器 初始化检测器clf
    # clf = KNN_pipe()

    clf = LOF(n_neighbors=2)

    featurn_need = train_set[index][0]
    label_need = train_set[index][1]
    idx_need = 0
    X_train = []
    for idx, tensor in enumerate(train_set):
        if tensor[1] == label_need:
            X_train.append(tensor[0].numpy())
        if tensor[0].equal(featurn_need):
            idx_need = idx
    # 使用X_train训练检测器clf
    # clf.fit(X_train)
    X_train = np.array(X_train).reshape(-1, 1)
    # X_train = torch.tensor([item.detach().cpu().numpy() for item in X_train]).cuda()
    X_train = X_train.to(device)
    print("kaishi")
    # X_train = torch.tensor([item.detach().numpy() for item in X_train])
    predict = clf.fit_predict(X_train)
    # 返回训练数据X_train上的异常标签和异常分值
    # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
    # predict = clf.labels_
    print("是否为异常值：{}".format(predict[idx_need]))

    # 返回训练数据上的异常值 (分值越大越异常)
    # y_train_scores = clf.decision_scores_

    # 用训练好的clf来预测未知数据中的异常值
    # y_test_pred = clf.predict(X_test)
    # y_test_scores = clf.decision_function(X_test)


# 迭代每次取5%error，flip之后，放回去重新训练，得到最终模型来detect
def get_detect_model(model, total_len, clean_len, bad_len, loss_ratio, epoch, optimizer, loss_function, logs_path):
    writer = SummaryWriter(logs_path)
    model.train()
    test_loader_final = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=1, shuffle=False)

    for i in range(epoch):
        train_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=1, shuffle=False)
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
        early_loss = []
        with torch.no_grad():
            for data in test_loader:
                test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model(test_feature)
                loss = loss_function(test_label_predict, test_label)
                early_loss.append(loss)

        b = sorted(enumerate(early_loss), key=lambda x: -x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
        c = [x[0] for x in b]  # 获取排序好后b坐标,下标在第0位

        bad_detect_set = c[:int(loss_ratio * total_len)]
        print(len(bad_detect_set), "[][][][]", len(early_loss), "[][][][]", len(train_clean_bad_set))
        for index in bad_detect_set:
            loss_tmp = early_loss[index]
            label_tmp = train_clean_bad_set[index][1]

            tmp_dataset = []
            tmp_data = train_clean_bad_set[index]
            for sum in range(len(classes) - 1):
                tmp_data[1] = (tmp_data[1] + 1) % len(classes)
                # print(label_tmp, "after flip:{}".format(tmp_data[1]))
                tmp_dataset.append(tmp_data)

            model.eval()
            flip_loader = DataLoader(MyDataSet(tmp_dataset), batch_size=1, shuffle=False)
            with torch.no_grad():
                for data in flip_loader:
                    feature, label = data
                    feature = feature.to(device)
                    label = label.to(device)
                    label_predict = model(feature)
                    loss = loss_function(label_predict, label)
                    if loss < loss_tmp:
                        loss_tmp = loss
                        label_tmp = label.item()

            train_clean_bad_set[index][1] = label_tmp

        print(len(train_clean_bad_set))


    model.eval()
    early_loss_final = []
    with torch.no_grad():
        num = 0
        for data in test_loader_final:
            test_feature, test_label = data
            test_feature = test_feature.to(device)
            test_label = test_label.to(device)
            test_label_predict = model(test_feature)
            loss = loss_function(test_label_predict, test_label)
            writer.add_scalar("test_loss", loss, num)
            early_loss_final.append(loss)
            num += 1
    b = sorted(enumerate(list(early_loss_final)),
               key=lambda x: -x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
    c = [x[0] for x in b]  # 获取排序好后b坐标,下标在第0位

    correct_num = 0
    bad_detected_idx = []
    early_loss_predicted = np.zeros((total_len, 1), dtype=np.int)
    early_loss_actual = np.ones((total_len, 1), dtype=np.int)
    for i in range(clean_len):
        early_loss_actual[i][0] = 0

    for i in range(bad_len):
        early_loss_predicted[c[i]][0] = 1
        if (total_len - 1) >= c[i] >= (clean_len - 1):
            correct_num += 1
            bad_detected_idx.append(c[i])
    print("loss最高的脏数据占比为:{}".format(correct_num / bad_len))

    # 计算总的精度
    acc = accuracy_score(early_loss_actual, early_loss_predicted)
    precision = precision_score(early_loss_actual, early_loss_predicted)
    recall = recall_score(early_loss_actual, early_loss_predicted)
    print("early loss detection: acc:{},precision:{},recall:{}".format(acc, precision, recall))

    len_bad = len(bad_detected_idx)
    print(len_bad)
    writer.close()