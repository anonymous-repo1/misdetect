from torch import nn
from model import Model
from train import *
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
if __name__ == '__main__':
    logs_acc = "acc_logs2"
    logs_early_loss = "early_loss_logs_3epo"
    logs_early_loss_about_ratio = "early_loss_about_ratio_logs4"
    logs_flip_label = "flip_label_logs"
    logs_flip_label_loop = "flip_label_loop_logs"
    logs_outlier_detection = "outlier_detection_logs"
    logs_get_detect_model = "get_detect_model_logs"

    train_loader_initial = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=256, shuffle=True)
    test_loader_initial = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=1, shuffle=False)
    train_clean_dataset_loader = DataLoader(dataset=MyDataSet(train_clean_dataset), batch_size=64, shuffle=True)
    train_bad_dataset_loader = DataLoader(dataset=MyDataSet(train_bad_dataset), batch_size=64, shuffle=True)
    model = Model()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 记录时间
    start_time = time.time()

    # acc
    # train_acc(model, train_loader_initial, train_clean_dataset_loader, train_bad_dataset_loader, len(train_set),
    #           len(train_clean_dataset), len(train_bad_dataset), epoch=100, optimizer=optimizer,
    #           loss_function=loss_function, logs_path=logs_acc)

    # early loss
    # train_early_loss(model, train_loader_initial, test_loader_initial, epoch=3, optimizer=optimizer,
    #                  loss_function=loss_function, logs_path=logs_early_loss)

    # early loss about ratio
    # train_early_loss_about_ratio(epoch=20, loss_function=loss_function, logs_path=logs_early_loss_about_ratio)

    # early loss准确率和flip准确率(0.73 and 0.47 with good_ratio 0.6) (0.698 and 0.5137 with good_ratio 0.7)
    # (0.65 and 0.55 with good_ratio 0.8) (0.57 and 0.61 with good_ratio 0.9)
    # 如果把训练的batch size改成1，则early loss=0.47 and flip label = 0.1657
    # 如果把训练的batch size改成128，则early loss=0.74 and flip label = 0.51
    # precision_and_recall(model, train_loader_initial, test_loader_initial, len(train_set),
    #                      len(train_clean_dataset), len(train_bad_dataset),
    #                      epoch=1, optimizer=optimizer, loss_function=loss_function)

    # detect_with_3_epoch(model, train_loader_initial, test_loader_initial, len(train_set),
    #                     len(train_clean_dataset), len(train_bad_dataset),
    #                     epoch=1, optimizer=optimizer, loss_function=loss_function)

    # early loss准确率with multiple epoch (0.77 and 0.55 with 3 epoch and 0.7744 and 0.47 with 8 epoch in good_ratio 0.6)
    # (0.6382 with 3 epoch and 0.6606 with 8 epoch in good_ratio 0.9)
    # early_loss_and_flip_with_multiple_epoch(model, train_loader_initial, test_loader_initial,
    #                                         len(train_set), len(train_clean_dataset), len(train_bad_dataset),
    #                                         epoch=8, optimizer=optimizer, loss_function=loss_function)

    # flip label(include retrain)
    # flip_label(model, train_loader_initial, len_random=100, epoch=1, optimizer=optimizer,
    #            loss_function=loss_function, logs_path=logs_flip_label)

    # flip label loop(no retrain)
    # flip_label_loop(model, train_loader_initial, len_random=1000, epoch=1, optimizer=optimizer,
    #                 loss_function=loss_function, logs_path=logs_flip_label_loop)

    # outlier detection
    # outlier_detection(model, train_loader_initial, len_random=100, epoch=1, optimizer=optimizer,
    #                   loss_function=loss_function, logs_path=logs_outlier_detection)

    # get detect model(0.67595 with 3 epoch and 0.60845 with 8 epoch in good_ratio 0.6)
    # get_detect_model(len(train_set), len(train_clean_dataset), len(train_bad_dataset), loss_ratio=0.002,
    #                  epoch=10, loss_function=loss_function, logs_path=logs_get_detect_model)


    # get_detect_model_myself(len(train_set), len(train_clean_dataset), len(train_bad_dataset), loss_ratio=0.002,
    #                         epoch=10, loss_function=loss_function)

    # 多轮detect，每轮选XX个
    # 0.6 with 0.6;  0.9 with 0.3
    # detect_with_multiple_epoch(detect_num=500)

    # 2000
    # detect_with_single_class(2000)

    # 多个模型取平均
    # early_loss_with_mean(1, len(train_set), len(train_clean_dataset), len(train_bad_dataset))

    # 多个模型投票决定
    # detect_with_vote(10, len(train_set), len(train_clean_dataset), len(train_bad_dataset))

    # 多轮detect，用influence function, 每轮选XX个
    # 0.6 with 0.5;  0.9 with 0.3; 0.7 with 0.5; 0.8 with 0.3
    # detect_with_influence_iteratively(detect_num=500, detect_iterate=int(len(train_bad_dataset) / 500),
    #                                   clean_pool_len=2000, bad_pool_len=500)

    # test_clean_pool(len(train_set), len(train_clean_dataset), clean_pool_len=2000)

    # sorted early loss
    # early_loss_sorted(model, train_loader_initial, test_loader_initial, epoch=1, optimizer=optimizer,
    #                   loss_function=loss_function, logs_path=logs_early_loss)

    ensemble_validation4(len(train_set), len(train_clean_dataset), len(train_bad_dataset), epoch=15)

    end_time = time.time()
    print("GPU time:{}".format(end_time - start_time))
