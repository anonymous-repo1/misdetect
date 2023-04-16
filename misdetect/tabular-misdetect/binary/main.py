import torch.optim
from torch import nn
from train import *
import time

loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
if __name__ == '__main__':
    logs_acc = "acc_logs"
    logs_early_loss = "early_loss_logs"
    logs_early_loss_about_ratio = "early_loss_about_ratio_logs"
    logs_flip_label = "flip_label_logs"
    logs_flip_label_loop = "flip_label_loop_logs"
    logs_outlier_detection = "outlier_detection_logs"
    logs_get_detect_model = "get_detect_model_logs"

    train_loader_initial = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader_initial = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=1, shuffle=False)
    train_clean_dataset_loader = DataLoader(dataset=MyDataSet(train_clean_dataset), batch_size=64, shuffle=True)
    train_bad_dataset_loader = DataLoader(dataset=MyDataSet(train_bad_dataset), batch_size=64, shuffle=True)
    model = Model()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 记录时间
    start_time = time.time()

    # acc
    # train_acc(model, train_loader_initial, train_clean_dataset_loader, train_bad_dataset_loader, len(train_set),
    #           len(train_clean_dataset), len(train_bad_dataset), epoch=100, optimizer=optimizer,
    #           loss_function=loss_function, logs_path=logs_acc)

    # early loss
    # train_early_loss(model, train_loader_initial, test_loader_initial, epoch=1, optimizer=optimizer,
    #                  loss_function=loss_function, logs_path=logs_early_loss)

    # early loss about ratio
    # train_early_loss_about_ratio(epoch=1, loss_function=loss_function, logs_path=logs_early_loss_about_ratio)

    # sensor 这个做了(SGD和Adam,flip差别大)
    # early loss准确率和flip准确率(0.467 and 0.99 with good_ratio 0.6) (0.399 and 0.99 with good_ratio 0.7)
    # (0.3264 and 1.0 with good_ratio 0.8) (0.04 and 0.887 with good_ratio 0.9)
    # precision_and_recall(model, train_loader_initial, test_loader_initial, len(train_set),
    #                      len(train_clean_dataset), len(train_bad_dataset),
    #                      epoch=1, optimizer=optimizer, loss_function=loss_function)

    # early loss准确率with multiple epoch (0.76905 with 3 epoch and 0.7744 with 8 epoch in good_ratio 0.6)
    # (0.6382 with 3 epoch and 0.6606 with 8 epoch in good_ratio 0.9)
    # early_loss_with_multiple_epoch(model, train_loader_initial, test_loader_initial,
    #                                len(train_set), len(train_clean_dataset), len(train_bad_dataset),
    #                                epoch=3, optimizer=optimizer, loss_function=loss_function)

    # flip label(include retrain)
    # flip_label(model, train_loader_initial, len_random=100, epoch=1, optimizer=optimizer,
    #            loss_function=loss_function, logs_path=logs_flip_label)

    # flip label loop(no retrain)
    # flip_label_loop(model, train_loader_initial, len_random=1000, epoch=1, optimizer=optimizer,
    #                 loss_function=loss_function, logs_path=logs_flip_label_loop)

    # outlier detection
    # outlier_detection(model, train_loader_initial, len_random=100, epoch=1, optimizer=optimizer,
    #                   loss_function=loss_function, logs_path=logs_outlier_detection)

    # judge_outlier(1152, train_clean_bad_set_outlier)

    # get detect model(0。67595 with 3 epoch and 0.60845 with 8 epoch in good_ratio 0.6)
    # get_detect_model(model, len(train_set), len(train_clean_dataset), len(train_bad_dataset), loss_ratio=0.05,
    #                  epoch=3, optimizer=optimizer, loss_function=loss_function, logs_path=logs_get_detect_model)

    ensemble_validation(len(train_set), len(train_clean_dataset), len(train_bad_dataset), epoch=100)

    end_time = time.time()
    print("GPU time:{}".format(end_time - start_time))
