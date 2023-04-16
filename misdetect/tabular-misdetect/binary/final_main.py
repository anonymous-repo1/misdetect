from util import *
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
if __name__ == '__main__':
    # 记录时间
    start_time = time.time()
    coteaching()
    # model_performance()

    # 多轮detect，用influence function, 每轮选XX个(ours)
    # 0.6 with 0.5 and clean 150, bad 100;0.9 with 0.3 and bad_pool_len 75 clean_len 200
    # 0.7 with 0.5 and clean 150, bad 100;0.8 with 0.3 and bad_pool_len 100 clean_len 150
    # detect_with_influence_iteratively(detect_num=int(len(train_bad_dataset) / 4), detect_iterate=4,
    #                                   clean_pool_len=int(len(train_bad_dataset) / 4), bad_pool_len=int(len(train_bad_dataset) / 4) + 10)

    # detect_with_influence_iteratively_on_one_model_with_dynamic_clean(
    #     detect_num=int(len(train_bad_dataset) / 4), detect_iterate=4, clean_pool_len=int(len(train_bad_dataset) / 8),
    #     bad_pool_len=int(len(train_bad_dataset) / 4) + 10)

    # test_score_entropy(detect_num=int(len(train_bad_dataset) / 20), detect_iterate=20)

    # detect_with_influence_iteratively_on_one_model_with_dynamic_clean_and_outlier(
    #     detect_num=int(len(train_bad_dataset) / 4), detect_iterate=4, clean_pool_len=int(len(train_bad_dataset) / 8),
    #     bad_pool_len=int(len(train_bad_dataset) / 4) + 10, confidence_threshold=3)

    # baseline_self_early_loss(
    #     detect_num=int(0.95*(len(train_bad_dataset) / 4)), detect_iterate=4, clean_pool_len=int(len(train_bad_dataset) / 4),
    #     bad_pool_len=int(len(train_bad_dataset) / 3), confidence_threshold=4)

    # detect_with_influence_iteratively_on_one_model_with_dynamic_clean_and_outlier_all_remained(
    #     detect_num=int(len(train_bad_dataset) / 4), detect_iterate=4, clean_pool_len=int(len(train_bad_dataset) / 4),
    #     bad_pool_len=int(len(train_bad_dataset) / 4) + 10, confidence_threshold=3)

    # 0.9 with 300; 0.6 with 100
    # test_clean_pool(len(train_set), len(train_clean_dataset), clean_pool_len=100)

    # KNN
    # KNN_sklearn(len(train_clean_bad_set), len(train_clean_dataset))

    # NCN_sklearn(len(train_clean_bad_set), len(train_clean_dataset))

    # Ensemble
    # Ensemble(len(train_clean_bad_set), len(train_clean_dataset))

    # clean lab
    # cleanLab(len(train_clean_bad_set), len(train_clean_dataset))

    # Training Set Debugging Using Trusted Items
    # DUTI(clean_pool_len=3, iterate_num=20)

    # partition filter
    # partition_filter(len(train_clean_bad_set), interval=int(len(train_clean_bad_set) / 10), step=10)

    # self-ensemble label filtering
    # SELF(len(train_clean_bad_set))

    # noisy cross-validation(NCV)
    # noisy_cross_validation(len(train_clean_bad_set), split_ratio=0.5, epoch=5)

    # clean_pool
    # clean_pool(clean_pool_len=int(len(train_clean_dataset) / 2))
    # ==========================================跟自己比=========================================
    # early loss
    # early_loss(len(train_set), len(train_clean_dataset), len(train_bad_dataset), epoch=1)

    # without dirty pool
    # without_dirty_pool(detect_num=len(train_bad_dataset), clean_pool_len=1000)

    # 多轮detect，每轮选XX个
    # without_influence(detect_num=50, detect_iterate=int(len(train_bad_dataset) / 50))

    # show picture
    # early_loss_sorted(epoch=1)
    end_time = time.time()
    print("GPU time:{}".format(end_time - start_time))
