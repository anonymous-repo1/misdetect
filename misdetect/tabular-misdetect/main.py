import math
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import pandas
import torch
from get_data_tabular import *
from model import *
from util import *
import math
import argparse

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Description of your script.")

    # 添加参数
    parser.add_argument("dataset", help="Select the desired dataset")
    parser.add_argument("mislabel_ratio", help="Specify the proportion of mislabels in the dataset")
    parser.add_argument("mislabel_distribution", help="Specify the distribution of mislabels in the dataset")
    parser.add_argument("method", help="Specify the method you want to conduct the experiment on")



    # 解析命令行参数
    args = parser.parse_args()

    # mis injection
    train_clean_dataset, train_bad_dataset, train_clean_bad_set, train_clean_bad_set_ground_truth = mis_injection(args.dataset, args.mislabel_ratio, args.mislabel_distribution)

    if args.method == "misdetect":
        misdetect(args.dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset, train_clean_bad_set_ground_truth)
    elif args.method == "knn":
        knn(train_clean_bad_set, train_clean_dataset, train_bad_dataset)
    elif args.method == "clean_pool":
        clean_pool(args.dataset, int(len(train_clean_dataset) / 2), train_clean_bad_set, train_clean_dataset, train_bad_dataset)
    elif args.method == "cleanlab":
        cleanlab(train_clean_bad_set, train_clean_dataset, train_bad_dataset)
    elif args.method == "forget_event":
        forget_event(train_clean_bad_set, train_clean_dataset, train_bad_dataset)
    elif args.method == "ensemble":
        ensemble(train_clean_bad_set, train_clean_dataset, train_bad_dataset)
    elif args.method == "coteaching":
        coteaching(args.dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset)
    elif args.method == "mentornet":
        mentornet(args.dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset)
    elif args.method == "non_iter":
        non_iter(args.dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset)
    elif args.method == "M_W_IM":
        M_W_IM(args.dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset)
    elif args.method == "M_W_M":
        M_W_M(args.dataset, train_clean_bad_set, train_clean_dataset, train_bad_dataset)