import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from model import *

class MyDataSet(Dataset):
    def __init__(self, loaded_data):
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

classes = {"mnist": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
           "kmnist": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
           "fashion-mnist": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
           "cifar100": ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                       'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                       'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                       'bottles', 'bowls', 'cans', 'cups', 'plates',
                       'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                       'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                       'bed', 'chair', 'couch', 'table', 'wardrobe',
                       'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                       'bear', 'leopard', 'lion', 'tiger', 'wolf',
                       'bridge', 'castle', 'house', 'road', 'skyscraper',
                       'cloud', 'forest', 'mountain', 'plain', 'sea',
                       'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                       'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                       'crab', 'lobster', 'snail', 'spider', 'worm',
                       'baby', 'boy', 'girl', 'man', 'woman',
                       'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                       'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                       'maple', 'oak', 'palm', 'pine', 'willow',
                       'bicycle', 'bus', 'motorcycle', 'pickup' 'truck', 'train',
                       'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
                        ],
           "cifar10": ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
           "svhn": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
           }

Models = {
           "mnist": Model1(),
           "kmnist": Model1(),
           "fashion-mnist": Model1(),
           "cifar10": Model2(),
           "cifar100": Model3(),
           "svhn": Model3(),
}

def mis_injection(dataset, mis_rate, mis_distribution):
    train_set_tmp = []
    test_set = []
    if dataset == "mnist":
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set_tmp = torchvision.datasets.MNIST(root="../../dataset/", train=True, transform=transform_train, download=True)
        test_set = torchvision.datasets.MNIST(root="../../dataset/", train=False, transform=transform_test, download=True)
    elif dataset == "cifar10":
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

        train_set_tmp = torchvision.datasets.CIFAR10(root="../../dataset/", train=True, transform=transform_train, download=True)
        test_clean_set = torchvision.datasets.CIFAR10(root="../../dataset/", train=False, transform=transform_test, download=True)
    elif dataset == "cifar100":
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

        train_set_tmp = torchvision.datasets.CIFAR100(root="../../dataset/", train=True, download=True,
                                                      transform=transform_train)

        test_set = torchvision.datasets.CIFAR100(root="../../dataset/", train=False, download=True,
                                                 transform=transform_test)
    elif dataset == "kmnist":
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set_tmp = torchvision.datasets.KMNIST(root="../../dataset/", train=True, transform=transform_train, download=True)
        test_set = torchvision.datasets.KMNIST(root="../../dataset/", train=False, transform=transform_test, download=True)
    elif dataset == "fashion-mnist":
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set_tmp = torchvision.datasets.FashionMNIST(root="../../dataset/", train=True, transform=transform_train,
                                                          download=True)
        test_set = torchvision.datasets.FashionMNIST(root="../../dataset/", train=False, transform=transform_test, download=True)
    elif dataset == "svhn":
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

        train_set_tmp = torchvision.datasets.SVHN(root="../../dataset/svhn/", split='train', transform=transform_train, download=True)
    # clean数据占比
    good_sample_ratio = 1 - float(mis_rate)
    train_set = []
    for i in train_set_tmp:
        train_set.append(list(i))

    if mis_distribution == "random":
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

        train_bad_dataset = []
        for i in bad_idx_array:
            train_bad_dataset.append(train_set_tmp[i])

        train_clean_bad_set_ground_truth = train_clean_dataset + train_bad_dataset

        train_clean_bad_set = train_clean_dataset + train_bad_dataset
        print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_set))
        return train_clean_dataset, train_bad_dataset, train_clean_bad_set, train_clean_bad_set_ground_truth

    else:
        # ---------------------------------------------------------------
        # 随机制造脏数据，而不是每个类取固定比例制造脏数据
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
            p = np.random.randint(0, len(classes[dataset]))
            while True:
                if p != i[1]:
                    i[1] = p
                    break
                p = np.random.randint(0, len(classes[dataset]))
        
        train_clean_bad_set_ground_truth = train_clean_dataset + train_bad_dataset
        train_clean_bad_set = train_clean_dataset + train_bad_dataset
        print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_set))
        return train_clean_dataset, train_bad_dataset, train_clean_bad_set, train_clean_bad_set_ground_truth
