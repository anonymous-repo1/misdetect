import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

METHODS = ['M-W-IM', 'M-W-M', 'MisDetect']

y_lims = [
    [0.0, 1.0],  # IMDBCLinear
    [0.0, 1.0],  # IMDBLargeCLinear
    [0.0, 1.0],  # Stack
    [0.0, 1.0],  # Taxi
    [0.0, 1.0],  # IMDBC5
    [0.0, 1.0],  # IMDBLargeC5
    # [0.0, 1.0],  # Brazil5
]

effect_list = [
    # '(a) USCensus', '(b) Credit', '(c) Mobile Price', '(d) Airline', '(e) SVHN', '(f) MINIST', '(g) EEG'
    # first round
    np.array([0.1667, 0.1851, 0.2150, 0.2385, 0.1823, 0.2836]), # Early Loss
    np.array([0.1962, 0.2173, 0.2340, 0.2564, 0.2035, 0.3040]), # Influence
    np.array([0.1962, 0.2173, 0.2340, 0.2564, 0.2035, 0.3040]), # MisDetect

    # second round
    np.array([0.3682, 0.3784, 0.4290, 0.4445, 0.3633, 0.5081]),  # Early Loss
    np.array([0.3854, 0.3902, 0.4429, 0.4698, 0.3891, 0.5371]),  # Influence
    np.array([0.3854, 0.3902, 0.4429, 0.4698, 0.3891, 0.5371]),  # MisDetect

    # third round
    np.array([0.5519, 0.5861, 0.6172, 0.6889, 0.4810, 0.6754]),  # Early Loss
    np.array([0.5800, 0.6181, 0.6444, 0.7290, 0.5111, 0.7047]),  # Influence
    np.array([0.5800, 0.6181, 0.6444, 0.7290, 0.5111, 0.7047]),  # MisDetect

    # forth round
    np.array([0.6819, 0.7660, 0.7471, 0.8445, 0.6456, 0.8161]),  # Early Loss
    np.array([0.7040, 0.7892, 0.7668, 0.8686, 0.6696, 0.8351]),  # Influence
    np.array([0.7811, 0.8651, 0.8448, 0.9404, 0.7416, 0.9148]),  # MisDetect
]



colors = [
    # [240, 230, 140],
    # [189, 183, 107],
    # [168, 168, 168],  # jointhen
    [150, 196, 203],
    # [234, 156, 170],
    # [218, 165, 32],
    [240, 128, 128],  # ours
    [192, 192, 192],
    # [218, 165, 32],
    # [102, 205, 170],
    # [220, 235, 92],
    # [150, 196, 203],
]

hatches = ['\\', 'o', '-',
           '/',
           '.', 'x', 'o',
           '+', 'O', '.', '|']

Linecolors = [
    [53, 116, 151],
    [214, 128, 19],
    [68, 27, 41],
    [173, 0, 20]
    #     [131,46, 87]
    #     [108,86,184]
]

# plt.rcParams['text.usetex'] = True
COLOR = "goldenrod"
plt.subplots_adjust(wspace=5, hspace=1)
TITLE_SIZE = 28
BAR_LABEL_SIZE = 28

# [0.1, 0.9] 画图 6个
# st_loc = 0.1
# width = (1 - 2 * st_loc) / 6  # 实际间隔
# bar_width = width * 0.8 # bar的宽度

# 坐标
# Locs = [st_loc + j * width for j in range(6)] # 6个方法
# print('Locs', Locs)

# Locs = np.arange(len(METHODS))
Locs = np.array([0, 1, 2,
                 4, 5, 6,
                 8, 9, 10,
                 12, 13, 14,
                 # 9, 10, 11, 12, 13,
                 # 18, 19, 20, 21, 22,
                 # 27, 28, 29, 30, 31,
                 # 36, 37, 38, 39, 40
                 ])
bar_width = 0.9

labels_ = [
    [3, 6, 9, 12], # uscensus 5% (4 * 1%)
    [3, 6, 9, 11], # credit 16.55% (3.9 * 4%)
    [3, 6, 9, 16], # mobile 30% (5 * 4%)
    [3, 6, 9, 15], # airline 40% (5 * 7%)
    [3, 6, 9, 18], # svhn 10% (6 * 1%)
    [3, 6, 9, 21], # minist 10% (7 * 1%)
]

with plt.style.context("seaborn-paper"):
    fig = plt.figure(figsize=(32, 5))

    titles = ['(a) USCensus', '(b) Credit', '(c) Mobile-Price', '(d) Airline', '(e) SVHN', '(f) MINIST']

    metrics = np.concatenate((['F1 Score'] * 6,))
    rounds = np.concatenate((['Epochs'] * 6,))

    for i in range(6):
        subplt = 171 + i

        ax = fig.add_subplot(subplt)
        #         for j in range(6):
        for j in range(12):
            plt.bar(Locs[j], effect_list[j][i], width=bar_width,
                    # label=METHODS[j % 5],
                    edgecolor="black",
                    linewidth=2,
                    color=(colors[j % 3][0] / 255., colors[j % 3][1] / 255., colors[j % 3][2] / 255.),
                    hatch=hatches[j % 3])
        plt.title(titles[i], fontsize=28, y=-0.5, loc='center', weight='normal')
        plt.ylim(y_lims[i][0], y_lims[i][1])

        #         plt.xlabel("Datasets", fontsize = 28)
        #         plt.ylabel("Accuracy", fontsize = 18,weight='bold')
        plt.ylabel(metrics[i], fontsize=28, weight='normal')
        plt.xlabel(rounds[i], fontsize=28, weight='normal')
        #         plt.title("Test Accuracy Comparison on Classification", fontsize = 27)
        #         plt.ylim(0.55,0.7)
        #         plt.xticks(np.arange(DatasetNum) + 0.4, Datasets, fontsize = 24)
        plt.yticks(fontsize=18)
        plt.xticks([1, 5, 9, 13], labels=labels_[i], fontsize=20)
        i = i + 1

    fig.tight_layout()
    plt.legend(METHODS, fontsize=26, loc='lower center', ncol=5, bbox_to_anchor=(-2.9, 1.1))
    fig = plt.gcf()
    # fig.show()
    fig.savefig('baseline_self_f1.pdf', format='pdf', bbox_inches='tight', dpi=5000)
