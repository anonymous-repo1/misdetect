import argparse
import time
from utils import *

def print_args(args):
    print(f'dataset : {args.dataset}')
    print(f'model : {args.model}')
    print(f'mislabel ratio : {args.mis_ratio}')
    print(f'mislabel distribution : {args.mis_distribution}')
    print(f'train batch size : {args.train_batch}')

    print(f'learning rate : {args.lr}')
    print(f'manual_seed: {args.seed}')
    print(f'optimizer: {args.optimizer}')
    print(f'knn for classification model: {args.kNN_k}')
    print(f'epochs: {args.epochs}')

    return

def get_options(parser=argparse.ArgumentParser(description='Parameter Processing')):
    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default=None, help='model')
    parser.add_argument('--mis_ratio', type=str, default="10%", help="mislabel ratio")
    parser.add_argument('--mis_distribution', type=str, default="random", help="mislabel distribution")

    parser.add_argument('--epochs', default=200, type=int, help='number of classification model to run')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Optimizer and scheduler
    parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')

    # Training
    parser.add_argument('--batch', '--batch-size', "-b", default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument("--train_batch", "-tb", default=None, type=int,
                     help="batch size for training, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument("--test_batch", "-testb", default=None, type=int,
                        help="batch size for test, if not specified, it will equal to batch size in argument --batch")


    args = parser.parse_args()

    print_args(args)
    return args