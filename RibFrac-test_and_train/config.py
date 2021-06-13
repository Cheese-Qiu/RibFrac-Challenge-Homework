import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=7,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_false',help='not use cpu only')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# Preprocess parameters
parser.add_argument('--n_labels', type=int, default=2,help='number of classes')
parser.add_argument('--upper', type=int, default=200, help='')
parser.add_argument('--lower', type=int, default=-200, help='')
parser.add_argument('--norm_factor', type=float, default=200.0, help='')
parser.add_argument('--expand_slice', type=int, default=15, help='')
parser.add_argument('--min_slices', type=int, default=48, help='')
parser.add_argument('--xy_down_scale', type=float, default=0.5, help='')
parser.add_argument('--slice_down_scale', type=float, default=1.0, help='')
parser.add_argument('--valid_rate', type=float, default=1, help='')

# data in/out and dataset
parser.add_argument('--dataset_path',default = './Homework_fixed_data',help='fixed trainset root path')
parser.add_argument('--save',default='up3',help='save path of trained model')
parser.add_argument('--batch_size', type=list, default=2,help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=200, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--early-stop', default=80, type=int, help='early stopping (default: 20)')
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--val_crop_max_size', type=int, default=256)

# test
parser.add_argument('--test_cut_size', type=int, default=48,help='')
parser.add_argument('--test_cut_stride', type=int, default=24,help='')

args = parser.parse_args()


