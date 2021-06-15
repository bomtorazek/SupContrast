import argparse
import math
import os

from numpy.core.arrayprint import IntegerFormat

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer', type=str, default='ADAM')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate') # 0.05 SGD
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    # network
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--num_cls', type=int, default=2)
    parser.add_argument('--model_transfer', type=str, default=None)
    parser.add_argument('--gpu', type=str, default= '0')

    parser.add_argument('--dataset', type=str, default='cifar10',
                         help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='target')
    parser.add_argument('--source_data_folder', type=str, default=None, help='source')
    parser.add_argument('--val_fold', type=int, default=1, help='validation fold')
    parser.add_argument('--test_fold', type=int, default=1, help='test fold')
    parser.add_argument('--train_util_rate', type=float, default=1.0, help='train util rate')
    # parser.add_argument('--translate', type=int, default=16, help='translation augment')
    # parser.add_argument('--rotate90', type=bool, default=True, help='rotation augment')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'Joint_Con', 'Joint_CE', 'CE'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',') # 700,800,900
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it)) # [700,800,900] exponential lr decay default

    opt.model_name = '{}_{}_{}_ur{}_fold{}_me{}_lr_{}_decay_{}_bsz_{}_rsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.train_util_rate,opt.val_fold, opt.epochs,opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.size, opt.temp, opt.trial)
    # Joint, MLCC, ResNet18, 0.001, 1e-4, bs, temp, 

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name) # tensorboard
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt