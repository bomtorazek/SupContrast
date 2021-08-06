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
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu')


    # optimization
    parser.add_argument('--optimizer', type=str, default='SGD')
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
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--num_cls', type=int, default=2)
    parser.add_argument('--model_transfer', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='office-31',
                        choices=['office-31', 'office-home', 'visda'],
                         help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--target_folder', type=str, default=None, help='target')
    parser.add_argument('--source_folder', type=str, default=None, help='source')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop or Resize')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['Joint_Con', 'Joint_CE', 'CE'], help='choose method')

    # temperature and hypers
    parser.add_argument('--pseudo_epoch', type=int, default=2,
                        help='epoch for starting pseudo labeling')
    parser.add_argument('--pseudo_threshold', type=float, default=0.8,
                        help='threshold for pseudo labeling')    
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--l_ce', type=float, default=1.0,
                        help='lambda for cont loss')
    parser.add_argument('--head', type=str, default='mlp',
                        choices=['mlp', 'fc'],
                        help='mlp or fc')
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='dim of feature head')
    parser.add_argument('--aug', type=str, default='sim',
                        help='augmentation type, rand_3_5, cutmix_0.5_PP or AB or EI, stacked_rand_2_10 ')
    parser.add_argument('--dp', action='store_true', default=False,
                        help='data parallel for whole model, dp for encoder by default ')
    parser.add_argument('--loss_type', type=str, default='SupCon',
                        choices=['SupCon', 'pos_denom', 'pos_numer'], help='choose loss')

                   
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()



    # set the path according to the environment
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',') # 700,800,900
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it)) # [700,800,900] exponential lr decay default

    opt.model_name = '{}_{}_{}_me{}_lr_{}_decay_{}_aug_{}_bsz_{}_rsz_{}_temp_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.epochs,opt.learning_rate,
               opt.weight_decay, opt.aug, opt.batch_size, opt.size, opt.temp)

    opt.model_name += '_'+opt.loss_type

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = opt.learning_rate/5
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    
    target_domain = os.path.basename(opt.target_folder)
    source_domain = os.path.basename(opt.source_folder)
    opt.model_name += f'_{source_domain}2{target_domain}'

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name) # tensorboard
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt