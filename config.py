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
                        help='num of workers to use, 16?')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=str, default='0,1,2,3',
                        help='gpu')
    parser.add_argument('--patience', type=int, default=200,
                        help='patience for training')

    # optimization
    parser.add_argument('--optimizer', type=str, default='ADAM')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate') # 0.05 SGD
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # network
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--num_cls', type=int, default=2)
    parser.add_argument('--model_transfer', type=str, default=None)
  
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                         help='dataset')
    parser.add_argument('--imgset_dir', type=str, default='fold.5-5/ratio/100%')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--target_folder', type=str, default=None, help='target')
    parser.add_argument('--source_folder', type=str, default=None, help='source')
    parser.add_argument('--val_fold', type=int, default=1, help='validation fold, 1 if whole_data_train')
    parser.add_argument('--test_fold', type=int, default=1, help='test fold, generally 1')
    parser.add_argument('--whole_data_train', action='store_true', default=False, help='whole data training, no validation set, make val+train to train')
    parser.add_argument('--train_util_rate', type=float, default=1.0, help='train util rate')
    parser.add_argument('--ur_from_imageset', action='store_true', default=False,  help='get UR from imageset txt')
    parser.add_argument('--ur_seed', type=int, default = 100, help='seed for ur_from_imageset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop or Resize')
    parser.add_argument('--sampling', type=str, default='unbalanced', choices=['unbalanced','balanced','warmup','warmIDS', 'warmKang', 'domainKang'])
    parser.add_argument('--class_balanced', action='store_true', default =False, help='use class-balanced dataset')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'Joint_Con', 'Joint_CE', 'CE'], help='choose method')
    parser.add_argument('--aug', type=str, default='sim',
                        help='augmentation type, rand_3_5, cutmix_0.5_PP_ox or AB or EI, stacked_rand_2_10 ')
    parser.add_argument('--mixup', type=str, default='',
                        choices=['inter_class', 'intra_class'])


    # temperature and hypers
    parser.add_argument('--temp', type=float, default=0.08,
                        help='temperature for loss function')
    parser.add_argument('--l_con', type=float, default=1.0,
                        help='lambda for cont loss')
    parser.add_argument('--head', type=str, default='mlp',
                        choices=['mlp'],
                        help='mlp only')
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='dim of feature head')
    parser.add_argument('--dp', action='store_true', default=False,
                        help='data parallel for whole model, dp for encoder by default. for cutmix, must activate dp ')
    parser.add_argument('--loss_type', type=str, default='SupCon',
                        choices=['SupCon', 'pos_denom', 'pos_numer'], 
                        help='choose loss, pos_denom means removing postivies of the denominator, and pos_numer means adding positives on the numerator')
    parser.add_argument('--one_crop', action='store_true', default=False,
                        help='if one_crop -> do not use two crop augmentation')


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

    ##------------Dataset------------
    if 'mlcc' in opt.dataset.lower():
        opt.imgset_dir = 'fold.5-5/ratio/100%'
    elif 'vis' in opt.dataset.lower():
        opt.imgset_dir = 'fold.0'
    elif 'interojo' in opt.dataset.lower():
        opt.imgset_dir = ''
    else:
        opt.imgset_dir = ''


    if opt.ur_from_imageset and not opt.whole_data_train:
        raise ValueError("when using 'ur_from_imageset, need to activate 'whole_data_train")

    ##------------Path------------
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    ##------------Iteration & LR scheduling------------
    iterations = opt.lr_decay_epochs.split(',') # 700,800,900
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it)) # [700,800,900] exponential lr decay default

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

    ##------------DSBN------------
    opt.dsbn = False
    if 'dsbn' in opt.model:
        opt.dsbn = True 
        if 'Joint' not in opt.method:
            raise ValueError("dsbn is only for Joint methods")
        if opt.sampling == 'unbalanced':
            raise NotImplementedError("unbalanced sampling with DSBN is not currently supported.")
        
    ##------------Sampling------------
    if opt.class_balanced: # FIXME
        opt.model_name += '_class_balanced'
        if opt.sampling != 'warmup':
            raise NotImplementedError("class_balanced method is only supported for warmup sampling now.")


    ##------------Model Name------------
    if opt.train_util_rate >= 1.0:
        opt.train_util_rate = int(opt.train_util_rate)
    opt.model_name = '{}_{}_{}_ur{}_me{}_lr_{}_decay_{}_aug_{}_twocrop_{}_bsz_{}_rsz_{}_temp_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.train_util_rate, opt.epochs,opt.learning_rate,
               opt.weight_decay, opt.aug, not opt.one_crop, opt.batch_size, opt.size, opt.temp)
    
    if opt.model_transfer is not None:
        opt.model_name += '_IL'
    if 'Con' in opt.method:
        opt.model_name += '_' + opt.loss_type + '_lambda' + str(opt.l_con)  + '_'
    opt.model_name += '_sampling_'+ opt.sampling 

    if opt.sampling != 'unbalanced':
        if len(opt.gpu) > 1: #FIXME
            raise NotImplementedError("""Sampling method is not supported for multi-gpu yet.\n   
                When using unbalaned sampling, recommend that not using multi-gpus due to distribution problems of mini-batches.""")
        elif 'Joint' not in opt.method:
            raise ValueError("CE method can only has unbalanced sampling method")

    if opt.whole_data_train:
        opt.model_name += '_whole_data'
    else:
        opt.model_name += f'_fold_{opt.val_fold}'

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    
    if opt.mixup:
        opt.model_name += f'_{opt.mixup}'

    opt.model_name += f'_seed_{opt.ur_seed}' # FIXME
    opt.model_name += f'_trial_{opt.trial}'


    print(f"model name:{opt.model_name}")
    ##------------Tensorboard & Save Folder------------    
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name) 
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    

    return opt