from __future__ import print_function

import os
import time

import tensorboard_logger as tb_logger

from util import adjust_learning_rate
from util import set_optimizer
from modules.data import set_loader
from modules.networks import set_model, save_model
from modules.runner import train, train_sampling, validate, test
from config import parse_option


def main():
    opt = parse_option()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    if not opt.whole_data_train:
        best_auc = 0.0
        best_acc05 =0.0
        best_bacc = 0.0
        best_epoch = 0
        best_f1 = 0.0

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if opt.sampling == 'unbalanced':
        trainer = train
    elif 'warm' in opt.sampling or opt.sampling == 'balanced':
        trainer = train_sampling

    # flag for initialize dataloader for every epoch or not (e.g., warm up, non-fixed source sampling)
    init_loader_everyep = \
        opt.sampling=='warmup' \
        or opt.source_util_rate < 1.0

    # training routine
    opt.t0 = time.time()
    for epoch in range(1, opt.epochs + 1):
        opt.epoch = epoch
        if epoch == 1 or init_loader_everyep:
            # build data loader
            loaders = set_loader(opt) # tuple or dict

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        loss, train_acc = trainer(loaders['train'], model, criterion, optimizer, epoch, opt)

        # tensorboard logger
        if opt.method == 'Joint_Con':
            logger.log_value('train_loss_CE', loss['CE'], epoch)
            logger.log_value('train_Con_TT', loss['Con'], epoch)
        elif 'CE' in opt.method:
            logger.log_value('train_loss', loss, epoch)
        else:
            raise NotImplementedError("unsupported method")
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        if not opt.whole_data_train:
            val_loss, val_auc, val_bacc, val_th, val_acc05, val_f1 = validate(loaders['val'], model, criterion, opt)
            logger.log_value('val_loss', val_loss, epoch)
            logger.log_value('val_auc', val_auc, epoch)
            logger.log_value('val_acc', val_bacc, epoch)
            logger.log_value('val_th', val_th, epoch)
            logger.log_value('val_acc_0.5', val_acc05, epoch)
            logger.log_value('val_f1', val_f1, epoch)

            if val_auc >= best_auc:
                best_epoch = epoch
                best_auc = val_auc
                save_file = os.path.join(opt.save_folder, 'auc_best.pth')
                save_model(model, optimizer, opt, epoch, save_file)
            if val_bacc >= best_bacc:
                best_epoch = epoch
                best_bacc = val_bacc
                best_th = val_th
                save_file = os.path.join(opt.save_folder, 'bacc_best.pth')
                save_model(model, optimizer, opt, epoch, save_file)
            if val_acc05 >= best_acc05:
                best_epoch = epoch
                best_acc05 = val_acc05
                save_file = os.path.join(opt.save_folder, 'acc05_best.pth')
                save_model(model, optimizer, opt, epoch, save_file)
            if val_f1 >= best_f1:
                best_epoch = epoch
                best_f1 = val_f1
                save_file = os.path.join(opt.save_folder, 'f1_best.pth')
                save_model(model, optimizer, opt, epoch, save_file)

            if epoch >= best_epoch + opt.patience:
                break
        else:
            if opt.debug:
                test_auc, test_acc, test_f1 = test(loaders['test'], model, opt, metric='last')
                print('Test auc: {:.4f}'.format(test_auc), end = ' ')
                print('Test acc: {:.4f}'.format(test_acc) ,end = ' ')
                print('Test f1: {:.4f}'.format(test_f1))
            if epoch == opt.epochs: # last epoch
                save_file = os.path.join(opt.save_folder, 'last.pth')
                save_model(model, optimizer, opt, epoch, save_file)

    if not opt.whole_data_train:
        test_auc = test(loaders['test'], model, opt, metric='auc')
        test_bacc, test_Bacc, test_th  = test(loaders['test'], model, opt, best_th=best_th)
        test_acc = test(loaders['test'], model, opt, metric='acc')
        test_f1 = test(loaders['test'], model, opt, metric='f1')
        print('Test auc: {:.4f}'.format(test_auc), end = ' ')
        print('Test acc: {:.4f}'.format(test_bacc) ,end = ' ')
        print('Test th: {:.2f}'.format(best_th))
    else:
        test_auc, test_acc, test_f1 = test(loaders['test'], model, opt, metric='last')
        print('Test auc: {:.4f}'.format(test_auc), end = ' ')
        print('Test acc: {:.4f}'.format(test_acc) ,end = ' ')
        print('Test f1: {:.4f}'.format(test_f1))

    import csv
    with open("result.csv", "a") as file:
        writer = csv.writer(file)
        if not opt.whole_data_train:
            row = [opt.model_name, 'auc', test_auc, 'acc', test_bacc, 'th', best_th,
                    'acc_0.5',test_acc,'acc_test_best', test_Bacc,'test_th',test_th, 'f1', test_f1 ]
        else:
            row = [opt.model_name, 'auc', test_auc, 'acc', test_acc, 'f1', test_f1, 'time', time.time()-opt.t0 ]
        writer.writerow(row)

if __name__ == '__main__':
    main()
