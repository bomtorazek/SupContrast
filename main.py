from __future__ import print_function

import os
import time

import tensorboard_logger as tb_logger

from util import adjust_learning_rate
from util import set_optimizer
from modules.data import set_loader, adjust_batch_size
from modules.networks import set_model, save_model
from modules.runner import train, train_sampling, train_sampling_dsbn, validate, test
from config import parse_option

# from modules.pytorch_grad_cam import GradCAMPlusPlus, AblationCAM, EigenCAM


def main():
    opt = parse_option()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    if not opt.whole_data_train:
        best_auc = 0.0
        best_acc05 =0.0
        best_bacc = 0.0
        best_epoch = 0
        best_f1 = 0.0

    # build data loader
    loaders = set_loader(opt) # tuple or dict

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if opt.sampling == 'unbalanced':
        trainer = train
    elif 'warm' in opt.sampling or opt.sampling == 'balanced':
        if opt.dsbn:
            trainer = train_sampling_dsbn
        else:
            trainer = train_sampling

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)


        if 'warm' in opt.sampling:
            #FIXME duplicated dataloading when using warmup
            target_dataset = loaders['train']['target'].dataset
            source_dataset = loaders['train']['source'].dataset
            if opt.class_balanced:
                target_dataset.get_class_balanced()
                source_dataset.get_class_balanced()
            loaders['train'] = adjust_batch_size(opt, target_dataset, source_dataset, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = trainer(loaders['train'], model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))



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
                save_file = os.path.join(
                    opt.save_folder, 'auc_best.pth')
                save_model(model, optimizer, opt, epoch, save_file)
            if val_bacc >= best_bacc:
                best_epoch = epoch
                best_bacc = val_bacc
                best_th = val_th
                save_file = os.path.join(
                    opt.save_folder, 'bacc_best.pth')
                save_model(model, optimizer, opt, epoch, save_file)
            if val_acc05 >= best_acc05:
                best_epoch = epoch
                best_acc05 = val_acc05
                save_file = os.path.join(
                    opt.save_folder, 'acc05_best.pth')
                save_model(model, optimizer, opt, epoch, save_file)
            if val_f1 >= best_f1:
                best_epoch = epoch
                best_f1 = val_f1
                save_file = os.path.join(
                    opt.save_folder, 'f1_best.pth')
                save_model(model, optimizer, opt, epoch, save_file)

            if epoch >= best_epoch + opt.patience:
                break
        elif epoch == opt.epochs: # last epoch
            save_file = os.path.join(
                opt.save_folder, 'last.pth')
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
            row = [opt.model_name, 'auc', test_auc, 'acc', test_acc, 'f1', test_f1 ]
        writer.writerow(row)

if __name__ == '__main__':
    main()
