from __future__ import print_function

import os
import time

import tensorboard_logger as tb_logger

from util import adjust_learning_rate
from util import set_optimizer
from modules.data import set_loader
from modules.networks import set_model, save_model
from modules.runner import train, validate
from config import parse_option

# from modules.pytorch_grad_cam import GradCAMPlusPlus, AblationCAM, EigenCAM


def main():
    opt = parse_option()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    best_acc = 0.0
    best_acc5 = 0.0

    # build data loader
    loaders = set_loader(opt) # tuple or dict

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc, mask_prob = train(loaders['train'], model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        if opt.method == 'Joint_Con':
            logger.log_value('train_loss_CE', loss['CE'], epoch)
            logger.log_value('train_Con', loss['Con'], epoch)

        elif 'CE' in opt.method:
            logger.log_value('train_loss', loss, epoch)
        else:
            raise NotImplementedError("unsupported method")
        
        if mask_prob is not None:
            logger.log_value('mask_prob', mask_prob, epoch)
        logger.log_value('train_acc_source', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)


        # evaluation
        val_loss, val_acc, val_acc5= validate(loaders['val'], model, criterion, opt)
        logger.log_value('val_loss', val_loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)
  

        if val_acc >= best_acc:
            best_epoch = epoch
            best_acc = val_acc
            best_acc5 = val_acc5
            save_file = os.path.join(
                opt.save_folder, f'acc_best_{best_epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)
        if epoch == opt.epochs:
            save_file = os.path.join(
                opt.save_folder, 'last.pth')
            save_model(model, optimizer, opt, epoch, save_file)
            last_acc = val_acc
            last_acc5 = val_acc5

 

    import csv
    with open("result_UDA.csv", "a") as file:
        writer = csv.writer(file)
        row = [opt.model_name, 'best_acc', best_acc, 'best_acc5', best_acc5, 'last_acc', last_acc, 'last_acc5', last_acc5]
        writer.writerow(row)

if __name__ == '__main__':
    main()
