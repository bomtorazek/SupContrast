import torch
import torch.backends.cudnn as cudnn

from modules.networks.resnet_big import SupHybResNet, SupCEResNet
from modules.runner.losses import SupConLoss


def set_model(opt):

    if opt.method == 'Joint_Con':
        model = SupHybResNet(name=opt.model, feat_dim=opt.feat_dim,num_classes=opt.num_cls) 
        criterion = {}
        criterion['Con'] = SupConLoss(temperature=opt.temp, 
                                    remove_pos_denom=opt.remove_pos_denom)
        criterion['CE'] = torch.nn.CrossEntropyLoss() 
    elif 'CE' in opt.method:
        model = SupCEResNet(name=opt.model, num_classes=opt.num_cls)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("unsupported method")

    if opt.model_transfer is not None:
        pretrained_dict = torch.load(opt.model_transfer)['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
   
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            if opt.dp: # dataparallel to whole model
                model = torch.nn.DataParallel(model)
            else: # dataparallel to only encoder
                model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        if opt.method == 'Joint_Con':
            criterion['CE']=criterion['CE'].cuda()
            criterion['Con']=criterion['Con'].cuda()
        elif 'CE' in opt.method:
            criterion = criterion.cuda()
        else:
            raise ValueError("check method")
        cudnn.benchmark = True

    return model, criterion

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state