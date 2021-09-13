import os
import os.path as osp
import numpy as np

## parameters
# root = '/data/esuh/vistakon/child/imageset/single_image.2class/fold.0'
root = '/data/esuh/mlcc/mlcc_crop/05A226_crop/original/imageset/single_image.2class/fold.5-5/ratio/100%'
# root = '/data/esuh/interojo/target/original/imageset/single_images.2class'
# root = '/data/esuh/interojo/target/original/imageset/single_image.2class'
train_path = root + '/train.1-1.txt'
val_path = root + '/validation.1-1.txt'
ur_tenth = False # for generating ur 0.1 with different seeds
optimal_seed = 1 # optimal seed, vistakon: 100, MLCC: 1


# end of parameters 
##-----------------------------------------------------------------------------##
with open(train_path, 'r') as fid:
    temp = fid.read()
train_names = temp.split('\n')[:-1]

with open(val_path, 'r') as fid:
    temp = fid.read()
val_names = temp.split('\n')[:-1]
whole_names = train_names + val_names

## for generating ur 0.1
if ur_tenth:
    for SEED in [1, 10, 100, 1000, 1000]:
        num_used = int(len(whole_names) * 0.1)
        np.random.seed(SEED)
        real_train_names = np.random.choice(whole_names, size=num_used, replace=False)

        write_path = train_path[:-4] + f'-ur0.1-{SEED}.txt'
        with open(write_path, 'w') as f:
            for img in real_train_names:
                f.write(img + '\n')

## make other ur imagesets based on ur 0.1
else:
    ur_tenth_train_path = root + f'/train.1-1-ur0.1-{optimal_seed}.txt'
    with open(ur_tenth_train_path, 'r') as fid:
        temp = fid.read()
    train_tenth_names = temp.split('\n')[:-1]
    train_10th_names = train_tenth_names
    # 10, 50, 100 images
    np.random.seed(optimal_seed)
    real_train_names = []
    for num_used in [10, 50, 100]:
        train_10th_names = [img for img in train_10th_names if img not in real_train_names]
        real_size = num_used - len(real_train_names)
        real_train_names += list(np.random.choice(train_10th_names, size=real_size, replace=False))
        write_path = train_path[:-4] + f'-ur{num_used}-{optimal_seed}.txt'
        with open(write_path, 'w') as f:
            for img in real_train_names:
                f.write(img + '\n')
    
    # 0.3, 0.5
    residual_names = [image for image in whole_names if image not in train_tenth_names]
    for ur in [0.3, 0.5]:
        num_used = int(len(whole_names) * ur) - len(train_tenth_names)
        real_train_names = np.random.choice(residual_names, size=num_used, replace=False)
        real_train_names = train_tenth_names +  list(real_train_names) 
        assert len(real_train_names) == len(set(real_train_names))

        write_path = train_path[:-4] + f'-ur{ur}-{optimal_seed}.txt'
        with open(write_path, 'w') as f:
            for img in real_train_names:
                f.write(img + '\n')


