import os
import os.path as osp
import numpy as np

root = '/data/esuh/vistakon/child/imageset/single_image.2class/fold.0'
train_path = root + '/train.1-1.txt'
val_path = root + '/validation.1-1.txt'

with open(train_path, 'r') as fid:
    temp = fid.read()
train_names = temp.split('\n')[:-1]

with open(val_path, 'r') as fid:
    temp = fid.read()
val_names = temp.split('\n')[:-1]
whole_names = train_names + val_names



SEED = 10000
num_used = int(len(whole_names) * 0.1)
np.random.seed(SEED)
train_names = np.random.choice(whole_names, size=num_used, replace=False)

write_path = train_path[:-4] + f'-ur0.1-{SEED}.txt'
with open(write_path, 'w') as f:
    for img in train_names:
        f.write(img + '\n')
