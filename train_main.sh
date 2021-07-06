# python main_ce.py --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.1 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --size=128 --cosine --method=CE --learning_rate=0.0001
# python main_joint_domain.py --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.1 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --source_data_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original --size=128 --cosine --method=Joint_CE --learning_rate=0.0001
# python main_joint_domain.py --batch_size=64 --val_fold=${FOLD} --head=mlp --train_util_rate=0.1 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --source_data_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original --size=128 --cosine --method=Joint_Con --learning_rate=0.0001

## RandAug
# python main_ce.py --aug=rand_1_7 --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.2 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --size=128 --cosine --method=CE --learning_rate=0.0001


## Cutmix
# python main_joint_domain_whole.py --dp --aug=cut_0.5_PP_oo --train_util_rate=0.1 --patience=200 --batch_size=64 --val_fold=${FOLD} --head=mlp --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --source_data_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original --size=128 --cosine --method=Joint_Con_Whole --learning_rate=0.0001           
# python main_ce.py --aug=cut_0.5_PP --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.2 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --size=128 --cosine --method=CE --learning_rate=0.0001
   


for FOLD in 1; do
    python main_ce.py \
        --train_util_rate=0.1 --batch_size=4 --val_fold=${FOLD} --dataset=Vistakon --size=512 \
        --data_folder=/data/esuh/vistakon/child \
        --source_data_folder=/data/esuh/vistakon/parent \
        --patience=200 --epochs=15 --weight_decay=0 --cosine --method=Joint_Con_Whole --learning_rate=0.0001
done
