# python main_ce.py --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.1 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --size=128 --cosine --method=CE --learning_rate=0.0001
# python main_joint_domain.py --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.1 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --source_data_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original --size=128 --cosine --method=Joint_CE --learning_rate=0.0001
# python main_joint_domain.py --batch_size=64 --val_fold=${FOLD} --head=mlp --train_util_rate=0.1 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --source_data_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original --size=128 --cosine --method=Joint_Con --learning_rate=0.0001

## RandAug
# python main_ce.py --aug=rand_1_7 --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.2 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --size=128 --cosine --method=CE --learning_rate=0.0001


## Cutmix
# python main_joint_domain_whole.py --dp --aug=cut_0.5_PP_oo --train_util_rate=0.1 --patience=200 --batch_size=64 --val_fold=${FOLD} --head=mlp --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --source_data_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original --size=128 --cosine --method=Joint_Con_Whole --learning_rate=0.0001           
# python main_ce.py --aug=cut_0.5_PP --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.2 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --size=128 --cosine --method=CE --learning_rate=0.0001
   


### -----------------------------Vistakon------------------------------------------------
# for TR in 0 1 2 3 4; do
#     for SEED in 100; do
#         for SAMP in balanced; do

#             python main.py \
#                 --train_util_rate=0.1 --batch_size=16 --dataset=Vistakon --size=512 \
#                 --target_folder=/data/esuh/vistakon/child \
#                 --source_folder=/data/esuh/vistakon/parent \
#                 --model=resnet18dsbn \
#                 --epochs=30 --weight_decay=0 --cosine --method=Joint_Con --learning_rate=0.0001 --loss_type=SupCon \
#                 --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=${TR} --sampling=${SAMP} \
#                 --gpu=0

#         done
#     done
# done

## -------------------------------MLCC --------------------------------

for TR in 0 1 2 3 4; do
    for SEED in 100; do

        python main.py \
            --train_util_rate=0.1 --batch_size=64 --dataset=MLCCA --size=256 \
            --target_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original \
            --source_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original \
            --model=resnet18 \
            --epochs=200 --weight_decay=0 --cosine --method=Joint_Con --learning_rate=0.0001 --loss_type=pos_denom \
            --whole_data_train --trial=${TR} --sampling=${SAMP} \
            --gpu=0

    done
done

## -------------------- other ur for vistakon ----------------------

for UR in 10 50 100; do
    for EPH in 5 10 30; do
        for SAMP in balanced warmup; do
            for TR in 0 1 2 3 4; do

            python main.py \
                --train_util_rate=${UR} --batch_size=8 --dataset=Vistakon --size=512 \
                --target_folder=/data/esuh/vistakon/child \
                --source_folder=/data/esuh/vistakon/parent \
                --model=resnet18 \
                --epochs=${EPH} --weight_decay=0 --cosine --method=Joint_Con --learning_rate=0.0001 --loss_type=SupCon \
                --whole_data_train --ur_from_imageset --ur_seed=100 --trial=${TR} --sampling=${SAMP} \
                --gpu=3

            done
        done
    done
done