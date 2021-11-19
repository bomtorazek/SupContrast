for TR in 2; do
    for SEED in 100; do
        #for UR in 0.1; do
        #for UR in 10 50 100 0.1 0.3 0.5 1.0; do
        for UR in 1.0; do
            GPU=${TR}
            #GPU=0
            ## Vistakon
            #TGTFD='/data/datasets/DR/lens/vistakon/child'
            #python main.py \
            #    --train_util_rate=0.1 --batch_size=8 --dataset=Vistakon --size=512 --aug nothing \
            #    --target_folder=${TGTFD} \
            #    --model=resnet18 \
            #    --epochs=30 --weight_decay=0 --cosine --method=CE --learning_rate=0.0001 \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=${TR} --sampling=${SAMP} \
            #    --gpu=0
            #TGTFD='/data/datasets/DR/lens/vistakon/child'
            #SRCFD='/data/datasets/DR/lens/vistakon/parent'
            #python main.py \
            #    --train_util_rate=0.1 --batch_size=8 --dataset=Vistakon --size=512 --aug nothing \
            #    --target_folder=${TGTFD} \
            #    --model=resnet18 \
            #    --epochs=30 --weight_decay=0 --cosine --method=Joint_CE --learning_rate=0.0001 \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=${TR} --sampling=${SAMP} \
            #    --gpu=0


            #TGTFD='/data/datasets/DR/DSub/d_sub_15pin/original' # ~ 0.94
            #SRCFD='/data/datasets/DR/DSub/d_sub_9pin/original'
            ## solo
            #python main.py \
            #    --train_util_rate=${UR} --batch_size=16 --dataset=D_Sub --size=288x512 \
            #    --target_folder=${TGTFD} \
            #    --model=resnet50 \
            #    --epochs=200 --weight_decay=1e-5 --cosine --method=CE --learning_rate=1e-4 --optimizer ADAMW \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=${TR} --sampling=unbalanced \
            #    --test_fold 2 --val_fold 1 --aug resize_crop \
            #    --gpu=${GPU}
            ## incremental learning
            #python main.py \
            #    --train_util_rate=${UR} --batch_size=16 --dataset=D_Sub --size=288x512 \
            #    --target_folder=${TGTFD} \
            #    --model=resnet50 --model_transfer './save/SupCon/D_Sub_models/D_Sub_resnet50_ur1.0_CE_me100_lr_0.0001_decay_1e-05_aug_pin_bsz_32_rsz_288_temp_0.08_sampling_unbalanced_whole_data_cosine_trial_999/last.pth' \
            #    --epochs=200 --weight_decay=1e-5 --cosine --method=CE --learning_rate=1e-4 --optimizer ADAMW \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=${TR} --sampling=unbalanced \
            #    --test_fold 2 --val_fold 1 --aug resize_crop \
            #    --gpu=${GPU}
            ## joint default
            #python main.py \
            #    --epochs=100 --trial=${TR} --test_fold 2 --val_fold 1 --train_util_rate=${UR} \
            #    --dataset=D_Sub --size=288x512 --aug resize_crop --batch_size=16 --sampling=unbalanced \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} \
            #    --target_folder=${TGTFD} --source_folder=${SRCFD} \
            #    --model=resnet50 --method=Joint_CE \
            #    --optimizer ADAMW --weight_decay=1e-5 --cosine --learning_rate=1e-4 \
            #    --gpu=${GPU}
            ## joint cont default
            #python main.py \
            #    --epochs=10 --trial=${TR} --test_fold 2 --val_fold 1 --train_util_rate=${UR} \
            #    --dataset=D_Sub --size=288x512 --aug resize_crop --batch_size=16 --sampling=warmup \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} \
            #    --target_folder=${TGTFD} --source_folder=${SRCFD} \
            #    --model=resnet50 --method=Joint_Con \
            #    --optimizer ADAMW --weight_decay=1e-5 --cosine --learning_rate=1e-4 \
            #    --gpu=${GPU}

            #TGTFD='/data/datasets/DR/DSub/d_sub_9pin/original' # ~ 0.955
            #SRCFD='/data/datasets/DR/DSub/d_sub_15pin/original'
            ## solo
            #python main.py \
            #    --train_util_rate=${UR} --batch_size=16 --dataset=D_Sub --size=288x512 \
            #    --target_folder=${TGTFD} \
            #    --model=resnet50 \
            #    --epochs=200 --weight_decay=1e-5 --cosine --method=CE --learning_rate=1e-4 --optimizer ADAMW \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=999${TR} --sampling=unbalanced \
            #    --test_fold 2 --val_fold 1 --aug resize_crop \
            #    --gpu=${GPU}
            ## incremental learning
            #python main.py \
            #    --train_util_rate=${UR} --batch_size=16 --dataset=D_Sub --size=288x512 \
            #    --target_folder=${TGTFD} \
            #    --model=resnet50 --model_transfer './save/SupCon/D_Sub_models/D_Sub_resnet50_ur1.0_CE_me200_lr_0.0001_decay_1e-05_aug_pin_bsz_16_rsz_288_temp_0.08_sampling_unbalanced_whole_data_cosine_trial_0/last.pth' \
            #    --epochs=200 --weight_decay=1e-5 --cosine --method=CE --learning_rate=1e-4 --optimizer ADAMW \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=999${TR} --sampling=unbalanced \
            #    --test_fold 2 --val_fold 1 --aug resize_crop \
            #    --gpu=${GPU}
            ## joint default
            #python main.py \
            #    --epochs=100 --trial=999${TR} --test_fold 2 --val_fold 1 --train_util_rate=${UR} \
            #    --dataset=D_Sub --size=288x512 --aug resize_crop --batch_size=16 --sampling=unbalanced \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} \
            #    --target_folder=${TGTFD} --source_folder=${SRCFD} \
            #    --model=resnet50 --method=Joint_CE \
            #    --optimizer ADAMW --weight_decay=1e-5 --cosine --learning_rate=1e-4 \
            #    --gpu=${GPU}
            ## joint cont default
            #python main.py \
            #    --epochs=100 --trial=999${TR} --test_fold 2 --val_fold 1 --train_util_rate=${UR} \
            #    --dataset=D_Sub --size=288x512 --aug resize_crop --batch_size=16 --sampling=warmup \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} \
            #    --target_folder=${TGTFD} --source_folder=${SRCFD} \
            #    --model=resnet50 --method=Joint_Con \
            #    --optimizer ADAMW --weight_decay=1e-5 --cosine --learning_rate=1e-4 \
            #    --gpu=${GPU}


            #TGTFD='/data/datasets/DR/SolarPannel/m2_6bb/original' # ~ 0.936
            #SRCFD='/data/datasets/DR/SolarPannel/m4_12bb/original' #
            #AUG=resize_crop
            ## solo
            #python main.py \
            #    --train_util_rate=${UR} --batch_size=16 --dataset=SolarPannel --size=600x300 --num_cls 6 \
            #    --target_folder=${TGTFD} \
            #    --model=resnet50 \
            #    --epochs=60 --weight_decay=1e-5 --cosine --method=CE --learning_rate=1e-4 --optimizer ADAMW \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=${TR} --sampling=unbalanced \
            #    --test_fold 1 --val_fold 1 --aug ${AUG} \
            #    --gpu=${GPU}
            ## incremental learning
            #python main.py \
            #    --epochs=60 --trial=${TR} --test_fold 1 --val_fold 1 --train_util_rate=${UR} --num_cls 6 \
            #    --dataset=SolarPannel --size=600x300 --aug ${AUG} --batch_size=16 --sampling=unbalanced \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} \
            #    --target_folder=${TGTFD} \
            #    --model=resnet50 --method=CE --model_transfer './save/SupCon/SolarPannel_models/SolarPannel_resnet50_ur1.0_CE_me60_lr_0.0001_decay_1e-05_aug_flip_bsz_16_rsz_[600, 300]_temp_0.08_sampling_unbalanced_whole_data_cosine_trial_999/last.pth' \
            #    --optimizer ADAMW --weight_decay=1e-5 --cosine --learning_rate=1e-4 \
            #    --gpu=${GPU}
            ## joint default
            #python main.py \
            #    --epochs=50 --trial=${TR} --test_fold 1 --val_fold 1 --train_util_rate=${UR} --num_cls 6 \
            #    --dataset=SolarPannel --size=600x300 --aug ${AUG} --batch_size=16 --sampling=unbalanced \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} \
            #    --target_folder=${TGTFD} --source_folder=${SRCFD} \
            #    --model=resnet50 --method=Joint_CE \
            #    --optimizer ADAMW --weight_decay=1e-5 --cosine --learning_rate=1e-4 \
            #    --gpu=${GPU}
            ## joint cont default
            #python main.py \
            #    --epochs=50 --trial=${TR} --test_fold 1 --val_fold 1 --train_util_rate=${UR} --num_cls 6 \
            #    --dataset=SolarPannel --size=600x300 --aug ${AUG} --batch_size=16 --sampling=warmup \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} \
            #    --target_folder=${TGTFD} --source_folder=${SRCFD} \
            #    --model=resnet50 --method=Joint_Con \
            #    --optimizer ADAMW --weight_decay=1e-5 --cosine --learning_rate=1e-4 \
            #    --gpu=${GPU}
            TGTFD='/data/datasets/DR/SolarPannel/m4_12bb/original' # ~ 0.9145
            SRCFD='/data/datasets/DR/SolarPannel/m2_6bb/original' #
            python main.py \
                --train_util_rate=${UR} --batch_size=16 --dataset=SolarPannel --size=600x300 --num_cls 6 \
                --target_folder=${TGTFD} \
                --model=resnet50 \
                --epochs=60 --weight_decay=1e-5 --cosine --method=CE --learning_rate=1e-4 --optimizer ADAMW \
                --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=000 --sampling=unbalanced \
                --test_fold 1 --val_fold 1 --aug flip \
                --gpu=${GPU}
        done
    done
done
