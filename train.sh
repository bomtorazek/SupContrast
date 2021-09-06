## Vistakon
for TR in 2; do
    for SEED in 100; do
        #for UR in 1.0; do
        for UR in 0.1 0.3 0.5 1.0; do
            GPU=1
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


            TGTFD='/data/datasets/DR/DSub/d_sub_15pin/original' # ~ 0.94
            SRCFD='/data/datasets/DR/DSub/d_sub_9pin/original'
            python main.py \
                --train_util_rate=${UR} --batch_size=32 --dataset=D_Sub --size=288 \
                --target_folder=${TGTFD} \
                --model=resnet50 \
                --epochs=100 --weight_decay=1e-5 --cosine --method=CE --learning_rate=1e-4 --optimizer ADAMW \
                --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=${TR} --sampling=unbalanced \
                --test_fold 2 --val_fold 1 --aug pin \
                --gpu=${GPU}
            python main.py \
                --train_util_rate=${UR} --batch_size=32 --dataset=D_Sub --size=288 \
                --target_folder=${TGTFD} \
                --source_folder=${SRCFD} \
                --model=resnet50 \
                --epochs=100 --weight_decay=1e-5 --cosine --method=Joint_CE --learning_rate=1e-4 --optimizer ADAMW \
                --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=${TR} --sampling=balanced \
                --test_fold 2 --val_fold 1 --aug pin \
                --gpu=${GPU}
            python main.py \
                --train_util_rate=${UR} --batch_size=16 --dataset=D_Sub --size=288 \
                --target_folder=${TGTFD} \
                --source_folder=${SRCFD} \
                --model=resnet50 \
                --epochs=100 --weight_decay=1e-5 --cosine --method=Joint_Con --learning_rate=1e-4 --optimizer ADAMW \
                --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=${TR} --sampling=balanced \
                --test_fold 2 --val_fold 1 --aug pin \
                --gpu=${GPU}

            #TGTFD='/data/datasets/DR/DSub/d_sub_9pin/original' # ~ 0.955
            #python main.py \
            #    --train_util_rate=1.0 --batch_size=32 --dataset=D_Sub --size=288 \
            #    --target_folder=${TGTFD} \
            #    --model=resnet50 \
            #    --epochs=100 --weight_decay=0 --cosine --method=CE --learning_rate=1e-4 \
            #    --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=${TR} --sampling=${SAMP} \
            #    --test_fold 2 --val_fold 1 --aug pin \
            #    --gpu=0
        done
    done
done
