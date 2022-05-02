for TR in 0 1 2 3 4; do
        UR=10
        BS=4
        GPU=0

        CUDA_VISIBLE_DEVICES=${GPU} python main.py \
            --train_util_rate=${UR} --batch_size=${BS} --dataset=d_sub --size=224 \
            --target_folder=/data/esuh/dsub/d_sub_15pin/original \
            --model=resnet50 \
            --epochs=50 --weight_decay=0 --cosine --method=CE --learning_rate=0.0001 \
            --whole_data_train --trial=${TR}\
            --gpu=${GPU} --aug=nothing
            
        CUDA_VISIBLE_DEVICES=${GPU} python main.py \
            --train_util_rate=${UR} --batch_size=${BS} --dataset=d_sub --size=224 \
            --target_folder=/data/esuh/dsub/d_sub_15pin/original \
            --model=CLIP_full_resnet50 \
            --epochs=50 --weight_decay=0 --cosine --method=CE --learning_rate=0.0001 \
            --whole_data_train --trial=${TR}\
            --gpu=${GPU}

        UR=50
        BS=8
        GPU=0

        CUDA_VISIBLE_DEVICES=${GPU} python main.py \
            --train_util_rate=${UR} --batch_size=${BS} --dataset=d_sub --size=224 \
            --target_folder=/data/esuh/dsub/d_sub_15pin/original \
            --model=resnet50 \
            --epochs=50 --weight_decay=0 --cosine --method=CE --learning_rate=0.0001 \
            --whole_data_train --trial=${TR}\
            --gpu=${GPU} --aug=nothing
            
        CUDA_VISIBLE_DEVICES=${GPU} python main.py \
            --train_util_rate=${UR} --batch_size=${BS} --dataset=d_sub --size=224 \
            --target_folder=/data/esuh/dsub/d_sub_15pin/original \
            --model=CLIP_full_resnet50 \
            --epochs=50 --weight_decay=0 --cosine --method=CE --learning_rate=0.0001 \
            --whole_data_train --trial=${TR}\
            --gpu=${GPU}

        UR=100
        BS=16
        GPU=0

        CUDA_VISIBLE_DEVICES=${GPU} python main.py \
            --train_util_rate=${UR} --batch_size=${BS} --dataset=d_sub --size=224 \
            --target_folder=/data/esuh/dsub/d_sub_15pin/original \
            --model=resnet50 \
            --epochs=50 --weight_decay=0 --cosine --method=CE --learning_rate=0.0001 \
            --whole_data_train --trial=${TR}\
            --gpu=${GPU} --aug=nothing
            
        CUDA_VISIBLE_DEVICES=${GPU} python main.py \
            --train_util_rate=${UR} --batch_size=${BS} --dataset=d_sub --size=224 \
            --target_folder=/data/esuh/dsub/d_sub_15pin/original \
            --model=CLIP_full_resnet50 \
            --epochs=50 --weight_decay=0 --cosine --method=CE --learning_rate=0.0001 \
            --whole_data_train --trial=${TR}\
            --gpu=${GPU}


done

# for TR in 3 4; do
#         UR=10
#         BS=4
#         GPU=0

#         python main.py \
#             --train_util_rate=${UR} --batch_size=${BS} --dataset=MLCCA --size=224 \
#             --target_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original \
#             --source_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original \
#             --model=resnet50 \
#             --epochs=50 --weight_decay=0 --cosine --method=Joint_CE --learning_rate=0.0001 \
#             --whole_data_train --trial=${TR}\
#             --gpu=${GPU} --aug=nothing

# done

# for TR in 3 4; do
#         UR=50
#         BS=8
#         GPU=0

#         python main.py \
#             --train_util_rate=${UR} --batch_size=${BS} --dataset=MLCCA --size=224 \
#             --target_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original \
#             --source_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original \
#             --model=resnet50 \
#             --epochs=50 --weight_decay=0 --cosine --method=Joint_CE --learning_rate=0.0001 \
#             --whole_data_train --trial=${TR}\
#             --gpu=${GPU} --aug=nothing

# done


# for TR in 3 4; do
#         UR=100
#         BS=16
#         GPU=0

#         python main.py \
#             --train_util_rate=${UR} --batch_size=${BS} --dataset=MLCCA --size=224 \
#             --target_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original \
#             --source_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original \
#             --model=resnet50 \
#             --epochs=50 --weight_decay=0 --cosine --method=Joint_CE --learning_rate=0.0001 \
#             --whole_data_train --trial=${TR}\
#             --gpu=${GPU} --aug=nothing

# done




