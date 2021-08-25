## Vistakon
for TR in 0; do
    for SEED in 100; do
        for SAMP in warmup; do
            python main.py \
                --train_util_rate=0.1 --batch_size=8 --dataset=Vistakon --size=512 \
                --target_folder='/data/datasets/DR/lens/vistakon/child' \
                --source_folder='/data/datasets/DR/lens/vistakon/parent' \
                --model=resnet18 \
                --epochs=30 --weight_decay=0 --cosine --method=Joint_CE --learning_rate=0.0001 \
                --whole_data_train --ur_from_imageset --ur_seed=${SEED} --trial=${TR}_fake --sampling=${SAMP} \
                --gpu=0
        done
    done
done
