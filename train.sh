# python main_ce.py --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.1 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --size=128 --cosine --method=CE --learning_rate=0.0001
# python main_joint_domain.py --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.1 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --source_data_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original --size=128 --cosine --method=Joint_CE --learning_rate=0.0001
# python main_joint_domain.py --batch_size=64 --val_fold=${FOLD} --head=mlp --train_util_rate=0.1 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --source_data_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original --size=128 --cosine --method=Joint_Con --learning_rate=0.0001
    
for FOLD in 1 2 3 4 5; do
    python main_ce.py --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.3 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --size=128 --cosine --method=CE --learning_rate=0.0001
    python main_ce.py --batch_size=64 --val_fold=${FOLD} --train_util_rate=0.5 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --size=128 --cosine --method=CE --learning_rate=0.0001
done
