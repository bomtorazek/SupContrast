for FOLD in 1 2 3 4 5; do
        python main_joint_domain_whole.py --dp --trial=dp256 --temp=0.08 --feat_dim=256 --batch_size=64 --val_fold=${FOLD} --head=mlp --train_util_rate=0.1 --epochs=200 --weight_decay=0 --dataset=MLCCA --data_folder=/data/esuh/mlcc/mlcc_crop/05A226_crop/original --source_data_folder=/data/esuh/mlcc/mlcc_crop/05B104_crop/original --size=128 --cosine --method=Joint_Con_Whole --learning_rate=0.0001    
done
