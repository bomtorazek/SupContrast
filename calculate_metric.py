import csv
import numpy as np


for dataset in ["MLCCA", "MVTEC_pill", "MVTEC_screw", "MVTEC_leather"]:
    gt_path = f"{dataset}_GT.csv"
    GT = []
    with open(gt_path, 'r') as file:
        rows = csv.reader(file)
        for _, gt in rows:
            GT.append(int(gt))

    for network in ["resnet18", "resnet50", "resnet101","mobilenet_v2"]:
        probs = [ [] for _ in range(5)]
        epoch = 100 if dataset == "MLCCA" else 30
        batch = 32 if dataset == "MLCCA" else 8
        for trial in range(5,10):
            csv_name = f"CE_{dataset}_{network}_ur1_me{epoch}_lr_0.0001_decay_0.0001_aug_flip_crop_twocrop_True_bsz_{batch}_rsz_256_temp_0.08_sampling_unbalanced_whole_data_cosine_seed_100_trial_{trial}.csv"

            with open(csv_name, 'r') as file:
                rows = csv.reader(file)
                for _, prob in rows:
                    probs[trial-5].append(float(prob))

        probs = np.array(probs)
        averaged_probs = np.mean(probs, axis=0)
        averaged_preds = [int(prob>=0.5) for prob in averaged_probs]
        suakit_soft = [ (1 - prob) if gt==1 else prob for prob, gt in zip(averaged_probs, GT)]
        accuracy = sum(pred == gt for pred, gt in zip(averaged_preds, GT)) / len(GT)
        print(accuracy, dataset, network)

        with open(f"{dataset}_{network}_PRED2.csv", "w") as file:
            writer = csv.writer(file)
            for pred in averaged_preds:
                writer.writerow([pred])

        with open(f"{dataset}_{network}_SS2.csv", "w") as file:
            writer = csv.writer(file)
            for ss in suakit_soft:
                writer.writerow([ss])