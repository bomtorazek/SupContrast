import csv
from collections import defaultdict

from pandas import *
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

pred_dict = defaultdict(dict)
ss_dict = defaultdict(dict)

pred_dict2 = defaultdict(dict)
ss_dict2 = defaultdict(dict)

for dataset in ["MLCCA", "MVTEC_pill", "MVTEC_screw", "MVTEC_leather"]:
    for network in ["resnet18", "resnet50", "resnet101","mobilenet_v2"]:

        with open(f"{dataset}_{network}_PRED.csv", "r") as file:
            rows = csv.reader(file)
            pred = np.array([float(row[0]) for row in rows])
                

        with open(f"{dataset}_{network}_SS.csv", "r") as file:
            rows = csv.reader(file)
            ss = np.array([float(row[0]) for row in rows])
        
        pred_dict[dataset][network] = pred
        ss_dict[dataset][network] = ss

        with open(f"{dataset}_{network}_PRED2.csv", "r") as file:
            rows = csv.reader(file)
            pred = np.array([float(row[0]) for row in rows])
                

        with open(f"{dataset}_{network}_SS2.csv", "r") as file:
            rows = csv.reader(file)
            ss = np.array([float(row[0]) for row in rows])
        
        pred_dict2[dataset][network] = pred
        ss_dict2[dataset][network] = ss



for dataset in ["MVTEC_leather", "MLCCA", "MVTEC_pill", "MVTEC_screw", "MVTEC_leather"]:
    network_list = ["mobilenet_v2","resnet18", "resnet50", "resnet101"]
    len_network = len(network_list)
    same_pred_mat = np.array([[None]*len_network for _ in range(len_network)])
    cor_pred_mat = np.array([[None]*len_network for _ in range(len_network)])
    cor_ss_mat = np.array([[None]*len_network for _ in range(len_network)])


    for i in range(len_network):
        for j in range(i, len(network_list)):
            if i==j:
                network_i = network_list[i]
                pred_i = pred_dict[dataset][network_i]
                pred_j = pred_dict2[dataset][network_i]
                ss_i = ss_dict[dataset][network_i]
                ss_j = ss_dict2[dataset][network_i]
            else:
                network_i = network_list[i]
                network_j = network_list[j]
                pred_i = pred_dict[dataset][network_i]
                pred_j = pred_dict[dataset][network_j]
                ss_i = ss_dict[dataset][network_i]
                ss_j = ss_dict[dataset][network_j]

            same_pred_mat[i][j] = same_pred_mat[j][i] = sum(pi == pj for pi, pj in zip(pred_i, pred_j))/len(pred_i)
            cor_pred_mat[i][j] =cor_pred_mat[j][i]= np.corrcoef(pred_i, pred_j)[1,0]
            cor_ss_mat[i][j] = cor_ss_mat[j][i] = np.corrcoef(ss_i, ss_j)[1,0]

    same_pred = DataFrame(same_pred_mat, columns = ["mobilenet_v2","resnet18", "resnet50", "resnet101"], index = ["mobilenet_v2","resnet18", "resnet50", "resnet101"]).astype(float)  
    cor_pred = DataFrame(cor_pred_mat, columns = ["mobilenet_v2","resnet18", "resnet50", "resnet101"], index = ["mobilenet_v2","resnet18", "resnet50", "resnet101"]).astype(float)
    cor_ss = DataFrame(cor_ss_mat, columns = ["mobilenet_v2","resnet18", "resnet50", "resnet101"], index = ["mobilenet_v2","resnet18", "resnet50", "resnet101"]).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(25, 10))
    fig.suptitle(f'{dataset}', fontsize=30)
    ax1 = sns.heatmap(same_pred, ax = axes[0], vmin = 0.5, vmax = 1, annot= True, fmt = '.2f',linewidths=.5)
    axes[0].set_title("Prediction")
    ax2 = sns.heatmap(cor_pred, ax = axes[1], vmin = 0.5, vmax = 1, annot= True, fmt = '.2f',linewidths=.5)
    axes[1].set_title("Prediction Correlation")
    ax3 = sns.heatmap(cor_ss, ax = axes[2], vmin = 0.5, vmax = 1, annot= True, fmt = '.2f',linewidths=.5)
    axes[2].set_title("Suakit-Soft Correlation")
    fig.savefig(f'{dataset}.png')


