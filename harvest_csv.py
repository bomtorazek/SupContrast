import pandas as pd
import numpy as np
import re
#######
# FIXME
#target_regex = '.*D_Sub_.*999.*'
target_regex = '.*SolarPannel_.*'
#######

regex = re.compile(target_regex)

csv = pd.read_csv('./result.csv', sep=',', index_col=0, names=range(7))

ls_name = csv.index.to_list()
ls_key = list(set([name.split('trial_')[0] for name in ls_name if regex.match(name)]))

# dictionary version
dic = {}
for key in ls_key:
    #if not regex.match(key):
    #    continue
    for i in range(0,3):
        idx = key+'trial_{}'.format(i)
        if idx in ls_name:
            val = csv.loc[idx]
            if len(val.shape) == 2:
                #val = val.iloc[0] # only first row
                val = val.iloc[-1] # only last row
            val_k = [vvk for ivk, vvk in enumerate(val) if ivk%2==0]
            val_v = [vvk for ivk, vvk in enumerate(val) if ivk%2==1]
            dic_val = {k:v for k,v in zip(val_k, val_v)}
        else:
            dic_val = {key: 0 for key in dic_val.keys()}

        if i == 0:
            dic[key] = {vvk:[] for vvk in val_k}

        for k,v in dic_val.items():
            dic[key][k].append(v)

# data frame version
df = pd.DataFrame([])
for exp, expv in dic.items():
    rowidx = []
    rowval = []
    for rk, rv in expv.items():
        temp_rowidx = [rk+'_fold{}'.format(i+1) for i in range(3)] + [rk+'_mean']
        rowidx.extend(temp_rowidx)
        rv.append(np.mean(rv))
        rowval.extend(rv)
    df_temp = pd.DataFrame([rowval], index=[exp], columns=rowidx)
    df = df.append(df_temp, ignore_index=False)
df = df.sort_index()
df.to_csv('./modified_records.csv')
