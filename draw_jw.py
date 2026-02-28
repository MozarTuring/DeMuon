import json
import argparse
import pandas as pd
import random
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.lines import Line2D
import matplotlib as mpl
import os
import sys
sys.path.append('/Users/maojingwei/project/common_tools/')
from jwu1 import quick2json


mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['lines.linestyle'] = '-'  

all_markers = list(Line2D.markers.keys())
# Sample e.g. 8 markers randomly (excluding None and ' ')
markers = ["o", "s", "D", "^", "v"]
valid_markers = markers + [m for m in all_markers if m not in [None, " ", ""]+markers]

if __name__ == "__main__":
    color_sampled=plt.cm.tab10.colors[:4]
    plt.rcParams['axes.prop_cycle'] = cycler(color=color_sampled, marker=valid_markers[:4])
    plt.rcParams["font.family"] = "serif"      # e.g., serif, sans-serif, monospace
    plt.rcParams["font.size"] = 18
    inp_dir='/Users/maojingwei/project/jwmoutput/neurips_code_runs/'
    draw_dir='/Users/maojingwei/project/jwmoutput/neurips_code_draws/2'
    tmp_dic={}
    alg_ls = ['dsgd', 'dsgd_gclip_decay', 'gt_nsgdm', 'demuon']
    label_ls = ['DSGD', 'DSGD_Clip', 'GT_NSGDm', 'DeMuon']
    alg2label=dict(zip(alg_ls, label_ls))
    commit_ids = []
    print(alg2label)
    hhhh_dic=dict()
    for k in alg_ls:
        hhhh_dic[k] = []
    y_label_ls = ['Training loss', 'Validation loss']
    with open(os.path.join(draw_dir,'config.json'), 'r') as rf:
        config_dic = json.load(rf)
    for k, v in config_dic.items():
        if k == 'comment':
            continue
        tmp_dic[k]=dict()
        for y_label_ele in y_label_ls:
            fig, ax = plt.subplots()
            if y_label_ele not in tmp_dic[k]:
                tmp_dic[k][y_label_ele]={
                    'fig':fig,
                    'ax':ax
                }
        for kk, vv in v.items():
            if vv == "":
                continue

            csvpath=os.path.join(inp_dir, vv, 'loss.csv')
            df=pd.read_csv(csvpath)
            
            df['train_avg']=df[[f'w{i}_train' for i in range(8)]].mean(axis=1)
            tmp_dic[k]['Training loss']['ax'].plot(df['round'], df['train_avg'], markevery=10, label=alg2label[kk])
            df['val_avg']=df[[f'w{i}_val' for i in range(8)]].mean(axis=1)
            tmp_dic[k]['Validation loss']['ax'].plot(df['round'], df['val_avg'], markevery=10, label=alg2label[kk])

    for k, v in tmp_dic.items():
        for y_label_ele in y_label_ls:
            v[y_label_ele]['ax'].legend(fontsize=12, loc='upper right')
            v[y_label_ele]['ax'].set_xlabel('Iteration')
            v[y_label_ele]['ax'].grid(True)
            v[y_label_ele]['ax'].set_ylabel(y_label_ele)
            v[y_label_ele]['fig'].savefig(os.path.join(draw_dir, f'{k}_{y_label_ele}.pdf'),bbox_inches="tight",)


"""
conda activate
python -m pip install pandas
python -m pip install matplotlib
cd ~/project/neurips_code/ && python draw_jw.py
"""
