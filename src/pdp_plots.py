#from itertools import product as product
import matplotlib.pyplot as pyplt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
import os

output_dir="C://Users//szb37//My Drive//Projects//PDP1//codebase//export figures"

pyplt.rcParams.update({'font.family': 'arial'})
title_fontdict = {'fontsize': 20, 'fontweight': 'bold'}
axislabel_fontdict = {'fontsize': 16, 'fontweight': 'bold'}
ticklabel_fontsize = 14
sns.set_style("darkgrid")

export_folder = "C://Users//szb37//My Drive//Projects//PDP1//codebase//export results"
model_df = pd.read_csv(os.path.join(export_folder,'pdp1_mixed_models_v1.csv'))
data_folder = "C://Users//szb37//My Drive//Projects//PDP1//codebase//data"
data_df = pd.read_csv(os.path.join(data_folder,'pdp1_v2.csv'))
data_df.replace('A7', 'D1+7', inplace=True)
data_df.replace('B7', 'D2+7', inplace=True)
data_df.replace('B30', 'D2+30', inplace=True)

for measure in data_df.measure.unique():
    print(measure)

    fig = pyplt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.lineplot(
        x='tp', y='result', data=data_df.loc[(data_df.measure==measure)],
        marker='o', markersize=12,
        err_style="bars", errorbar="ci", err_kws={'capsize':4, 'elinewidth': 1.5, 'capthick': 1.5})

    y_high = ax.get_ylim()[1]
    y_low  = ax.get_ylim()[0]
    y_boost = 0.3*(y_high-y_low)
    ax.set_ylim([y_low, y_high+y_boost])

    ax.set_xlabel('Timepoint', fontdict=axislabel_fontdict)
    ax.set_ylabel('Score', fontdict=axislabel_fontdict)
    ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
    ax.set_title(measure, fontdict=title_fontdict)
    #fig.show()

    dpi=300
    fig.savefig(
        fname=os.path.join(output_dir, f'pdp1_{measure}.png'),
        format='png',
        dpi=dpi,
    )
    fig.savefig(
        fname=os.path.join(output_dir, f'pdp1_{measure}.svg'),
        format='svg',
        dpi=dpi,
    )
    del fig
