from numpy import median, mean
import pandas as pd
import itertools
import math
import os

data_folder = "C://Users//szb37//My Drive//Projects//PDP1//codebase//data"
data_df = pd.read_csv(os.path.join(data_folder,'pdp1_cantab_v0.1.csv'))
data_df.replace('A7', 'D1+7', inplace=True)
data_df.replace('B7', 'D2+7', inplace=True)
data_df.replace('B30', 'D2+30', inplace=True)

df = data_df.filter(items=['pID', 'tp', 'measure', 'result','standard_score', 'perc'])
df = df.dropna()

# Has PERC: 'PALTEA', 'PALFAMS', 'OTSPSFC', 'SWMBE468', 'SWMS'
# Has PERC and sig change: 'PALTEA', 'PALFAMS', 'SWMBE468'
measures = ['PALTEA', 'PALFAMS', 'SWMBE468']
tps = ['bsl', 'D1+7', 'D2+7', 'D2+30']

for measure, tp in itertools.product(measures, tps):
    med = round(median(df.loc[(df.tp==tp) & (df.measure==measure)].perc))
    print(f'Median percentage at {tp} of {measure}: {med}')

for measure, tp in itertools.product(measures, tps):
    avg = round(mean(df.loc[(df.tp==tp) & (df.measure==measure)].perc))
    print(f'Mean percentage at {tp} of {measure}: {avg}')
