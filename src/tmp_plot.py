import matplotlib.pyplot as plt
import src.folders as folders
import src.config as config
import seaborn as sns
import pandas as pd
import os

df=pd.read_csv(
    os.path.join(folders.processed, 'pdp1_vitals_v1.csv'))


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
measure = 'temp' #['dia', 'hr', 'sys', 'temp']

ax = sns.lineplot(
    data = df.loc[(df.measure==measure)],
    x = 'time',
    y = 'score',
    hue = 'tp',
    errorbar = ("se"),
    err_style = "bars",
    err_kws={"capsize": 5, "elinewidth": 1.5},
    style="tp",
    markers="D",
    markersize=10,
    dashes=False,
)


ax.set_xticks([0, 30, 60, 90, 120, 240, 360, 420])
ax.set_xlabel('Time [min]')
ax.set_xlabel('Time (min)')
plt.show()
