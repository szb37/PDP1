import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.color_palette("colorblind")
sns.despine()

savePNG=True
saveSVG=False

plt.rcParams.update({'font.family': 'arial'})
title_fontdict = {'fontsize': 20, 'fontweight': 'bold'}
axislabel_fontdict = {'fontsize': 16, 'fontweight': 'bold'}
ticklabel_fontsize = 14

valid_pIDs=[1002,1020,1034,1047,1051,1055,1083,1085,1086,1129,1142,1145]
valid_str_pIDs=['1002','1020','1034','1047','1051','1055','1083','1085','1086','1129','1142','1145']
