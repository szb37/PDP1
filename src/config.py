import matplotlib.pyplot as plt
#import seaborn as sns

plt.rcParams.update({'font.family': 'arial'})
title_fontdict = {'fontsize': 20, 'fontweight': 'bold'}
axislabel_fontdict = {'fontsize': 16, 'fontweight': 'bold'}
ticklabel_fontsize = 14

savePNG=True
saveSVG=True

valid_pIDs=[1002,1020,1034,1047,1051,1055,1083,1085,1086,1129,1142,1145]
valid_str_pIDs=['1002','1020','1034','1047','1051','1055','1083','1085','1086','1129','1142','1145']

cantab_measures={
  'PAL': ['PALFAMS','PALTEA'], # Memory
  'RTI': ['RTIFMDMT','RTIFMDRT','RTISMDMT', 'RTISMDRT'],  # Attention & Psychomotor Speed
  'MTS': ['MTSCFAPC', 'MTSPS82','MTSRCAMD','MTSRFAMD'], # Attention & Psychomotor Speed ('MTSCTAPC' removed as all values=100)
  'OTS': ['OTSMDLFC', 'OTSPSFC'], # Executive Function
  'SWM': ['SWMBE12','SWMBE4','SWMBE468','SWMBE6','SWMBE8','SWMS'], # Executive Function
}

corr_types = {
    'pearson':{
        'est': 'pearson_cor',
        'p': 'pearson_p',
        'sig': 'pearson_sig'
    },
    'spearman':{
        'est': 'spearman_rho',
        'p': 'spearman_p',
        'sig': 'spearman_sig'
    },
    'kendall':{
        'est': 'kendall_tau',
        'p': 'kendall_p',
        'sig': 'kendall_sig'
    },
}
