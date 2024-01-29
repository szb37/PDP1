import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'arial'})
plt.rcParams['svg.fonttype'] = 'none'  # Ensure fonts are embedded
plt.rcParams['text.usetex'] = False  # Use TeX to handle text (embeds fonts)
title_fontdict = {'fontsize': 20, 'fontweight': 'bold'}
axislabel_fontdict = {'fontsize': 20, 'fontweight': 'bold'}
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

all_outcomes = [
    'UPDRS_1', 'UPDRS_2', 'UPDRS_3', 'UPDRS_4', 'UPDRS_SUM',
    'HAMA', 'MADRS', 'NPIQ_DIS', 'NPIQ_SEV',  'CSSRS', 'ESAPS', 'CCFQ',
    'PRL', 'Z_MTS', 'Z_OTS', 'Z_PAL', 'Z_RTI', 'Z_SWM',
    'MTSCFAPC', 'MTSPS82', 'MTSRCAMD', 'MTSRFAMD', 'OTSMDLFC', 'OTSPSFC',
    'PALFAMS', 'PALTEA', 'RTIFMDMT', 'RTIFMDRT', 'RTISMDMT', 'RTISMDRT',
    'SWMBE12', 'SWMBE4', 'SWMBE468', 'SWMBE6', 'SWMBE8', 'SWMS',]

outcomes = [
    'UPDRS_1', 'UPDRS_2', 'UPDRS_3', 'UPDRS_4', 'UPDRS_SUM',
    'HAMA', 'MADRS', 'NPIQ_DIS', 'NPIQ_SEV',  'CSSRS', 'ESAPS', 'CCFQ',
    'PRL', 'Z_MTS', 'Z_OTS', 'Z_PAL', 'Z_RTI', 'Z_SWM', ]
