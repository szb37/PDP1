import src.format_data as data
import src.folders as folders
import src.config as config
import src.plots as plots
import pandas as pd
import os

if True: # create master DF from raw data files
    df = data.Controllers.get_master_df()

if False: # make histograms
    plots.Controllers.make_histograms(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_master_v1.csv')))

if False: # make time evolution plots
    plots.Controllers.make_timeevolutions(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_master_v1.csv')))
