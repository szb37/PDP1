"""
To reproduce findings:
    - run from the included conda environment
    - run from codebase folder
"""
import src.format_data as data
import src.folders as folders
import src.config as config
import src.plots as plots
import pandas as pd
import os


""" process data """
if False: # create master DF from raw data files
    df = data.Controllers.get_master_df(
        save=True,
        process_raws=True)

if False: # create master DF from raw data files
    df_5d = data.Controllers.get_5dasc_df(
        save=True,)

if True: # create master DF from raw data files
    df_vitals = data.Controllers.get_vitals_df(
        save=True,)


""" plots """
if False: # make histograms
    plots.Controllers.make_histograms(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER_v1.csv')))

if False: # make time evolution plots
    plots.Controllers.make_timeevolutions(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER_v1.csv')),
            within_sub_errorbar=True,
            boost_y=False,
            output_dir=folders.timeevols_dir,)

    plots.Controllers.make_timeevolutions(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER_v1.csv')),
            within_sub_errorbar=True,
            boost_y=True,
            output_dir=os.path.join(folders.timeevols_dir,'boost_y'))
