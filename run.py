"""
To reproduce findings:
    - run from the included conda environment
    - run from codebase folder
"""
import src.format_data as data
import src.analysis as analysis
import src.folders as folders
import src.config as config
import src.plots as plots
import pandas as pd
import os


""" process data """
if False: # create master DF from raw data files
    df = data.Controllers.get_master_df(
        save=False,
        process_raws=True)

if False: # create 5DASC DF from raw data files
    df_5d = data.Controllers.get_5dasc_df(
        save=False,)

if False: # create vitals DF from raw data files
    df_vitals = data.Controllers.get_vitals_df(
        save=False,)


""" plots """
if False: # make vitals
    plots.Controllers.make_vitals(
        df=pd.read_csv(
            os.path.join(folders.processed, 'pdp1_vitals_v1.csv')),
            os.path.join(folders.processed, 'pdp1_vitals_v1.csv')),

        y='temp',)

    plots.Controllers.make_vitals(
        df=pd.read_csv(
            os.path.join(folders.processed, 'pdp1_vitals_v1.csv')),
        y='hr',)

    plots.Controllers.make_vitals(
        df=pd.read_csv(
            os.path.join(folders.processed, 'pdp1_vitals_v1.csv')),
        y='sys',)

    plots.Controllers.make_vitals(
        df=pd.read_csv(
            os.path.join(folders.processed, 'pdp1_vitals_v1.csv')),
        y='dia',)

if False: # make histograms
    plots.Controllers.make_histograms(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER_v1.csv')))

if False: # make time evolution plots

    plots.Controllers.make_ind_timeevols(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER_v1.csv')),
            out_dir=folders.ind_timeevols)

    plots.Controllers.make_agg_timeevols(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER_v1.csv')),
            errorbar_corr=True,
            boost_y=True,
            out_dir=os.path.join(folders.timeevols,'boost_y'))


""" analysis """
if False: # make delta max

    analysis.Controllers.delta_max_vitals(
        df=pd.read_csv(
            os.path.join(folders.processed, 'pdp1_vitals_v1.csv')),
    )

    '''
    analysis.Controllers.delta_max_5DASC(
        df=pd.read_csv(
            os.path.join(folders.processed, 'pdp1_5dasc_v1.csv')),)
    '''
