"""
To reproduce findings:
    - run from the included conda environment
    - run from codebase folder
"""
import src.data_wrangling as data
import src.analysis as analysis
import src.folders as folders
import src.config as config
import src.plots as plots
import pandas as pd
import os


""" process data """
if True: # create master DF from raw data files
    df = data.Controllers.get_master_df()

if False: # create 5DASC DF from raw data files
    df_5d = data.Controllers.get_5dasc_df()

if False: # create vitals DF from raw data files
    df_vitals = data.Controllers.get_vitals_df()

if False: # create master DF with all potential bsl covariates
    df = data.Controllers.get_covariates_master_df(
        df = pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')),
        df_5dasc = pd.read_csv(
            os.path.join(folders.data, 'pdp1_5dasc.csv')),
        df_demo = pd.read_csv(
            os.path.join(folders.data, 'pdp1_demography.csv')),)


""" plots """
if False: # make vitals
    plots.Controllers.make_vitals(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_vitals.csv')),
        y='temp',)

    plots.Controllers.make_vitals(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_vitals.csv')),
        y='hr',)

    plots.Controllers.make_vitals(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_vitals.csv')),
        y='sys',)

    plots.Controllers.make_vitals(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_vitals.csv')),
        y='dia',)

if False: # make histograms
    plots.Controllers.make_histograms(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')))

if False: # make agg/ind time evolution plots

    plots.Controllers.make_ind_timeevols(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')),
            out_dir=folders.ind_timeevols)

    plots.Controllers.make_agg_timeevols(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')),
            errorbar_corr=True,
            boost_y=False,
            out_dir=folders.agg_timeevols)

    plots.Controllers.make_agg_timeevols(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')),
            errorbar_corr=True,
            boost_y=True,
            out_dir=os.path.join(folders.agg_timeevols, 'boost_y'))


""" analysis """
if False: # make delta max

    analysis.Controllers.delta_max_vitals(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_vitals.csv')),)

    analysis.Controllers.delta_max_5DASC(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_5dasc.csv')),)
