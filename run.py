"""
To reproduce findings:
    - run from the included conda environment
    - run from codebase folder
"""
import src.folders as folders
import src.config as config
import src.plots as plots
import src.core as core
import pandas as pd
import os


""" process data """
if False: # create master DF from raw data files
    df = core.DataWrangl.get_master_df()

if False: # create 5DASC DF from raw data files
    df_5d = core.DataWrangl.get_5dasc_df()

if False: # create vitals DF from raw data files
    df_vitals = core.DataWrangl.get_vitals_df()

if False: # create master DF with all potential bsl covariates
    df = core.DataWrangl.get_covariates_master_df(
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
            os.path.join(folders.data, 'pdp1_vitals.csv')),)

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
            errorbar_corr = True,
            boost_y = False,
            out_dir = folders.agg_timeevols)

    plots.Controllers.make_agg_timeevols(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')),
            errorbar_corr=True,
            boost_y=True,
            out_dir=os.path.join(folders.agg_timeevols, 'boost_y'))

if True: # make 5DASC compariosn plots

    plots.Controllers.make_5dasc(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_5dasc.csv')),)


""" analysis """
if False: # make delta max

    core.Analysis.delta_max_vitals(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_vitals.csv')),)

    core.Analysis.delta_max_5DASC(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_5dasc.csv')),)
