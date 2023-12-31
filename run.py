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

if False: # create master DF with delta-cytokine covariates
    df = core.DataWrangl.get_cytokine_master_df(
        df = pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')),
        df_cyto = pd.read_csv(
            os.path.join(folders.raw, 'PDP1_cytokineData.csv')),)


""" plots """
if False: # make vitals

    plots.Controllers.make_vitals(
        df=pd.read_csv(os.path.join(folders.data, 'pdp1_vitals.csv')),
        out_dir=folders.vitals)

if False: # make histograms

    plots.Controllers.make_histograms(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')))

if True: # make agg time evolution plots

    plots.Controllers.make_agg_timeevols(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')),
            errorbar_corr = True,
            boost_y = False,
            out_dir = folders.agg_timeevols)

if False: # make ind time evolution plots

    plots.Controllers.make_ind_timeevols(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')),
            out_dir=folders.ind_timeevols)

if False: # make 5DASC compariosn plots

    plots.Controllers.make_5dasc(
        df=pd.read_csv(os.path.join(folders.data, 'pdp1_5dasc.csv')),
        out_dir=folders.fivedasc,
        horizontal=False)

if False: # make cytokine correlation matrix

    plots.Controllers.make_cytokine_corrmat(
        df=pd.read_csv(os.path.join(folders.exports, 'pdp1_cytokine_delta_corrmat.csv')),)

if False: # make bsl predictors correlation matrix

    plots.Controllers.make_bslpreds_corrmat(
        df = pd.read_csv(os.path.join(folders.exports, 'pdp1_bslpreds_delta_corrmat.csv')),
        filename = 'corrmat_bslpreds_delta',
        tp = 'B7')

    plots.Controllers.make_bslpreds_corrmat(
        df = pd.read_csv(os.path.join(folders.exports, 'pdp1_bslpreds_delta_corrmat.csv')),
        filename = 'corrmat_bslpreds_delta',
        tp = 'B30')


""" analysis """
if False: # make vitals analysis

    core.Analysis.vitals_dmax(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_vitals.csv')),)

    core.Analysis.vitals_avg(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_vitals.csv')),)

    core.Analysis.fivedasc_pairedt(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_5dasc.csv')),)

if False: # make observed scores table

    core.Analysis.observed_scores_df(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')),)
