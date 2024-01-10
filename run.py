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

if False: # create vitals DF from raw data files
    df_vitals = core.DataWrangl.get_vitals_df()

if True: # create master DF with all potential bsl covariates
    df = core.DataWrangl.add_covs_df(
        df = pd.read_csv(os.path.join(folders.exports, 'pdp1_data_master.csv')),)


""" plots """
if False: # make vitals

    plots.Controllers.make_vitals(
        df=pd.read_csv(os.path.join(folders.data, 'pdp1_vitals.csv')),
        out_dir=folders.vitals)

if False: # make histograms

    plots.Controllers.make_histograms(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')))

if False: # make ind time evolution plots

    plots.Controllers.make_ind_timeevols(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')),
            out_dir=folders.ind_timeevols)

if False: # make agg time evolution plots

    plots.Controllers.make_agg_timeevols(
        df=pd.read_csv(
            os.path.join(folders.data, 'pdp1_MASTER.csv')),
            errorbar_corr = True,
            boost_y = False,
            out_dir = folders.agg_timeevols)

if False: # make 5DASC comparison plots

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

if False: # make 5DASC correlation matrix

    plots.Controllers.make_5dasc_corrmat(
        df = pd.read_csv(os.path.join(folders.data, 'pdp1_5dasc.csv')),
        filename = 'corrmat_5dasc_delta',)


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
