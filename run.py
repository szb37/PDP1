import src.folders as folders
import src.config as config
import src.plots as plots
import src.core as core
import pandas as pd
import os


""" process data """
if True: # create master DF from raw data files
    df = core.Controllers.get_master_df(save=True)

if False: # create vitals DF from raw data files
    df_vitals = core.Controllers.get_vitals_df(save=True)

if False: # create master DF with all potential bsl covariates
    df_wcovs = core.Controllers.add_covs_df(
        df = pd.read_csv(os.path.join(folders.exports, 'pdp1_data_master.csv')),)

if False: # calculate corrmats
    core.Core.get_corrmats_df(
        df = pd.read_csv(os.path.join(folders.exports, 'pdp1_data_wcovs.csv')),)


""" plots """
if False: # make vitals
    plots.Controllers.make_vitals(
        df=pd.read_csv(os.path.join(folders.data, 'pdp1_vitals.csv')),
        out_dir=folders.vitals)

if False: # make histograms
    plots.Controllers.make_histograms(
        df=pd.read_csv(
            os.path.join(folders.exports, 'pdp1_data_master.csv')))

if False: # make ind time evolution plots
    plots.Controllers.make_ind_timeevols(
        df=pd.read_csv(
            os.path.join(folders.exports, 'pdp1_data_master.csv')),
            out_dir=folders.ind_timeevols)

if False: # make agg time evolution plots
    plots.Controllers.make_agg_timeevols(
        df=pd.read_csv(
            os.path.join(folders.exports, 'pdp1_data_master.csv')),
        out_dir = folders.agg_timeevols)

if False: # make 5DASC comparison plots
    plots.Controllers.make_5dasc(
        df=pd.read_csv(os.path.join(folders.data, 'pdp1_5dasc.csv')),
        out_dir=folders.fivedasc,)

if False: # make TSQ plot
    plots.Controllers.make_tsq(
        df=pd.read_csv(os.path.join(folders.data, 'pdp1_tsq.csv')),
        out_dir=folders.tsq)


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
            os.path.join(folders.exports, 'pdp1_data_master.csv')),)
