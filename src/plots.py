import matplotlib.pyplot as plt
import src.config as config
import src.folders as folders
import seaborn as sns
import pandas as pd
import numpy as np
import os

class Controllers():

    @staticmethod
    def make_histograms(df, ignore_measure=[], out_dir=folders.histograms):

        for measure in df.measure.unique():
            print(f'Create HIST plot: {measure}')

            if measure in ignore_measure:
                continue

            tmp_df = df.loc[(df.measure==measure)]
            tps = tmp_df.tp.unique()
            n_subplots = len(tps)

            fig, axs = plt.subplots(
                nrows=1,
                ncols=n_subplots,
                figsize=(n_subplots*4, 4))

            for idx, tp in enumerate(tps):

                ax = axs[idx]

                sns.histplot(
                    x='score', data=tmp_df.loc[(tmp_df.tp==tps[idx])],
                    ax = ax)

                ax.set_xlabel('Score', fontdict=config.axislabel_fontdict)
                ax.set_ylabel('Count', fontdict=config.axislabel_fontdict)
                ax.tick_params(axis='both', which='major', labelsize=config.ticklabel_fontsize)
                ax.set_title(f'{measure} at {tp}', fontdict=config.title_fontdict)

            plt.tight_layout()

            Helpers.save_fig(
                fig = fig,
                out_dir = out_dir,
                filename = f'histogram_{measure}')

    @staticmethod
    def make_agg_timeevols(df, errorbar_corr=True, boost_y=True, out_dir=folders.agg_timeevols):

        for measure in df.measure.unique():
            print(f'Create AGG timeevol plot: {measure}; errorbar_corr: {errorbar_corr}, boost_y: {boost_y}')

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            tmp_df = df.loc[(df.measure==measure)].copy()
            tps = tmp_df.tp.unique()

            tmp_df['tp'] = tmp_df['tp'].astype(
                pd.CategoricalDtype(
                categories=['bsl', 'A7', 'B7', 'B30'],
                ordered=True))

            if errorbar_corr:
                tmp_df = Helpers.get_errorbar_corr(tmp_df)

            sns.lineplot(
                x='tp', y='score', data=tmp_df,
                marker='o', markersize=12,
                err_style="bars", errorbar="ci", err_kws={'capsize':4, 'elinewidth': 1.5, 'capthick': 1.5})

            ax.xaxis.grid(False)

            if boost_y:
                y_high = ax.get_ylim()[1]
                y_low  = ax.get_ylim()[0]
                y_boost = 0.3*(y_high-y_low)
                ax.set_ylim([y_low, y_high+y_boost])

            ax.set_xlabel('Timepoint', fontdict=config.axislabel_fontdict)
            ax.set_ylabel('Score', fontdict=config.axislabel_fontdict)
            ax.tick_params(axis='both', which='major', labelsize=config.ticklabel_fontsize)
            ax.set_title(measure, fontdict=config.title_fontdict)

            Helpers.save_fig(
                fig = fig,
                out_dir = out_dir,
                filename = f'pdp1_agg_timeevol_{measure}')

    @staticmethod
    def make_ind_timeevols(df, out_dir=folders.ind_timeevols):

        for measure in df.measure.unique():
            print(f'Create IND timeevol plot: {measure}')

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            tmp_df = df.loc[(df.measure==measure)].copy()
            tps = tmp_df.tp.unique()

            tmp_df['tp'] = tmp_df['tp'].astype(
                pd.CategoricalDtype(
                categories=['bsl', 'A7', 'B7', 'B30'],
                ordered=True))

            sns.lineplot(
                x='tp',
                y='score',
                hue='pID',
                palette='muted',
                dashes=False,
                data=tmp_df,
                #marker='o',
                markersize=10,)

            plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
            fig.set_size_inches([9.6, 4.8])

            ax.xaxis.grid(False)
            ax.set_xlabel('Timepoint', fontdict=config.axislabel_fontdict)
            ax.set_ylabel('Score', fontdict=config.axislabel_fontdict)
            ax.tick_params(axis='both', which='major', labelsize=config.ticklabel_fontsize)
            ax.set_title(measure, fontdict=config.title_fontdict)

            Helpers.save_fig(
                fig = fig,
                out_dir = out_dir,
                filename = f'pdp1_ind_timeevol_{measure}')

    @staticmethod
    def make_vitals(df, y, errorbar_corr=True, out_dir=folders.vitals):

        assert y in ['temp', 'dia', 'sys', 'hr']

        """ Plot temperature """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        tmp_df = df.loc[(df.measure==y)]

        if errorbar_corr:
            tmp_df = Helpers.get_errorbar_corr(tmp_df)

        ax = sns.lineplot(
            data = tmp_df,
            x = 'time',
            y = 'score',
            hue = 'tp',
            errorbar = ("se"),
            err_style = "bars",
            err_kws={"capsize": 5, "elinewidth": 1.5},
            style="tp",
            markers=["D", "D"],
            markersize=10,
            dashes=False,
            legend=False,
            palette = 'deep'
        )

        ax.set_xticks([0, 30, 60, 90, 120, 240, 360, 420])
        ax.set_xlabel('Time [min]', fontdict=config.axislabel_fontdict)
        ax.tick_params(axis='both', which='major', labelsize=config.ticklabel_fontsize)
        ax.xaxis.grid(False)

        if y=='temp':
            ax.set_ylabel(
                'Body temperature [°C]',
                fontdict=config.axislabel_fontdict)
        elif y=='dia':
            ax.set_ylabel(
                'Diastolic BP [mmHg]',
                fontdict=config.axislabel_fontdict)
        elif y=='sys':
            ax.set_ylabel(
                'Systolic BP [mmHg]',
                fontdict=config.axislabel_fontdict)
        elif y=='hr':
            ax.set_ylabel(
                'Heart rate [BPM]',
                fontdict=config.axislabel_fontdict)
        else:
            assert False

        Helpers.save_fig(
            fig = fig,
            out_dir = out_dir,
            filename = f'vitals_{y}')


class Helpers:

    @staticmethod
    def save_fig(fig, out_dir, filename):

        if config.savePNG:
            fig.savefig(
                fname=os.path.join(out_dir, f'{filename}.png'),
                format='png',
                dpi=300,)

        if config.saveSVG:
            fig.savefig(
                fname=os.path.join(out_dir, f'{filename}.svg'),
                format='svg',
                dpi=300,)

        plt.close()

    @staticmethod
    def get_errorbar_corr(df):
        """ Create adjustment factor: (grand_mean - each_subject_mean)
            Adjust error bars for within subject design, see:
                - https://stats.stackexchange.com/questions/574379/correcting-repeated-measures-data-to-display-error-bars-that-show-within-subject
                - https://www.cogsci.nl/blog/tutorials/156-an-easy-way-to-create-graphs-with-within-subject-error-bars
        """

        df = df.reset_index()
        grand_mean = df['score'].mean()
        df['ws_adj_factor']=0
        col_idx = df.columns.get_loc('ws_adj_factor')

        for pID in df.pID.unique():
            subject_mean = df.loc[(df.pID==pID), 'score'].mean()
            rows = df.loc[df.pID==pID].index
            df.iloc[rows, col_idx] = grand_mean - subject_mean

        subset = df[['score', 'ws_adj_factor']].copy()
        df.loc[:, 'adj_score'] = subset['score'] + subset['ws_adj_factor']

        df = df.drop(columns=['score', 'ws_adj_factor'])
        df = df.rename(columns={'adj_score': 'score'})

        return df
