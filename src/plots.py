import matplotlib.pyplot as plt
import src.config as config
import src.folders as folders
import seaborn as sns
import pandas as pd
import numpy as np
import math
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

            tmp_df['tp'] = tmp_df['tp'].astype(
                pd.CategoricalDtype(
                categories=['bsl', 'A7', 'B7', 'B30'],
                ordered=True))

            if errorbar_corr:
                tmp_df = Helpers.get_errorbar_corr(tmp_df)

            sns.lineplot(
                x='tp',
                y='score',
                data=tmp_df,
                marker='o',
                markersize=12,
                #color = 'black',
                err_style="bars",
                errorbar="ci",
                err_kws={
                    'capsize':4,
                    'elinewidth': 1.5,
                    'capthick': 1.5})

            ax = Helpers.set_yaxis(measure, ax, boost_y)

            ax.set_title(measure, fontdict=config.title_fontdict)
            ax.set_xlabel('Timepoint', fontdict=config.axislabel_fontdict)
            ax.set_ylabel('Score', fontdict=config.axislabel_fontdict)

            ax.tick_params(axis='both', which='major', labelsize=config.ticklabel_fontsize)
            ax.tick_params(axis='y', which='both', left=False, labelleft=True)
            ax.tick_params(axis='x', direction='inout', length=8)

            sns.despine(top=True, right=True, left=True, bottom=True)
            #ax.spines['bottom'].set_color('black')
            ax.yaxis.grid(True, linewidth=0.5, alpha=.75)
            ax.xaxis.grid(False)

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

            sns.despine(top=True, right=True, left=True, bottom=True)

            Helpers.save_fig(
                fig = fig,
                out_dir = out_dir,
                filename = f'pdp1_ind_timeevol_{measure}')

    @staticmethod
    def make_vitals(df, errorbar_corr=True, out_dir=folders.vitals):

        for y in ['temp', 'dia', 'sys', 'hr']:

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
                errorbar = "ci",
                err_style = "bars",
                err_kws = {"capsize": 5, "elinewidth": 1.5},
                style = "tp",
                markers = ["o", "D"],
                palette = {
                    'A0': '#56A0FB',
                    'B0': '#F71480'},
                markersize = 10,
                #dashes = False,
                legend = True,
            )

            ax.set_xticks([0, 30, 60, 90, 120, 240, 360, 420])
            ax.set_xlabel('Time [min]', fontdict=config.axislabel_fontdict)
            ax.tick_params(axis='both', which='major', labelsize=config.ticklabel_fontsize)

            sns.despine(top=True, right=True, left=True, bottom=True)
            ax.yaxis.grid(True, linewidth=0.5, alpha=.75)
            ax.xaxis.grid(False)

            plt.legend(title='Psilocybin doses', labels=['10mg (A7)', '25mg (B7)'])

            if y=='temp':
                ax.set_yticks([35.9, 36.1, 36.3, 36.5])
                ax.set_ylabel(
                    'Body temperature [°C]',
                    fontdict=config.axislabel_fontdict)
            elif y=='hr':
                ax.set_yticks([65, 70, 75, 80])
                ax.set_ylabel(
                    'Heart rate [BPM]',
                    fontdict=config.axislabel_fontdict)
            elif y=='dia':
                ax.set_yticks([70, 74, 78, 82])
                ax.set_ylabel(
                    'Diastolic BP [mmHg]',
                    fontdict=config.axislabel_fontdict)
            elif y=='sys':
                ax.set_yticks([120, 130, 140, 150])
                ax.set_ylabel(
                    'Systolic BP [mmHg]',
                    fontdict=config.axislabel_fontdict)
            else:
                assert False

            Helpers.save_fig(
                fig = fig,
                out_dir = out_dir,
                filename = f'vitals_{y}')

    @staticmethod
    def make_5dasc(df, out_dir=folders.exports):

        df = df[df['measure'].str.contains('fivedasc_')]
        tmp = df['measure'].copy()
        tmp = tmp.str.replace('fivedasc_', '')
        tmp = tmp.str.replace('_total', '')
        df['measure'] = tmp

        df = df.replace({
            'util': 'Experience\nof unity',
            'sprit': 'Spiritual\nexperience',
            'bliss': 'Blissful\nstate',
            'insight': 'Insightfulness',
            'dis': 'Disembodiment',
            'imp': 'Impaired control\nand cognition',
            'anx': 'Anxiety',
            'cimg': ' Complex\nimagery',
            'eimg': 'Elementary\nimagery',
            'av': 'Audio-Visual\nsynesthesia',
            'per': 'Changed meaning\nof percepts',}, regex=True,)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.set_size_inches([4*4.8, 4.8])

        sns.boxplot(
            data=df,
            x='measure',
            y='score',
            hue='tp',
            palette = {
                'A0': '#56A0FB',
                'B0': '#F71480'},)

        ax.tick_params(axis='y', which='major', labelsize=config.ticklabel_fontsize)
        ax.tick_params(axis='x', which='major', labelsize=13)
        ax.set_ylabel('Score', fontdict=config.axislabel_fontdict)
        ax.set_xlabel('')
        ax.set_yticks([25, 75])
        sns.despine(top=True, right=True, left=True, bottom=True)
        ax.yaxis.grid(True, linewidth=0.5, alpha=.75)
        ax.xaxis.grid(False)

        Helpers.save_fig(
            fig = fig,
            out_dir = out_dir,
            filename = '5dasc')

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

    @staticmethod
    def set_yaxis(measure, ax, boost_y):

        if measure=='MADRS':
            ax.set_yticks([10, 14, 18, 22])
            plt.axhline(y=19, color='r', linestyle='--', alpha=0.5)
        elif measure=='Z_SWM':
            ax.set_yticks([-0.3, 0, 0.3, 0.6])
        elif measure=='NPIQ_SEV':
            ax.set_yticks([1, 3, 5, 7])
        elif measure=='UPDRS_1':
            ax.set_yticks([7, 12, 17, 22])
        elif measure=='UPDRS_2':
            ax.set_yticks([8, 11, 14, 17])
        elif measure=='UPDRS_3':
            ax.set_yticks([32, 34, 36, 38])

        if boost_y:
            y_high = ax.get_ylim()[1]
            y_low  = ax.get_ylim()[0]
            y_boost = 0.3*(y_high-y_low)
            ax.set_ylim([y_low, y_high+y_boost])

        return ax
