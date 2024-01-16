from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import src.config as config
import src.folders as folders
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os

#plt.rcParams['font.weight'] = 'bold'

class Controllers():

    @staticmethod
    def make_histograms(df, ignore_measure=[], out_dir=folders.histograms):

        for measure in config.measures:
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
    def make_agg_timeevols(df, measures=config.outcomes, errorbar_corr=True, out_dir=folders.agg_timeevols):

        has_B90=['HAMA', 'MADRS', 'NPIQ_SEV', 'NPIQ_DIS']
        sns.set_context("paper", font_scale=1.75)

        for measure in measures:

            print(f'Create AGG timeevol plot: {measure}; errorbar_corr: {errorbar_corr}')

            fig = plt.figure(figsize=((4.8, 6.4)))
            ax = fig.add_subplot(1, 1, 1)

            df_measure = df.loc[(df.measure==measure)].copy()
            df_measure['tp'] = df_measure['tp'].replace({
                'bsl': 0 ,
                'A7': 24,
                'B7': 24+14,
                'B30': 24+14-7+30,
                'B90': 24+14-7+90,})

            if errorbar_corr:
                df_measure = Helpers.apply_errorbar_correction(df_measure)

            sns.lineplot(
                x='tp',
                y='score',
                data=df_measure,
                marker='o',
                markersize=12,
                color = '#00317f',
                err_style="bars",
                errorbar="ci",
                err_kws={
                    'capsize': 4,
                    'elinewidth': 0.75,
                    'capthick': 0.75})

            if measure in has_B90:
                intervals = [0, 24, 24+14, 24+14-7+30, 24+14-7+90]
                ax.set_xticks(intervals)
                plt.xticks(intervals, ['Baseline', 'A7', 'B7', 'B30', 'B90'])

            else:
                intervals = [0, 24, 24+14, 24+14-7+30]
                ax.set_xticks(intervals)
                plt.xticks(intervals, ['Baseline', '7d post 10mg', '7d post 25mg', '30d post 25mg'])

            ax.set_ylabel(measure)
            ax = Helpers.set_yaxis(measure, ax, boost_y=False)
            ax.set_xlabel('')

            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

            sns.despine(top=True, right=True, left=False, bottom=False, offset=10, trim=True)
            #ax.tick_params(axis='both', which='major', labelsize=config.ticklabel_fontsize)
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

            Helpers.save_fig(
                fig = fig,
                out_dir = out_dir,
                filename = f'pdp1_agg_timeevol_{measure}_scale1.75')

    @staticmethod
    def make_ind_timeevols(df, measures=config.outcomes, out_dir=folders.ind_timeevols):

        for measure in measures:
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

        for measure in ['temp', 'dia', 'sys', 'hr']:

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            df_measure = df.loc[(df.measure==measure)]

            if errorbar_corr:
                df_measure = Helpers.apply_errorbar_correction(df_measure)

            ax = sns.lineplot(
                data = df_measure,
                x = 'time',
                y = 'score',
                hue = 'tp',
                markersize = 10,
                legend = True,
                style = 'tp',
                markers = ["o", "D"],
                palette = {
                    'A0': '#56A0FB',
                    'B0': '#F71480'},
                errorbar = "ci",
                err_style = "bars",
                err_kws={
                    'capsize': 4,
                    'elinewidth': 0.75,
                    'capthick': 0.75},
            )

            plt.legend(title='Psilocybin doses', labels=['10mg', '25mg'])

            ax.set_xlabel('Time [min]', fontdict=config.axislabel_fontdict)
            ax.set_xticks([0, 30, 60, 90, 120, 240, 360, 420])

            if measure=='temp':
                ax.set_yticks([35.7, 35.9, 36.1, 36.3, 36.5, 36.7])
                ax.set_ylabel(
                    'Body temperature [°C]',
                    fontdict=config.axislabel_fontdict)
            elif measure=='hr':
                ax.set_yticks([60, 65, 70, 75, 80, 85])
                ax.set_ylabel(
                    'Heart rate [BPM]',
                    fontdict=config.axislabel_fontdict)
            elif measure=='dia':
                ax.set_yticks([65, 70, 75, 80, 85])
                ax.set_ylabel(
                    'Diastolic BP [mmHg]',
                    fontdict=config.axislabel_fontdict)
            elif measure=='sys':
                ax.set_yticks([110, 120, 130, 140, 150, 160])
                ax.set_ylabel(
                    'Systolic BP [mmHg]',
                    fontdict=config.axislabel_fontdict)
            else:
                assert False

            ax.tick_params(axis='both', which='major', labelsize=config.ticklabel_fontsize)
            sns.despine(top=True, right=True, left=False, bottom=False, offset=10, trim=True)
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

            Helpers.save_fig(
                fig = fig,
                out_dir = out_dir,
                filename = f'vitals_{measure}')

    @staticmethod
    def make_5dasc(df, out_dir=folders.exports, horizontal=False):

        assert isinstance(horizontal, bool)

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
            'imp': 'Impaired\ncontrol and cog.',
            'anx': 'Anxiety',
            'cimg': ' Complex\nimagery',
            'eimg': 'Elementary\nimagery',
            'av': 'Audio-Visual\nsynesthesia',
            'per': 'Changed meaning\nof percepts',}, regex=True,)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if horizontal:
            fig.set_size_inches([4*4.8, 2*4.8])
            sns.barplot(
                data=df,
                x='measure',
                y='score',
                hue='tp',
                palette = {
                    'A0': '#56A0FB',
                    'B0': '#F71480'},)

            ax.set_yticks([0, 25, 50, 75, 100])
            sns.despine(top=True, right=True, left=False, bottom=True)
            ax.tick_params(axis='y', which='major', labelsize=config.ticklabel_fontsize)
            ax.tick_params(axis='x', which='major', labelsize=13)
            ax.set_ylabel('Score', fontdict=config.axislabel_fontdict)
            ax.tick_params(axis='x', length=0)
            ax.set_xlabel('')

        else:
            fig.set_size_inches([2*4.8, 4*4.8])
            sns.barplot(
                data=df,
                y='measure',
                x='score',
                hue='tp',
                errorbar="se",
                capsize=0.15,
                errwidth=0.95,
                width=0.6,
                palette = {
                    'A0': '#56A0FB',
                    'B0': '#F71480'},)

            ax.set_xticks([0, 25, 50, 75, 100])
            ax.tick_params(axis='both', which='major', labelsize=config.ticklabel_fontsize)
            ax.set_xlabel('Score', fontdict=config.axislabel_fontdict)
            ax.set_yticklabels(ax.get_yticklabels(), ha='center', va='center')
            ax.set_ylabel('')

        # Collect handles and labels for the legend
        legend_handles = []
        legend_labels = []
        handles, labels = ax.get_legend_handles_labels()
        legend_handles.extend(handles)
        legend_labels.extend(labels)
        custom_legend_labels = ["10 mg", "25 mg"]
        plt.legend(legend_handles, custom_legend_labels)

        sns.despine(top=True, right=True, left=False, bottom=False, offset=10, trim=True)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)

        Helpers.save_fig(
            fig = fig,
            out_dir = out_dir,
            filename = f'5dasc')

    @staticmethod
    def make_corrmat(coeffs_df, pvalues_df, method, tp, pred_set, out_dir=folders.corrmats, out_fname='corrmat'):

        coeffs_df = coeffs_df[coeffs_df.index.isin(pred_set)]
        pvalues_df = pvalues_df[pvalues_df.index.isin(pred_set)]

        # Plot results
        fig, ax = plt.subplots(dpi=300)
        ax.set_title(f'{method.upper()} @{tp}', fontsize=14, fontweight='bold')

        sns.heatmap(
            data = coeffs_df.astype(float),
            ax = ax,
            annot = pvalues_df.applymap(Helpers.sig_marking),
            vmin = -1,
            vmax = 1,
            linewidths = .05,
            cmap = 'vlag',
            fmt = '')

        plt.xticks(rotation=45)
        ax.set_xlabel(f'{tp}-baseline scores')

        Helpers.save_fig(
            fig = fig,
            out_dir = out_dir,
            filename = f'{out_fname}_{tp}_{method}')


class Helpers:

    @staticmethod
    def save_fig(fig, out_dir, filename):

        if config.savePNG:
            fig.savefig(
                fname=os.path.join(out_dir, f'{filename}.png'),
                bbox_inches="tight",
                format='png',
                dpi=300,)

        if config.saveSVG:
            fig.savefig(
                fname=os.path.join(out_dir, f'{filename}.svg'),
                bbox_inches="tight",
                format='svg',
                dpi=300,)

        plt.close()

    @staticmethod
    def apply_errorbar_correction(df):
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

        if 'Z_' in measure:
            ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])

        if measure=='UPDRS_1':
            ax.set_yticks([5, 10, 15, 20, 25])
            ax.set_ylabel('Non-motor EDL (UPDRS1)', fontdict=config.axislabel_fontdict)
        elif measure=='UPDRS_2':
            ax.set_yticks([5, 10, 15, 20])
            ax.set_ylabel('Motor EDL (UPDRS2)', fontdict=config.axislabel_fontdict)
        elif measure=='UPDRS_3':
            ax.set_yticks([30, 35, 40])
            ax.set_ylabel('Motor exam (UPDRS3)', fontdict=config.axislabel_fontdict)
        elif measure=='MADRS':
            ax.set_yticks([5, 10, 15, 20, 25, 30])
            ax.set_ylabel('Depression (MADRS)', fontdict=config.axislabel_fontdict)
            #ax.add_patch(patches.Rectangle(
            #	(-5, 19), 50, 30,
            #	edgecolor=None, facecolor='blue', alpha=0.15))
        elif measure=='HAMA':
            ax.set_yticks([5, 10, 15, 20])
            ax.set_ylabel('Anxiety (HAMA)', fontdict=config.axislabel_fontdict)
        elif measure=='ESAPS':
            ax.set_yticks([-1, 0, 1, 2, 3, 4])
        elif measure=='Z_PAL':
            ax.set_ylabel('Associate learning (PAL z-score)') #, fontweight='bold'
        elif measure=='Z_SWM':
            ax.set_ylabel('Working memory (SWM z-score)')
        elif measure=='PRL':
            ax.set_yticks([2, 4, 6, 8])
            ax.set_ylabel('Reversal learning (# of reversals)')
        else:
            ax.set_ylabel('Score', fontdict=config.axislabel_fontdict)

        if boost_y:
            y_high = ax.get_ylim()[1]
            y_low  = ax.get_ylim()[0]
            y_boost = 0.3*(y_high-y_low)
            ax.set_ylim([y_low, y_high+y_boost])

        return ax

    @staticmethod
    def sig_marking(value):
        if 0.05 > value >= 0.01:
            return '*'
        elif 0.01 > value >= 0.001:
            return '**'
        elif 0.001 > value:
            return '***'
        else:
            return ''
