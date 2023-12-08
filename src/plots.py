import matplotlib.pyplot as plt
import src.config as config
import src.folders as folders
import seaborn as sns
import pandas as pd
import os

class Controllers():

    @staticmethod
    def make_histograms(df, ignore_measure=[], output_dir=folders.histograms_dir):

        for measure in df.measure.unique():
            print(f'Creating histograms for {measure}.')

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
                fig=fig,
                measure = measure,
                fig_type = 'histogram',
                output_dir=output_dir,
                dpi=300)

    @staticmethod
    def make_timeevolutions(df, within_sub_errorbar=True, boost_y=True, output_dir=folders.timeevols_dir):

        for measure in df.measure.unique():
            print(f'Creating time evolution plot for {measure}; settings: within_sub_errorbar={within_sub_errorbar}, boost_y={boost_y}.')

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            tmp_df = df.loc[(df.measure==measure)]
            tps = tmp_df.tp.unique()

            tmp_df['tp'] = pd.Categorical(
                tmp_df['tp'],
                categories=['bsl', 'A7', 'B7', 'B30'],
                ordered=True)

            if within_sub_errorbar:
                tmp_df = Helpers.get_within_sub_errorbar(tmp_df)

            sns.lineplot(
                x='tp', y='score', data=tmp_df,
                marker='o', markersize=12,
                err_style="bars", errorbar="ci", err_kws={'capsize':4, 'elinewidth': 1.5, 'capthick': 1.5})

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
                fig=fig,
                measure = measure,
                fig_type = 'timeevol',
                output_dir = output_dir,
                dpi=300)

    @staticmethod
    def make_vitals(df, y, y_label, within_sub_errorbar=True, output_dir=folders.vitals_dir):

        pass



class Helpers:

    @staticmethod
    def save_fig(fig, measure, fig_type, output_dir, dpi=300):

        fig.savefig(
            fname=os.path.join(output_dir, f'pdp1_{fig_type}_{measure}.png'),
            format='png',
            dpi=dpi,
        )
        fig.savefig(
            fname=os.path.join(output_dir, f'pdp1_{fig_type}_{measure}.svg'),
            format='svg',
            dpi=dpi,
        )
        del fig
        plt.close()

    @staticmethod
    def get_within_sub_errorbar(df):
        """ Create adjustment factor: (grand_mean - each_subject_mean)
            Adjust error bars for within subject design, see:
                - https://stats.stackexchange.com/questions/574379/correcting-repeated-measures-data-to-display-error-bars-that-show-within-subject
                - https://www.cogsci.nl/blog/tutorials/156-an-easy-way-to-create-graphs-with-within-subject-error-bars
        """

        grand_mean = df['score'].mean()

        for pID in df.pID.unique():
            subject_mean = df[df['pID'] == pID]['score'].mean()
            df.loc[(df.pID==pID), 'ws_adj_factor'] = grand_mean - subject_mean

        df['adj_score'] = df['score'] + df['ws_adj_factor']
        df = df.drop(columns=['score', 'ws_adj_factor'])
        df = df.rename(columns={'adj_score': 'score'})

        return df
