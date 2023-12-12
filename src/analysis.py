import src.config as config
import src.folders as folders
from scipy.stats import ttest_rel
import pandas as pd
import numpy as np
import math
import os


class Controllers():

    @staticmethod
    def delta_max_5DASC(df):

        for measure in df.measure.unique():
            a0s = df.loc[(df.measure==measure) & (df.tp=='A0')].sort_values(by='pID')
            b0s = df.loc[(df.measure==measure) & (df.tp=='B0')].sort_values(by='pID')

            if not (np.array_equal(a0s.pID, b0s.pID)):
                print(f'Unequal pID arrays: {measure}')
                continue

            t, p = ttest_rel(a0s.score, b0s.score)
            sig=''
            if p<0.05:
                sig='*'

            print(f'{measure}: t={round(t,2)} p={round(p,3)} {sig}')

    @staticmethod
    def delta_max_vitals(df):

        for measure in df.measure.unique():
            a0_deltamaxs=[]
            b0_deltamaxs=[]

            # Extract delta max for each participant
            for pID in df.pID.unique():
                a0_delta_max = Helpers.find_delta_max(df[(df['measure']==measure) & (df['pID']==pID) & (df['tp']=='A0')])
                b0_delta_max = Helpers.find_delta_max(df[(df['measure']==measure) & (df['pID']==pID) & (df['tp']=='B0')])

                if math.isnan(a0_delta_max) or math.isnan(b0_delta_max):
                    print(f'Failed to find max delta; measure:{measure}, pID:{pID}')
                    continue

                a0_deltamaxs.append(a0_delta_max)
                b0_deltamaxs.append(b0_delta_max)

            # Do paired t-test of delta maxes
            t, p = ttest_rel(a0_deltamaxs, b0_deltamaxs)

            sig=''
            if p<0.05:
                sig='*'

            print(f'{measure}: t={round(t,2)} p={round(p,3)} {sig}')


class Helpers:

    @staticmethod
    def find_delta_max(df):

        df = df.reset_index()
        assert len(df[(df['time']==0)].index)==1
        t0_row_idx = df[(df['time']==0)].index[0]
        score_col_idx = df.columns.get_loc('score')
        t0_value = df.iloc[t0_row_idx, score_col_idx]

        idx_max=t0_row_idx
        delta_max=math.nan

        for idx in df.index:
            delta = abs(df.iloc[idx, score_col_idx]-t0_value)

            if (delta>delta_max) or math.isnan(delta_max):
                idx_max = idx
                delta_max = delta

        return round((df.iloc[idx_max, score_col_idx]-t0_value),3)
