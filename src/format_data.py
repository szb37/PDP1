import src.folders as folders
import src.config as config
import pandas as pd
import numpy as np
import datetime
import copy
import os


class Controllers():

    @staticmethod
    def get_master_df(save=True, process_raws=True):
        """ Creates and saves master DF from raw input data
        """

        assert isinstance(save, bool)
        assert isinstance(process_raws, bool)

        if process_raws:
            Core.format_demographic_data(save=save)
            Core.format_clinical_data(save=save)
            Core.format_CANTAB_data(save=save)
            Core.format_UPDRS_data(save=save)
            Core.format_PRL_data(save=save)

        df_clinical = pd.read_csv(os.path.join(folders.processed_data, 'pdp1_clinical_v1.csv'))
        df_cantab = pd.read_csv(os.path.join(folders.processed_data, 'pdp1_cantab_v1.csv'))
        df_updrs = pd.read_csv(os.path.join(folders.processed_data, 'pdp1_updrs_v1.csv'))
        df_demo = pd.read_csv(os.path.join(folders.processed_data, 'pdp1_demography_v1.csv'))
        df_prl = pd.read_csv(os.path.join(folders.processed_data, 'pdp1_prl_v1.csv'))

        df_master = pd.concat([df_cantab, df_clinical, df_prl, df_updrs], ignore_index=True)
        df_master['pID'] = df_master['pID'].astype(int)
        df_master = df_master[df_master['measure']!='MTSCTAPC'] # Failed measure, all values=100
        df_master = df_master.reset_index(drop=True)
        df_master = Helpers.add_covariates(df_master, df_demo)

        ### Do not remove extreme values in consensus w Ellen
        '''
        row_idx_todelete = [idx for idx in df_master[
            (df_master['pID']==1051) & (df_master['measure']=='RTISMDMT')].index]
        df_master = df_master.drop(row_idx_todelete)

        row_idx_todelete = [idx for idx in df_master[
            (df_master['pID']==1051) & (df_master['measure']=='RTISMDRT')].index]
        df_master = df_master.drop(row_idx_todelete)
        '''

        ### Save results
        Helpers.save_data(save, df_master, path=os.path.join(folders.data, 'pdp1_MASTER_v1.csv'))
        return df_master


class Core():

    @staticmethod
    def format_clinical_data(save):

        df = pd.read_csv(
            os.path.join(
                folders.raw_data,
                'Clinical',
                'PDP1-PDP1clinicalOutcomes_DATA_2023-Jul-17.csv'),
            dtype={'record_id': str})

        df = df.loc[(df.record_id.isin(config.valid_str_pIDs))]
        df['record_id'] = df['record_id'].astype(int)

        df = df.rename(columns={
            'record_id': 'pID',
            'redcap_event_name': 'tp'})

        df = df.replace({
            "screening_baseline_arm_1": "bsl",
            "day_a0_dose_1_arm_1": "A0",
            "day_a1_arm_1": "A1",
            "day_a7_arm_1": "A7",
            "day_b0_dose_2_arm_1": "B0",
            "day_b1_arm_1": "B1",
            "day_b7_arm_1": "B7",
            "day_b11_arm_1": "B11",
            "day_ab25_arm_1": "B25",
            "day_ab30_arm_1": "B30",
            "day_ab90_phone_arm_1": "B90"}, regex=True,)

        df = df.loc[(df.tp.isin(['bsl', 'A7', 'B7', 'B30']))]

        df = df.loc[
            (df.cssrs_rater==4) |
            (df.esaps_rater==4) |
            (df.madrs_rater==4) |
            (df.hama_rater==4)  |
            (df.ccfq_complete==2) |
            (df.ccfq_bl_complete==2)]

        measures = ['cssrs', 'esaps', 'madrs', 'hama', 'ccfq', 'ccfq_bl']
        keep_columns = ['tp', 'pID'] + [f'{measure}_total' for measure in measures]
        df = df[keep_columns]

        df = pd.melt(
            df,
            id_vars= ['pID', 'tp'],
            value_vars=[f'{measure}_total' for measure in measures],
            var_name='measure',
            value_name='score',
            ignore_index=True)

        df = df.replace({
            'cssrs_total': 'CSSRS',
            'esaps_total': 'ESAPS',
            'madrs_total': 'MADRS',
            'hama_total': 'HAMA',
            'ccfq_total': 'CCFQ',
            'ccfq_bl_total': 'CCFQ'})

        df['test'] = df['measure']
        df = Helpers.standardize_df(df)
        Helpers.save_data(save, df, path=os.path.join(folders.processed_data, 'pdp1_clinical_v1.csv'))
        return df

    @staticmethod
    def format_CANTAB_data(save):

        df = pd.read_csv(os.path.join(
            folders.raw_data,
            'CANTAB',
            'RowByMeasureNorms_PDP1_duplicate_clean.csv'))

        df = df.rename(columns={
            'Participant ID': 'pID',
            'Visit ID': 'tp',
            'Result': 'score',
            'Measure Code': 'measure'})

        df = df[['pID', 'tp', 'measure', 'score']]

        df['pID'] = df['pID'].str[6:]
        df['pID'] = df['pID'].astype(int)
        df = df.loc[(df.pID.isin(config.valid_pIDs))]

        df = df.replace({'Screen': 'bsl', 'A/B30': 'B30'})

        key_measures=[
          'PALFAMS','PALTEA', # Memory
          'RTIFMDMT','RTIFMDRT','RTISMDMT', 'RTISMDRT', # Attention & Psychomotor Speed; NO NORM
          'MTSCFAPC','MTSCTAPC','MTSPS82','MTSRCAMD','MTSRFAMD', # Attention & Psychomotor Speed; NO NORM
          'OTSMDLFC', 'OTSPSFC', # Executive Function
          'SWMBE12','SWMBE4','SWMBE468','SWMBE6','SWMBE8','SWMS' # Executive Function
        ]

        df = df.loc[(df.measure.isin(key_measures))]
        df['test'] = df.apply(Helpers.get_test, axis=1)

        df = Helpers.standardize_df(df)
        Helpers.save_data(save, df, path=os.path.join(folders.processed_data, 'pdp1_cantab_v1.csv'))
        return df

    @staticmethod
    def format_PRL_data(save):

        df_index = pd.read_csv(os.path.join(
            folders.raw_data,
            'PRL',
            'PDP1_reversalLearningIndex.csv'))

        raws_folder = os.path.join(folders.raw_data, 'PRL', 'raws')
        for idx, csv_filename in enumerate([f for f in os.listdir(raws_folder) if f.endswith('.csv')]):
            if idx==0:
                df = pd.read_csv(os.path.join(raws_folder, csv_filename))
            else:
                df_to_add = pd.read_csv(os.path.join(raws_folder, csv_filename))
                df = pd.concat([df, df_to_add], ignore_index=True)

        df_index['date'] = pd.to_datetime(df_index['date'])
        df['startdate'] = pd.to_datetime(df['startdate'])
        df = df.merge(df_index, left_on=['subjectid', 'startdate'], right_on=['participantID', 'date'], how='inner')

        df['score'] = df[['countReversals_test1', 'countReversals_test2', 'countReversals_test3']].sum(axis=1)
        df = df.rename(columns={'subjectid': 'pID','visit': 'tp',})
        df['measure'] = 'PLR'
        df['test'] = 'PLR'
        df = df.loc[(df.abort==0)]
        df['tp'] = df['tp'].replace('Screening', 'bsl', regex=True)

        df = Helpers.standardize_df(df)
        Helpers.save_data(save, df, path=os.path.join(folders.processed_data, 'pdp1_prl_v1.csv'))
        return df

    @staticmethod
    def format_demographic_data(save):

        df = pd.read_csv(os.path.join(
            folders.raw_data,
            'CANTAB',
            'RowByMeasureNorms_PDP1_duplicate_clean.csv'))

        df = df.rename(columns={
            'Participant ID': 'pID',
            'Gender': 'gender',
            'Level of Education': 'edu',})

        df['pID'] = df['pID'].str[6:]
        df['pID'] = df['pID'].astype(int)

        df['Date of Birth'] = pd.to_datetime(df['Date of Birth'])
        df['age'] = df.apply(Helpers.get_age, axis=1)

        df = df.loc[(df.pID.isin(config.valid_pIDs))]
        df = df[['pID', 'gender', 'edu', 'age']]
        df.drop_duplicates(inplace=True)

        ### Process LED intake
        df_led = pd.read_csv(os.path.join(
            folders.raw_data,
            'pdp1_LED_intake.csv',))

        df_led = df_led.rename(columns={
            'ParticipantID': 'pID',
            'levodopaEquivalents': 'LED'})

        df_led = df_led[['pID', 'LED']]
        df_led = df_led.dropna(how='all')
        df_led['pID'] = df_led['pID'].astype(int)

        df = pd.merge(df, df_led, on='pID', how='inner')

        Helpers.save_data(save, df, path=os.path.join(folders.processed_data, 'pdp1_demography_v1.csv'))
        return df

    @staticmethod
    def format_UPDRS_data(save):

        df = pd.read_csv(
            os.path.join(
                folders.raw_data,
                'Clinical',
                'PDP1-PDP1clinicalOutcomes_DATA_2023-Jul-17.csv'),
            dtype={'record_id': str})

        df = df.loc[(df.record_id.isin(config.valid_str_pIDs))]
        df['record_id'] = df['record_id'].astype(int)

        df = df.rename(columns={
          'record_id': 'pID',
          'redcap_event_name': 'tp'})

        df = df.replace({
          "screening_baseline_arm_1": "bsl",
          "day_a0_dose_1_arm_1": "A0",
          "day_a1_arm_1": "A1",
          "day_a7_arm_1": "A7",
          "day_b0_dose_2_arm_1": "B0",
          "day_b1_arm_1": "B1",
          "day_b7_arm_1": "B7",
          "day_b11_arm_1": "B11",
          "day_ab25_arm_1": "B25",
          "day_ab30_arm_1": "B30",
          "day_ab90_phone_arm_1": "B90"}, regex=True,)

        del_idx = df.loc[
            (df.redcap_repeat_instrument=='mdsupdrs') &
            (df.redcap_repeat_instance==1) &
            (df.pID==1002)].index.values
        assert len(del_idx)==1
        df = df.drop(del_idx[0])

        df1 = copy.deepcopy(df)
        df2 = copy.deepcopy(df)
        df3 = copy.deepcopy(df)
        df4 = copy.deepcopy(df)

        ### Calc sum scores for UPDRS component 1
        updrs1_cols = df1.columns[df1.columns.str.startswith('updrs1') ].drop('updrs1_rater')
        assert len(updrs1_cols)==13
        df1[updrs1_cols] = df1[updrs1_cols].replace(9.0,0)
        df1['UPDRS_1'] = df1[updrs1_cols].sum(axis=1,skipna=False)
        df1=df1[['pID', 'tp', 'UPDRS_1']].dropna()

        ### Calc sum scores for UPDRS component 2
        updrs2_cols = df2.columns[df2.columns.str.startswith('updrs2')]
        assert len(updrs2_cols)==13
        df2[updrs2_cols] = df2[updrs2_cols].replace(9.0,0)
        df2['UPDRS_2'] = df2[updrs2_cols].sum(axis=1,skipna=False)
        df2=df2[['pID', 'tp', 'UPDRS_2']].dropna()

        ### Calc sum scores for UPDRS component 3
        colsToDrop = [
          'updrs3_meds',
          'updrs3_onoff',
          'updrs3_ldopa',
          'updrs3_ld_time',
          'updrs3_dysk_pres',
          'updrs3_dysk_impact',
          'updrs3_hy']

        updrs3_value_cols = df.columns[df.columns.str.startswith('updrs3') & ~ df.columns.str.contains('v2') ].drop(colsToDrop)
        updrs3_value_cols = [col_name for col_name in updrs3_value_cols]

        df3[updrs3_value_cols] = df3[updrs3_value_cols].replace(9.0, np.nan)
        df3['updrs_pt3_sum'] = df3[updrs3_value_cols].sum(axis=1)
        df3 = df3.dropna(subset=updrs3_value_cols, how='all')

        df3 = df3[['pID', 'tp', 'updrs_pt3_sum']]
        df3 = df3.loc[(df3.tp.isin(['A7', 'B7', 'B30', 'bsl']))]
        df3 = df3.rename(columns={'updrs_pt3_sum': 'UPDRS_3'})

        # Two UPDRS_3 scores for 1083 at bsl (=baseline):
        # this participant had a delay in scheduling their A0 due to travel so their UPDRS Part 3 (*not* the entire UPDRS) expired. They had to repeat the Part 3 only on 2022-Sep-28. They should not have a Part 1, Part 2, or Part 4 for that date. They still had valid scores for those parts from the first baseline 2022-Aug-23. So we'll just drop their first Part 3 baseline score per protocol.
        row_idx = df3.loc[(df3.pID==1083) & (df3.UPDRS_3==39) & (df3.tp=='bsl')].index
        assert len(row_idx)==1
        df3 = df3.drop(row_idx[0])

        ### Calc sum scores for UPDRS component 4
        updrs4_cols = [
          'updrs4_dys_t_s',
          'updrs4_dys_impact_s',
          'updrs4_off_t_s',
          'updrs4_off_impact_s',
          'mdsupdrs4_off_complexity_s',
          'updrs4_dystonia_t_s']
        df4[updrs4_cols] = df4[updrs4_cols].replace(9.0,0)
        df4['UPDRS_4'] = df4[updrs4_cols].sum(axis=1,skipna=False)
        df4=df4[['pID', 'tp', 'UPDRS_4']].dropna()

        ### Calculate UPDRS sum_scores
        df_sum = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)
        df_sum = pd.melt(
            df_sum,
            id_vars= ['pID', 'tp'],
            var_name='measure',
            value_name='score',
            ignore_index=True)
        df_sum = df_sum.dropna(subset=['score'])

        df_sum = pd.pivot_table(df_sum, index=['tp','pID'], columns='measure', values= 'score')
        df_sum.columns.name = None
        df_sum = df_sum.reset_index()
        df_sum = df_sum.dropna()
        df_sum['UPDRS_SUM'] = df_sum[['UPDRS_1','UPDRS_2','UPDRS_3','UPDRS_4']].sum(axis=1)
        df_sum = df_sum.drop(['UPDRS_1','UPDRS_2','UPDRS_3','UPDRS_4'], axis=1)

        df = pd.concat([df1, df2, df3, df4, df_sum], axis=0, ignore_index=True)
        df = pd.melt(
            df,
            id_vars= ['pID', 'tp'],
            var_name='measure',
            value_name='score',
            ignore_index=True)
        df = df.dropna(subset=['score'])
        df['test'] = 'UPDRS'

        df = Helpers.standardize_df(df)
        Helpers.save_data(save, df, path=os.path.join(folders.processed_data, 'pdp1_updrs_v1.csv'))
        return df


class Helpers():

    @staticmethod
    def get_test(row):

        if row['measure'] in ['PALFAMS','PALTEA']:
            return 'PAL'
        elif row['measure'] in ['RTIFMDMT','RTIFMDRT','RTISMDMT', 'RTISMDRT']:
            return 'RTI'
        elif row['measure'] in ['MTSCFAPC','MTSCTAPC','MTSPS82','MTSRCAMD','MTSRFAMD']:
            return 'MTS'
        elif row['measure'] in [ 'OTSMDLFC', 'OTSPSFC']:
            return 'OTS'
        elif row['measure'] in [ 'SWMBE12','SWMBE4','SWMBE468','SWMBE6','SWMBE8','SWMS']:
            return 'SWM'
        else:
            assert False

    @staticmethod
    def get_age(row):
        return round((datetime.datetime.now() - row['Date of Birth']).days / 365.25, 1)

    @staticmethod
    def standardize_df(df):

        df = df.loc[(df.pID.isin(config.valid_pIDs))]
        df = df[['pID', 'tp', 'test', 'measure', 'score']]
        df = df.dropna(subset=['score'])
        df = df.sort_values(by=['pID', 'tp', 'measure'])
        df = df.drop_duplicates()
        return df

    @staticmethod
    def save_data(save, df, path):

        if save is False:
             return
        else:
            df.to_csv(path, index=False)

    @staticmethod
    def add_covariates(df_master, df_demo):

        ### Add variables from df_demo
        for bsl_var in ['gender', 'edu', 'age', 'LED']:
            df_master[bsl_var]=None
            col_idx = df_demo.columns.get_loc(bsl_var)

            for pID in df_master.pID.unique():
                row_idx = df_demo[df_demo['pID']==pID].index[0]
                bsl_val = df_demo.iloc[row_idx, col_idx]
                df_master.loc[(df_master.pID==pID), bsl_var] = bsl_val

        ### Add UPDRS_3 as bsl_severity measure
        col_idx = df_master.columns.get_loc('score')
        for pID in df_master.pID.unique():

            row_idx = df_master.loc[
                (df_master.pID==pID) &
                (df_master.measure=='UPDRS_3') &
                (df_master.tp=='bsl')].index

            assert len(row_idx)==1

            bsl_val = df_master.iloc[ row_idx[0], col_idx]
            df_master.loc[(df_master.pID==pID), 'severity'] = bsl_val

        return df_master
