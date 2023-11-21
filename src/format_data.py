import src.folders as folders
import src.config as config
import pandas as pd
import datetime
import os


class Controllers():

    @staticmethod
    def get_master_df(save=True):
        """ Creates and saves master DF from raw input data
        """
        assert isinstance(save, bool)

        df_clinical = Core.format_clinical_data(save=save)
        df_cantab = Core.format_CANTAB_data(save=save)
        df_prl = Core.format_PRL_data(save=save)
        df_demo = Core.format_demographic_data(save=save)

        df_master = pd.concat([df_cantab, df_clinical, df_prl], ignore_index=True)
        df_master['pID'] = df_master['pID'].astype(int)

        ### Add demographic + baseline data as new columns: 'gender', 'edu', 'age' 'bsl_UPDRS'
        for bsl_var in ['gender', 'edu', 'age']:
            df_master[bsl_var]=None
            col_idx = df_demo.columns.get_loc(bsl_var)

            for pID in df_master.pID.unique():
                row_idx = df_demo[df_demo['pID']==pID].index[0]
                bsl_val = df_demo.iloc[row_idx, col_idx]
                df_master.loc[(df_master.pID==pID), bsl_var] = bsl_val
                #df_master.loc[df_master['pID']=='pID', bsl_var] = bsl_val

        Helpers.save_data(save, df_master, path=os.path.join(folders.data, 'pdp1_master_v1.csv'))
        return df_master


class Core():

    @staticmethod
    def format_clinical_data(save):

        df = pd.read_csv(os.path.join(
            folders.data,
            'Clinical',
            'PDP1-PDP1clinicalOutcomes_DATA_2023-Jul-17.csv'),)

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

        df = df.loc[
            (df.cssrs_rater==4) |
            (df.esaps_rater==4) |
            (df.madrs_rater==4) |
            (df.hama_rater==4)]

        measures = ['cssrs', 'esaps', 'madrs', 'hama']
        keep_columns = ['tp', 'pID'] + [f'{measure}_total' for measure in measures]
        df = df[keep_columns]

        df = pd.melt(
            df,
            id_vars= ['pID', 'tp'],
            value_vars=[f'{measure}_total' for measure in measures],
            var_name='measure',
            value_name='score',
            ignore_index=True)

        df = df.rename(columns={
            'cssrs_total': 'cssrs',
            'esaps_total': 'esaps',
            'madrs_total': 'madrs',
            'hama_total': 'hama'})

        df = Helpers.clean_source_data(df)
        Helpers.save_data(save, df, path=os.path.join(folders.data, 'pdp1_clinical_v1.csv'))
        return df

    @staticmethod
    def format_CANTAB_data(save):

        df = pd.read_csv(os.path.join(
            folders.data,
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

        key_measures=[
          'PALFAMS','PALTEA', # Memory
          'RTIFMDMT','RTIFMDRT','RTISMDMT', 'RTISMDRT', # Attention & Psychomotor Speed; NO NORM
          'MTSCFAPC','MTSCTAPC','MTSPS82','MTSRCAMD','MTSRFAMD', # Attention & Psychomotor Speed; NO NORM
          'OTSMDLFC', 'OTSPSFC', # Executive Function
          'SWMBE12','SWMBE4','SWMBE468','SWMBE6','SWMBE8','SWMS' # Executive Function
        ]
        df = df.loc[(df.measure.isin(key_measures))]

        df = df.replace({'Screen': 'bsl', 'A/B30': 'B30'})

        df = Helpers.clean_source_data(df)
        Helpers.save_data(save, df, path=os.path.join(folders.data, 'pdp1_cantab_v1.csv'))
        return df

    @staticmethod
    def format_PRL_data(save):

        df_index = pd.read_csv(os.path.join(
            folders.data,
            'PRL',
            'PDP1_reversalLearningIndex.csv'))

        raws_folder = os.path.join(folders.data, 'PRL', 'raws')
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
        df['measure'] = 'plr'
        df = df.loc[(df.abort==0)]
        df = df[['pID', 'tp','measure', 'score']]
        df['tp'] = df['tp'].replace('Screening', 'bsl', regex=True)

        # Clean up and save
        df = Helpers.clean_source_data(df)
        Helpers.save_data(save, df, path=os.path.join(folders.data, 'pdp1_prl_v1.csv'))
        return df

    @staticmethod
    def format_demographic_data(save):

        df = pd.read_csv(os.path.join(
            folders.data,
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
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)

        Helpers.save_data(save, df, path=os.path.join(folders.data, 'pdp1_demography_v1.csv'))
        return df


class Helpers():
    """
    """

    @staticmethod
    def get_age(row):
        return round((datetime.datetime.now() - row['Date of Birth']).days / 365.25, 1)

    @staticmethod
    def clean_source_data(df):

        df = df[['pID', 'tp', 'measure', 'score']]
        df = df.loc[(df.pID.isin(config.valid_pIDs))]
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def save_data(save, df, path):

        if save is False:
             return
        else:
            df.to_csv(path, index=False)
