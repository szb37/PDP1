from scipy.stats import ttest_rel,  wilcoxon
from statistics import mean, stdev
from scipy.stats import zscore
from itertools import product
import src.folders as folders
import src.config as config
import src.plots as plots
from scipy import stats
import pandas as pd
import numpy as np
import datetime
import math
import copy
import os


class Controllers():

    @staticmethod
    def get_master_df(out_dir=folders.exports, out_fname='pdp1_data_master.csv', save=False):
        """ Creates and saves master DF from raw input data
        """

        Core.get_demographic_df()
        Core.get_clinical_df()
        Core.get_CANTAB_df()
        Core.get_UPDRS_df()
        Core.get_PRL_df()
        Core.get_NPIQ_df()
        Core.get_PsychQ_df()
        Core.get_cytokine_df()
        Core.get_5dasc_df()
        Core.get_demographic_df()
        Core.get_tsq_df()

        df_clinical = pd.read_csv(os.path.join(folders.data, 'pdp1_clinical.csv'))
        df_cantab = pd.read_csv(os.path.join(folders.data, 'pdp1_cantab.csv'))
        df_updrs = pd.read_csv(os.path.join(folders.data, 'pdp1_updrs.csv'))
        df_prl = pd.read_csv(os.path.join(folders.data, 'pdp1_prl.csv'))
        df_npiq = pd.read_csv(os.path.join(folders.data, 'pdp1_npiq.csv'))
        df_psychq = pd.read_csv(os.path.join(folders.data, 'pdp1_psychq.csv'))
        df_cytokine = pd.read_csv(os.path.join(folders.data, 'pdp1_cytokine.csv'))
        df_5dasc = pd.read_csv(os.path.join(folders.data, 'pdp1_5dasc.csv'))
        df_tsq = pd.read_csv(os.path.join(folders.data, 'pdp1_tsq.csv'))

        df_master = pd.concat([
            df_cantab,
            df_clinical,
            df_prl,
            df_updrs,
            df_npiq,
            df_psychq,
            df_cytokine,
            df_5dasc,
            df_tsq,], ignore_index=True)

        df_master['pID'] = df_master['pID'].astype(int)
        df_master = df_master.reset_index(drop=True)
        df_master = Helpers.get_deltas(df_master)

        if save:
            df_master.to_csv(os.path.join(out_dir, out_fname), index=False)

        return df_master

    @staticmethod
    def get_vitals_df(out_dir=folders.exports, out_fname='pdp1_data_vitals.csv', save=False):

        df = Helpers.get_REDCap_export()

        cols = [col for col in df if (col.startswith('vs_bl')) | (col.startswith('vs_dose')) | (col.startswith('vs_ortho')) ]
        df = df[ ['pID','tp'] + cols]

        df = df.loc[(df.tp.isin(['A0','B0']))]
        df = df.rename(columns={
            'vs_bl_dia':'vs_dose0_dia1',
            'vs_bl_sys':'vs_dose0_sys1',
            'vs_bl_hr':'vs_dose0_hr1',
            'vs_bl_temp':'vs_dose0_temp1',
            'vs_ortho2_sys_sup':'vs_dose420_sys1',
            'vs_ortho2_dia_sup':'vs_dose420_dia1',
            'vs_ortho2_hr_sup':'vs_dose420_hr1',
            'vs_ortho2_temp':'vs_dose420_temp1'})

        cols = [col for col in df if ('sys1' in col) or \
                ('hr1' in col) or \
                ('temp1' in col) or \
                ('dia1' in col)]

        df = df[ ['pID','tp'] + cols]

        # melt all the columns that have vitals data from the dosing days, each of which contains the string 'vs_dose
        df = df.melt(
            id_vars=['pID','tp'],
            value_vars= [col for col in df if ('vs_dose' in col)]
        )

        # split what was the old columns name, eg 'vs_dose60_hr1'
        # but is now a column of its own, after melting, into two new columns
        # which are, in this case, '60' and 'hr'
        df[['measurementTime','vitalSign']] = df['variable'].str.extract(r'^vs_dose(\d+)_(\w+)\d$')
        df['measurementTime'] = df['measurementTime'].astype(int)
        df['vitalSign'].unique()

        df = df.rename(columns={
            'value': 'score',
            'vitalSign': 'measure',
            'measurementTime': 'time'})

        del df['variable']
        df = df.dropna(subset='score')

        # Convert temperatures values greater than 60 from Fahrenheit to Celsius
        i = (df['measure']=='temp') & (df['score']>60)
        df.loc[i, 'score'] = Helpers.fahrenheit_to_celsius(df.loc[i, 'score'])

        if save:
            df.to_csv(os.path.join(out_dir, out_fname), index=False)

        return df

    @staticmethod
    def add_covs_df(df, out_dir=folders.exports, out_fname='pdp1_data_wcovs.csv', save=False):

        df = Core.add_depanx_covs(df)
        df = Core.add_severity_covs(df)
        df = Core.add_5dasc_covs(df)
        df = Core.add_cytokine_covs(df)
        df = Core.add_delta_PRL_covs(df)
        df = Core.add_demographic_covs(
            df,
            df_demo = pd.read_csv(os.path.join(
                folders.exports, 'pdp1_demography.csv')))

        if save:
            df.to_csv(os.path.join(out_dir, out_fname), index=False)

        return df


class Core():

    @staticmethod
    def get_5dasc_df(out_dir=folders.data, out_fname='pdp1_5dasc.csv'):
        """
        ASC (Dittrich, 1998) can be divided into either 5 (Dittrich, 1998) (94 items), or 11 dimensions (Studerus et al., 2010) (42 items), these are 2 ways to analyze the dataset.
        Here we use the 11d representation due to its better psychometric properties.
        """

        df = Helpers.get_REDCap_export()
        asc_items = [
            "fivedasc_util_total",
            "fivedasc_sprit_total",
            "fivedasc_bliss_total",
            "fivedasc_insight_total",
            "fivedasc_dis_total",
            "fivedasc_imp_total",
            "fivedasc_anx_total",
            "fivedasc_cimg_total",
            "fivedasc_eimg_total",
            "fivedasc_av_total",
            "fivedasc_per_total",]
        df = df[['tp', 'pID'] + asc_items]
        df = df.dropna(subset = asc_items, how='all')


        '''
        # These 3 dimensions below are from the 5d representation of the ASC. Not planning to use these, but leaving here just in case
        # To get the 2 missing dimensions of the 5dASC, the raw redcap data must be reanalyzed at item level

        boundless_items =[
                    "fivedasc_util_total",
                    "fivedasc_sprit_total",
                    "fivedasc_bliss_total",
                    "fivedasc_insight_total",]
        anxEgoDis_items = [
            "fivedasc_dis_total",
            "fivedasc_imp_total",
            "fivedasc_anx_total"
        ]
        visual_items = [
            "fivedasc_cimg_total",
            "fivedasc_eimg_total",
            "fivedasc_av_total",
            "fivedasc_per_total"
        ]

        # create 3 summary columns that are the averages of the 3 major factors - these are what we'll do our statistics on
        df['boundless'] = df[boundless_items].mean(axis=1)
        df['anxiousEgo'] = df[anxEgoDis_items].mean(axis=1)
        df['visionary'] = df[visual_items].mean(axis=1)
        '''

        # Normalize these "total" columns by the # of questions that made up the total, so that they become
        # the average of the questions that make them up, rather than the total
        # Experience of Unity 5D-ASC: (18, 34, 41, 42, 52) = 5 questions
        # Spiritual Experience  5D-ASC: (9, 81, 94) = 3 questions
        # Blissful State  5D-ASC: (12, 86, 91) = 3 questions
        # Insightfulness  5D-ASC: (50, 69, 77) = 3 questions
        # Disembodiment 5D-ASC: (26, 62, 63) = 3 questions
        # Impaired Control and Cognition 5D-ASC: (8, 27, 38, 47, 64, 67, 78) = 7 questions
        # Anxiety 5D-ASC: (32, 43, 44, 46, 56, 89) = 6 questions
        # Complex Imagery 5D-ASC: (39, 79, 82) = 3 questions
        # Elementary Imagery 5D-ASC: (14, 22, 33) = 3 questions
        # Audio-Visual Synesthesia 5D-ASC: (20, 23, 75) = 3 questions
        # Changed Meaning of Percepts  5D-ASC: (28, 31, 54) = 3 questions

        df['unity'] = df['fivedasc_util_total'] / 5
        df['spirit'] = df['fivedasc_sprit_total'] / 3
        df['bliss'] = df['fivedasc_bliss_total'] / 3
        df['insight'] = df['fivedasc_insight_total'] / 3
        df['disem'] = df['fivedasc_dis_total'] / 3
        df['impcc'] = df['fivedasc_imp_total'] / 7
        df['anx'] = df['fivedasc_anx_total'] / 6
        df['cimg'] = df['fivedasc_cimg_total'] / 3
        df['eimg'] = df['fivedasc_eimg_total'] / 3
        df['avs'] = df['fivedasc_av_total'] / 3
        df['chmper'] = df['fivedasc_per_total'] / 3
        df['MEAN'] = df[['unity','spirit','bliss','insight','disem','impcc','anx','cimg','eimg','avs','chmper',]].mean(axis=1)

        df = df.drop(columns=asc_items)

        df = df.melt(
            id_vars = ['pID', 'tp'],
            var_name = 'measure',
            value_name = 'score',)

        df = df.dropna(subset='score')
        df['score'] = round(df['score'], 2)
        df['test'] = '11dasc'

        df.to_csv(os.path.join(out_dir, out_fname), index=False)
        return df

    @staticmethod
    def get_cytokine_df(out_dir=folders.data, out_fname='pdp1_cytokine.csv'):

        df_cyto = pd.read_csv(
            os.path.join(folders.raw,'PDP1_cytokineData.csv'))

        del df_cyto['sampleCode']
        df_cyto = df_cyto.replace({"BL": "bsl"}, regex=True,)

        df_cyto = pd.melt(
            df_cyto,
            id_vars= ['pID', 'tp'],
            var_name='measure',
            value_name='score',
            ignore_index=True)
        df_cyto['test']='cytokine'
        df_cyto = df_cyto.dropna(subset='score')

        df_cyto.to_csv(os.path.join(out_dir, out_fname), index=False)
        return df_cyto

    @staticmethod
    def get_NPIQ_df(out_dir=folders.data, out_fname='pdp1_npiq.csv'):

        df = pd.read_csv(
            os.path.join(
                folders.raw,
                'npiq_raana.csv'),
            dtype={'record_id': str})

        df = df.loc[(df.record_id.isin(config.valid_str_pIDs))]
        df['record_id'] = df['record_id'].astype(int)

        df = df.rename(columns={
            'record_id': 'pID',
            'redcap_event_name': 'tp',
            'total_sev': 'NPIQ_SEV',
            'total_dis': 'NPIQ_DIS'})

        df = df.replace({
            "Baseline": "bsl",}, regex=True,)

        df = df[['pID', 'tp', 'NPIQ_SEV', 'NPIQ_DIS']]

        df = pd.melt(
            df,
            id_vars= ['pID', 'tp'],
            value_vars=['NPIQ_SEV', 'NPIQ_DIS'],
            var_name='measure',
            value_name='score',
            ignore_index=True)

        df = df.loc[(df.tp.isin(['bsl', 'A7', 'B7', 'B30', 'B90']))]
        df['test']='NPIQ'

        df = Helpers.standardize_df(df)
        df.to_csv(os.path.join(out_dir, out_fname), index=False)
        return df

    @staticmethod
    def get_PsychQ_df(out_dir=folders.data, out_fname='pdp1_psychq.csv'):

        df = Helpers.get_REDCap_export()

        # Illogical naming convention from Redcap:
        # Items with 'ps' correspond to
        post_bsl_freq = [f'psychq_{n}' for n in range(1,14)]
        bsl_freq = [f'psychq_ps_{n}' for n in range(1,14)]
        freq = bsl_freq+post_bsl_freq

        post_bsl_severityerity = [f'psychq_{n}a' for n in range(1,14)]
        bsl_severity = [f'psychq_ps_a{n}' for n in range(1,14)]
        sev = bsl_severity + post_bsl_severityerity

        # Fill up with zeros missing data at the appropiate timepoints for each column
        df.loc[(df.tp!='bsl'), post_bsl_severityerity] = df.loc[(df.tp!='bsl'), post_bsl_severityerity].replace(np.nan, 0)
        df.loc[(df.tp=='bsl'), bsl_severity] =df.loc[(df.tp=='bsl'), bsl_severity].replace(np.nan, 0)

        keep_cols = ['pID', 'tp'] + freq + sev
        df = df[keep_cols]
        df = df.dropna(subset=freq, how='all')

        # Copy over baseline values to freq and severity column
        for n in range(1,14):

            freq_n = f'psychq_{n}'
            bsl_freq_n = f'psychq_ps_{n}'
            sev_n = f'psychq_{n}a'
            bsl_severity_n = f'psychq_ps_a{n}'

            for index, row in df.iterrows():

                if pd.isnull(row[freq_n]):
                    df.at[index, freq_n] = row[bsl_freq_n]
                    assert df.at[index, 'tp']=='bsl'
                elif pd.notnull(row[freq_n]) and pd.notnull(row[bsl_freq_n]):
                    assert df.at[index, freq_n]==row[bsl_freq_n]

                if pd.isnull(row[sev_n]):
                    df.at[index, sev_n] = row[bsl_severity_n]
                    assert df.at[index, 'tp']=='bsl'
                elif pd.notnull(row[sev_n]) and pd.notnull(row[bsl_severity_n]):
                    assert df.at[index, sev_n]==row[bsl_severity_n]

        # Get rid of columns with bsl values for clarity
        freq = [f'psychq_{n}' for n in range(1,14)]
        severity = [f'psychq_{n}a' for n in range(1,14)]
        keep_cols = ['pID', 'tp'] + freq + severity
        df = df[keep_cols]

        for idx, row in df.iterrows():
            df.loc[idx, 'PsychQ'] = sum([row[i]*row[j]  for i,j in zip(freq, severity)])

        df = df[['pID', 'tp', 'PsychQ']]

        df = pd.melt(
            df,
            id_vars= ['pID', 'tp'],
            value_vars=['PsychQ'],
            var_name='measure',
            value_name='score',
            ignore_index=True)

        df['test']='PsychQ'
        df = Helpers.standardize_df(df)
        df.to_csv(os.path.join(out_dir, out_fname), index=False)
        return df

    @staticmethod
    def get_clinical_df(out_dir=folders.data, out_fname='pdp1_clinical.csv'):

        df = Helpers.get_REDCap_export()

        df = df.loc[(df.tp.isin(['bsl', 'A7', 'B7', 'B30', 'B90']))]

        df = df.loc[
            (df.cssrs_rater==4) |
            (df.esaps_rater==4) |
            (df.madrs_rater==4) |
            (df.hama_rater==4)  |
            (df.ccfq_complete==2) |
            (df.ccfq_bl_complete==2)]

        measures = ['cssrs', 'esaps', 'madrs', 'hama', 'ccfq', 'ccfq_bl']
        keep_cols = [f'{measure}_total' for measure in measures]
        keep_cols = keep_cols + ['tp', 'pID']
        df = df[keep_cols]

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

        # For 1034 there was internal disagreement between raters that was not
        # resolved at the time data was recorded on REDCap. The consensus score
        # of 14 was later agreed on, thus, manually adding consensus score here
        df.loc[(df.pID==1034) &
            (df.tp=='bsl') &
            (df.measure=='HAMA'), 'score'] = 14

        df.to_csv(os.path.join(out_dir, out_fname), index=False)
        return df

    @staticmethod
    def get_CANTAB_df(add_z=True, out_dir=folders.data, out_fname='pdp1_cantab.csv'):

        df = pd.read_csv(os.path.join(
            folders.raw,
            'CANTAB',
            'RowByMeasureNorms_PDP1_duplicate_clean.csv'))

        df = df.rename(columns={
            'Participant ID': 'pID',
            'Visit ID': 'tp',
            'Result': 'score',
            'Measure Code': 'measure'})

        df = df[['pID', 'tp', 'measure', 'score']]
        df = df[df['measure']!='MTSCTAPC'] # Failed measure, all values=100

        df['pID'] = df['pID'].str[6:]
        df['pID'] = df['pID'].astype(int)
        df = df.loc[(df.pID.isin(config.valid_pIDs))]

        df = df.replace({'Screen': 'bsl', 'A/B30': 'B30'})
        key_measures = [ # Extracts all strings within dict values
            string for string_list in config.cantab_measures.values() for string in string_list]

        df = df.loc[(df.measure.isin(key_measures))]
        df['test'] = df.apply(Helpers.get_test, axis=1)

        if add_z:
            df = Core.add_CANTAB_meanZ(df)

        df = Helpers.standardize_df(df)
        df.to_csv(os.path.join(out_dir, out_fname), index=False)
        return df

    @staticmethod
    def get_PRL_df(out_dir=folders.data, out_fname='pdp1_prl.csv'):

        df_index = pd.read_csv(os.path.join(
            folders.raw,
            'PRL',
            'PDP1_reversalLearningIndex.csv'))

        raws_folder = os.path.join(folders.raw, 'PRL', 'raws')
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
        df['measure'] = 'PRL'
        df['test'] = 'PRL'
        df = df.loc[(df.abort==0)]
        df['tp'] = df['tp'].replace('Screening', 'bsl', regex=True)

        df = Helpers.standardize_df(df)
        df.to_csv(os.path.join(out_dir, out_fname), index=False)
        return df

    @staticmethod
    def get_demographic_df(out_dir=folders.exports, out_fname='pdp1_data_demography.csv'):

        df = pd.read_csv(os.path.join(
            folders.raw,
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
            folders.raw,
            'pdp1_LED_intake.csv',))

        df_led = df_led.rename(columns={
            'ParticipantID': 'pID',
            'levodopaEquivalents': 'LED'})

        df_led = df_led[['pID', 'LED']]
        df_led = df_led.dropna(how='all')
        df_led['pID'] = df_led['pID'].astype(int)

        df = pd.merge(df, df_led, on='pID', how='inner')
        df = pd.melt(
            df,
            id_vars= ['pID'],
            var_name='measure',
            value_name='score',
            ignore_index=True)

        df=df.dropna(subset='score')
        df['test'] = 'demography'
        df['tp'] = 'bsl'

        df.to_csv(os.path.join(out_dir, out_fname), index=False)
        return df

    @staticmethod
    def get_UPDRS_df(out_dir=folders.data, out_fname='pdp1_updrs.csv'):

        df = Helpers.get_REDCap_export()

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
        df4[updrs4_cols] = df4[updrs4_cols].replace(9.0, 0)
        df4['UPDRS_4'] = df4[updrs4_cols].sum(axis=1, skipna=False)
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
        df.to_csv(os.path.join(out_dir, out_fname), index=False)
        return df

    @staticmethod
    def get_tsq_df(out_dir=folders.data, out_fname='pdp1_tsq.csv'):


        df = Helpers.get_REDCap_export()

        cols = [col for col in df.columns if (
            ('tsq' in col) and
            ('_ra_check' not in col) and
            ('_date' not in col) and
            ('_total_missing' not in col) and
            ('_6' not in col) and # 6,7,8 are text responses
            ('_7' not in col) and
            ('_8' not in col) and
            ('_complete' not in col))]

        df = df[['pID', 'tp'] + cols]
        df = df.dropna(subset=cols, how='all')

        df = df.melt(
            id_vars = ['pID','tp'],
            value_name = 'score',
            var_name = 'measure')

        df['test'] = 'TSQ'

        df = Helpers.standardize_df(df)
        df.to_csv(os.path.join(out_dir, out_fname), index=False)
        return df

    @staticmethod
    def get_corrmats_df(df, out_dir=folders.corrmats):

        predictors = [
            'severity', 'age', 'LED',
            '11d_util', '11d_sprit', '11d_bliss', '11d_insight', '11d_dis', '11d_imp', '11d_anx', '11d_cimg', '11d_eimg', '11d_av', '11d_per', '11d_MEAN',
            'IFN_gamma_delta_A1', 'IFN_gamma_delta_B1', 'IFN_gamma_delta_B30',
            'IL6_delta_A1', 'IL6_delta_B1', 'IL6_delta_B30',
            'IL8_delta_A1', 'IL8_delta_B1', 'IL8_delta_B30',
            'IL10_delta_A1', 'IL10_delta_B1', 'IL10_delta_B30',
            'TNF_alpha_delta_A1', 'TNF_alpha_delta_B1', 'TNF_alpha_delta_B30',
            'PRL_delta_A7', 'PRL_delta_B7', 'PRL_delta_B30',]

        outcomes = [
            'UPDRS_1', 'UPDRS_2', 'UPDRS_3','UPDRS_4',
            'HAMA', 'MADRS', 'ESAPS', 'CCFQ',
            'Z_MTS', 'Z_OTS', 'Z_PAL', 'Z_RTI', 'Z_SWM',]

        for method, tp in product(['pearson', 'spearman', 'kendall'], ['A7', 'B7', 'B30']):

            coeffs_df = pd.DataFrame(columns=outcomes, index=predictors)
            pvalues_df = pd.DataFrame(columns=outcomes, index=predictors)

            for outcome, predictor in product(outcomes, predictors):

                tmp = df.loc[(df.measure==outcome) & (df.tp==tp)][[predictor, 'delta_score']]
                tmp = tmp.dropna()

                if method == 'pearson':
                    corcoeff, p = stats.pearsonr(tmp['delta_score'], tmp[predictor])
                elif method == 'spearman':
                    corcoeff, p = stats.spearmanr(tmp['delta_score'], tmp[predictor])
                elif method == 'kendall':
                    corcoeff, p = stats.kendalltau(tmp['delta_score'], tmp[predictor])

                coeffs_df.at[predictor, outcome] = corcoeff
                pvalues_df.at[predictor, outcome] = p

            ### Save results
            coeffs_df.to_csv(os.path.join(folders.corrmats, f'coeffs_{method}_{tp}.csv'))
            pvalues_df.to_csv(os.path.join(folders.corrmats, f'pvalues_{method}_{tp}.csv'))

            ### Make figures
            # Use 11d asc
            plots.Controllers.make_corrmat(
                coeffs_df, pvalues_df, method, tp,
                pred_set = [
                    '11d_util',
                    '11d_sprit',
                    '11d_bliss',
                    '11d_insight',
                    '11d_dis',
                    '11d_imp',
                    '11d_anx',
                    '11d_cimg',
                    '11d_eimg',
                    '11d_av',
                    '11d_per',
                    '11d_MEAN'],
                out_fname='corrmat_11dasc')

            # Use everything else
            plots.Controllers.make_corrmat(
                coeffs_df, pvalues_df, method, tp,
                pred_set = [
                    'severity', 'age', 'LED',
                    'IFN_gamma_delta_A1', 'IFN_gamma_delta_B1',
                    'IL6_delta_A1', 'IL6_delta_B1',
                    'IL8_delta_A1', 'IL8_delta_B1',
                    'IL10_delta_A1', 'IL10_delta_B1',
                    'TNF_alpha_delta_A1', 'TNF_alpha_delta_B1',
                    'PRL_delta_A7', 'PRL_delta_B7', 'PRL_delta_B30'],
                out_fname='corrmat')

    @staticmethod
    def add_CANTAB_meanZ(df):
        """ Calculates z-score for each CANTAB outcome and then calculates
            mean z-score across measures for each test.
            This mean z-score across the test's measures is saved as Z_{test name}
        """

        for test in config.cantab_measures.keys():
            tmp = df[df['measure'].isin(config.cantab_measures[test])].copy()
            tmp = pd.pivot_table(tmp, index=['tp','pID'], columns='measure', values= 'score')
            tmp.columns.name = None
            tmp = tmp.reset_index()

            for measure in config.cantab_measures[test]:
                tmp[f'Z_{measure}'] = zscore(tmp[measure], nan_policy='omit')
                del tmp[measure]

                if measure in ['PALFAMS', 'OTSPSFC', 'MTSCFAPC']:
                    # For these scales greater scores are better (lower is better for all others)
                    # Flipping these scales so that improvement=lower scores for all scales
                    tmp[f'Z_{measure}'] = (-1)*tmp[f'Z_{measure}'].copy()

            tmp[f'Z_{test}'] = round(tmp[tmp.filter(like='Z_').columns].mean(axis=1),3)
            tmp = tmp[['pID', 'tp', f'Z_{test}']]
            tmp['measure'] = f'Z_{test}'
            tmp['test'] = f'Z_{test}'
            tmp = tmp.rename(columns={f'Z_{test}': 'score'})

            df = pd.concat([df, tmp])

        return df


    @staticmethod
    def add_demographic_covs(df, df_demo):

        score_idx = df_demo.columns.get_loc('score')
        for measure, pID in product(df_demo.measure.unique(), df.pID.unique()):
            tmp = df_demo[(df_demo['pID']==pID) & (df_demo['measure']==measure)]

            if tmp.shape[0]==0:
                continue

            assert tmp.shape[0]==1
            df.loc[df['pID']==pID, measure] = df_demo.iloc[tmp.index[0], score_idx]

        return df

    @staticmethod
    def add_severity_covs(df):
        """ Add UPDRS_3 baseline as 'severity' measure
        """

        col_idx = df.columns.get_loc('score')
        for pID in df.pID.unique():

            row_idx = df.loc[
                (df.pID==pID) &
                (df.measure=='UPDRS_3') &
                (df.tp=='bsl')].index

            assert len(row_idx)==1

            bsl_val = df.iloc[row_idx[0], col_idx]
            df.loc[(df.pID==pID), 'severity'] = bsl_val

        return df

    @staticmethod
    def add_delta_PRL_covs(df):

        col_idx = df.columns.get_loc('delta_score')

        for pID, tp in product(df.pID.unique(), ['A7', 'B7', 'B30']):

            row_idx = df.loc[
                (df.pID==pID) &
                (df.measure=='PRL') &
                (df.tp==tp)].index

            if len(row_idx)==0:
                continue

            assert len(row_idx)==1
            df.loc[(df.pID==pID), f'PRL_delta_{tp}'] = df.iloc[row_idx[0], col_idx]

        return df

    @staticmethod
    def add_5dasc_covs(df):

        df_5dasc = df.loc[(df.test=='5dasc')].copy()

        for measure, pID in product(df_5dasc.measure.unique(), df.pID.unique()):
            tmp = df_5dasc[(df_5dasc['pID']==pID) & (df_5dasc['measure']==measure)]
            assert tmp.shape[0]==2
            pID_mean = round(tmp.score.mean(), 2)
            df.loc[df['pID']==pID, measure] =  pID_mean

        return df

    @staticmethod
    def add_depanx_covs(df):
        """ 15 <= MADRS @ baseline => depressed
            14 <= HAMA @ baseline => anxious
        """
        score_idx = df.columns.get_loc('score')

        for pID in df.pID.unique():

            ### Add is_anxious
            hama_pid = df[(df['pID']==pID) & (df['measure']=='HAMA') & (df['tp']=='bsl')]
            assert hama_pid.shape[0]==1
            bsl_hama = df.iloc[hama_pid.index[0], score_idx]
            df.loc[df['pID']==pID, 'is_anx'] = bsl_hama>=14

            ### Add is_depressed
            madrs_pid = df[(df['pID']==pID) & (df['measure']=='MADRS') & (df['tp']=='bsl')]
            assert madrs_pid.shape[0]==1
            bsl_madrs = df.iloc[madrs_pid.index[0], score_idx]
            df.loc[df['pID']==pID, 'is_dep'] = bsl_madrs>=15

        return df

    @staticmethod
    def add_cytokine_covs(df):

        df_cytokine = df.loc[(df.test=='cytokine')].copy().reset_index()
        delta_score_idx = df_cytokine.columns.get_loc('delta_score')
        tps = ['A1', 'B1', 'B30']

        for measure, pID, tp in product(df_cytokine.measure.unique(), df.pID.unique(), tps):
            tmp = df_cytokine[
                (df_cytokine['pID']==pID) &
                (df_cytokine['tp']==tp) &
                (df_cytokine['measure']==measure)]

            if tmp.shape[0]==0:
                continue

            assert tmp.shape[0]==1
            delta_value = df_cytokine.iloc[tmp.index[0], delta_score_idx]
            df.loc[df['pID']==pID, f'{measure}_delta_{tp}'] = delta_value

        return df


class Analysis():

    @staticmethod
    def observed_scores_df(df, out_dir=folders.exports, out_fname='pdp1_observed.csv', save=False):

        master_df = pd.DataFrame(columns=['measure', 'tp', 'obs'])

        measures = [
            'UPDRS_1', 'UPDRS_2', 'UPDRS_3', 'UPDRS_4',
            'HAMA', 'MADRS', 'CCFQ', 'CSSRS', 'ESAPS', 'NPIQ_DIS', 'NPIQ_SEV',
            'PALTEA', 'PALFAMS', 'SWMS', 'RTIFMDRT', 'RTISMDRT']

        for measure in measures:
            measure_df = df.loc[(df.measure==measure)]

            for tp in measure_df.tp.unique():

                observed_dict={'measure':[], 'tp':[], 'obs':[]}
                observed_dict['measure'].append(measure)

                scores = measure_df.loc[(measure_df.tp==tp)].score
                scores = scores.dropna()

                observed_dict['tp'].append(tp)
                observed_dict['obs'].append(
                    f'{round(mean(scores), 1)}±{round(stdev(scores), 1)}')

                master_df = pd.concat([master_df, pd.DataFrame(observed_dict)])

        master_df = master_df.reset_index()
        if save:
            master_df.to_csv(os.path.join(out_dir, out_fname), index=False)

        return master_df

    @staticmethod
    def fivedasc_pairedt(df, out_dir=folders.vitals, out_fname='pdp1_fivedasc_pairedt.csv', save=False):

        rows=[]
        for measure in df.measure.unique():
            a0s = df.loc[(df.measure==measure) & (df.tp=='A0')].sort_values(by='pID')
            b0s = df.loc[(df.measure==measure) & (df.tp=='B0')].sort_values(by='pID')

            if not (np.array_equal(a0s.pID, b0s.pID)):
                print(f'Unequal pID arrays: {measure}')
                continue

            t, tp = ttest_rel(a0s.score, b0s.score, nan_policy='omit')
            w, wp = wilcoxon(a0s.score, b0s.score, nan_policy='omit')
            rows.append([measure, round(t,3), round(tp,4), round(w,3), round(wp,4)])

        df = pd.DataFrame(columns=['measure', 't', 't.p', 'w', 'w.p'], data=rows)

        if save:
            df.to_csv(os.path.join(out_dir, out_fname), index=False)

        return df

    @staticmethod
    def vitals_dmax(df, out_dir=folders.vitals, out_fname='pdp1_vitals_dmax.csv', save=False):

        rows=[]
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
            t, tp = ttest_rel(a0_deltamaxs, b0_deltamaxs, nan_policy='omit')
            w, wp = wilcoxon(a0_deltamaxs, b0_deltamaxs, nan_policy='omit')
            rows.append([
                measure, round(t,3), round(tp,4), round(w,3), round(wp,4),
                round(mean(a0_deltamaxs),2), round(mean(b0_deltamaxs),2)])

        df = pd.DataFrame(columns=['measure', 't', 't.p', 'w', 'w.p', 'A0_deltamax_mean', 'B0_deltamax_mean'], data=rows)

        if save:
            df.to_csv(os.path.join(out_dir, out_fname), index=False)

        return df

    @staticmethod
    def vitals_avg(df, out_dir=folders.vitals, out_fname='pdp1_vitals_avg.csv', save=False):

        rows=[]
        for measure in df.measure.unique():
            a0_avgs=[]
            b0_avgs=[]

            for pID in df.pID.unique():

                a0_avgs.append(
                    df[(df['measure']==measure) & (df['pID']==pID) & (df['tp']=='A0')].score.mean())
                b0_avgs.append(
                    df[(df['measure']==measure) & (df['pID']==pID) & (df['tp']=='B0')].score.mean())

            t, tp = ttest_rel(a0_avgs, b0_avgs, nan_policy='omit')
            w, wp = wilcoxon(a0_avgs, b0_avgs, nan_policy='omit')
            rows.append([
                measure, round(t,3), round(tp,4), round(w,3), round(wp,4),
                round(mean(a0_avgs),2), round(mean(b0_avgs),2)])

        df = pd.DataFrame(columns=['measure', 't', 't.p', 'w', 'w.p', 'A0_avg', 'B0_avg'], data=rows)

        if save:
            df.to_csv(os.path.join(out_dir, out_fname), index=False)

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
    def fahrenheit_to_celsius(temp_f):
        return round((temp_f-32)*5/9,2)

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

    @staticmethod
    def get_REDCap_export():

        df = pd.read_csv(
            os.path.join(
                folders.raw,
                'REDCap export',
                #'PDP1-PDP1clinicalOutcomes_DATA_2023-Jul-17.csv'),
                'PDP1-PDP1clinicalOutcomes_DATA_2024-Jan-18.csv'),
            dtype={
                'record_id': str,
                'vs_notes': str,
                'vs_dose30_time2': str,
                'vs_dose240_time2': str,
                'vs_ortho2_hr_std3': str,
                'vs_extra_measures': str,
                'ccfq_ra_check_bl': str,
                'npid_ra_check_bl': str,
                'npid_ra_check': str,
                'moca_score': str,})

        df = df.loc[(df.record_id.isin(config.valid_str_pIDs))]
        df['record_id'] = df['record_id'].astype(int)

        df = df.rename(columns={
            'record_id': 'pID',
            'redcap_event_name': 'tp'})

        df = df.replace({
            "screening_baseline_arm_1": "bsl",
            "phone_screen_arm_1": "bsl", # for some reason for PsychQ this is how bsl was encoded
            "day_a0_dose_1_arm_1": "A0",
            "day_a1_arm_1": "A1",
            "day_a7_arm_1": "A7",
            "day_b0_dose_2_arm_1": "B0",
            "day_b1_arm_1": "B1",
            "day_b7_arm_1": "B7",
            "day_b11_arm_1": "B11",
            "day_ab18_arm_1": "B18",
            "day_ab25_arm_1": "B25",
            "day_ab30_arm_1": "B30",
            "day_ab90_phone_arm_1": "B90"}, regex=True,)

        return df

    @staticmethod
    def get_deltas(df):

        df['delta_score'] = math.nan
        score_col_idx = df.columns.get_loc('score')
        dscore_col_idx = df.columns.get_loc('delta_score')

        for pID, measure, tp in product(df.pID.unique(), df.measure.unique(), df.tp.unique()):

            if tp=='bsl':
                continue

            # Find baseline row idx
            bsl_row_idx = df.loc[(df.pID==pID) & (df.measure==measure) & (df.tp=='bsl')].index
            assert ((len(bsl_row_idx)==1) or (len(bsl_row_idx)==0))

            # Find score row idx
            tp_row_idx = df.loc[(df.pID==pID) & (df.measure==measure) & (df.tp==tp)].index
            assert ((len(tp_row_idx)==1) or (len(tp_row_idx)==0))

            if ((len(bsl_row_idx)==0) or (len(tp_row_idx)==0)):
                continue

            # Add delta score
            bsl_score = df.iloc[bsl_row_idx[0], score_col_idx]
            tp_score = df.iloc[tp_row_idx[0], score_col_idx]
            df.iloc[tp_row_idx[0], dscore_col_idx] = tp_score-bsl_score

        return df
