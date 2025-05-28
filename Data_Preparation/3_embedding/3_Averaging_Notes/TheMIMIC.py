import pandas as pd


class TheMIMIC():
    def __init__(self):
        self.MIMIC_Path = '/lustre/home/almusawiaf/PhD_Projects/MIMIC_resources'
        
        self.tables_with_icustay_id = ['ICUSTAYS', 'CHARTEVENTS', 'OUTPUTEVENTS', 'INPUTEVENTS_MV', 'INPUTEVENTS_CV']

        df_ICUs = self.process_ICUSTAYS()
        self.visits    = df_ICUs['HADM_ID'].unique()
        self.patients  = df_ICUs['SUBJECT_ID'].unique()
        self.ICU_stays = df_ICUs['ICUSTAY_ID'].unique()

        print(f'Patients  = {len(self.patients)}')
        print(f'visits    = {len(self.visits)}')
        print(f'ICU stays = {len(self.ICU_stays)}')
        
    

    def read_table(self, table_name):
        """Generic function to read MIMIC-III CSV tables."""
        print(f'Reading {table_name}.csv file.')
        temp_DF = pd.read_csv(f'{self.MIMIC_Path}/{table_name}.csv')
        print('|---- Reading complete.')
        return temp_DF

    def read_ICUSTAYS(self):
        return self.read_table('ICUSTAYS')

    def read_CHARTEVENTS(self):
        return self.read_table('CHARTEVENTS')

    def read_LABEVENTS(self):
        return self.read_table('LABEVENTS')

    def read_OUTPUTEVENTS(self):
        return self.read_table('OUTPUTEVENTS')

    def read_INPUTEVENTS_MV(self):
        return self.read_table('INPUTEVENTS_MV')

    def read_INPUTEVENTS_CV(self):
        return self.read_table('INPUTEVENTS_CV')

    def read_PRESCRIPTIONS(self):
        return self.read_table('PRESCRIPTIONS')

    def read_PROCEDUREEVENTS_MV(self):
        return self.read_table('PROCEDUREEVENTS_MV')

    def read_DIAGNOSES_ICD(self):
        return self.read_table('DIAGNOSES_ICD')

    def read_DRGCODES(self):
        return self.read_table('DRGCODES')

    def read_NOTEEVENTS(self):
        return self.read_table('NOTEEVENTS')
    
    def read_ADMISSIONS(self):
        return self.read_table('ADMISSIONS')

    def process_NOTEEVENTS(self, visits):
        df_NoteEvents = self.read_NOTEEVENTS()
        new_NoteEvents = df_NoteEvents[df_NoteEvents['HADM_ID'].isin(visits)].reset_index(drop=True)
        new_NoteEvents.info()
        print(new_NoteEvents['CATEGORY'].unique())

        categories_to_keep = ["Nursing/other", "Nursing", "Physician", "Radiology"]
        df_filtered = new_NoteEvents[new_NoteEvents['CATEGORY'].isin(categories_to_keep)]
        df_filtered = df_filtered.reset_index(drop=True)
        return df_filtered
    
    def process_ICUSTAYS(self):
        df_ICUs = self.read_ICUSTAYS()
        df_ICUs = df_ICUs[['SUBJECT_ID','HADM_ID','ICUSTAY_ID','LOS']]

        df_ICUs.dropna(subset=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','LOS'], inplace=True)

        # dropping LOS less than a day
        df_ICUs = df_ICUs[df_ICUs['LOS']>=1]    
        
        return df_ICUs
        


