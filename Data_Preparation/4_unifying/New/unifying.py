import os
import pickle
import pandas as pd
import numpy as np
import ast
from multiprocessing import Pool
from functools import partial

class Unifying():
    def __init__(self, structured_path, unstructured_path, saving_path, the_model):
        self.embedding_model = the_model
        self.saving_path = f'{saving_path}/{self.embedding_model}'
        
        self.XB, self.XD, self.XL, self.XM, self.XP, self.XS, self.LoS, new_emb = self.extract_shared_data(structured_path, unstructured_path)
        self.process_and_save_embeddings(new_emb)
        self.concatenate_and_save()

    def extract_shared_data(self, structured_path, unstructured_path):
        print('Loading the data')
        XB = self.load_pickle(f'{structured_path}/X_B.pkl')
        XD = self.load_pickle(f'{structured_path}/X_D.pkl')
        XL = self.load_pickle(f'{structured_path}/X_L.pkl')
        XM = self.load_pickle(f'{structured_path}/X_M.pkl')
        XP = self.load_pickle(f'{structured_path}/X_P.pkl')
        XS = self.load_pickle(f'{structured_path}/X_S.pkl')
        VIY = self.load_pickle(f'{structured_path}/VIY.pkl')
        df_emb = pd.read_csv(f'{unstructured_path}/merged_embeddings_{self.embedding_model}.csv')
        
        df_viy = pd.DataFrame(VIY, columns=['HADM_ID', 'ICU_ID', 'LoS'])
        common_hadm_ids = np.intersect1d(df_viy['HADM_ID'].values, df_emb['HADM_ID'].values)
        filtered_viy = df_viy[df_viy['HADM_ID'].isin(common_hadm_ids)].sort_values(by='HADM_ID').reset_index(drop=True)
        new_emb = df_emb[df_emb['HADM_ID'].isin(common_hadm_ids)].sort_values(by='HADM_ID').reset_index(drop=True)
        mask = np.isin(VIY[:, 0], common_hadm_ids)

        return XB[mask], XD[mask], XL[mask], XM[mask], XP[mask], XS[mask], VIY[mask, 2], new_emb

    def process_embedding_column(self, col_data):
        col, data = col_data
        print(f"Processing {col}...")
        try:
            processed_data = data.apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
            embeddings_array = np.vstack(processed_data.to_numpy())
            self.save_list_as_pickle(embeddings_array, self.saving_path, col)
            print(f"Saved {col} with shape {embeddings_array.shape}")
        except Exception as e:
            print(f"Error processing column {col}: {e}")

    def process_and_save_embeddings(self, merged_df):
        print('Processing and saving the embeddings ...')
        embedding_columns = ['EMB_TEXT', 
                             'EMB_1_t5_small2_SUMMARY', 
                             'EMB_3_bart_large_cnn_SUMMARY', 
                             'EMB_4_medical_summarization_SUMMARY']
        
        with Pool(processes=4) as pool:  # Adjust number of processes based on available cores
            pool.map(self.process_embedding_column, [(col, merged_df[col]) for col in embedding_columns])

    def classify_los_3_classes(self, los_list):
        return [0 if los < 3 else 1 if 3 <= los <= 7 else 2 for los in los_list]

    def concatenate_and_save(self):    
        print('Reading the data ...')
        T0 = self.load_pickle(f'{self.saving_path}/EMB_TEXT.pkl')
        T1 = self.load_pickle(f'{self.saving_path}/EMB_1_t5_small2_SUMMARY.pkl')
        T2 = self.load_pickle(f'{self.saving_path}/EMB_3_bart_large_cnn_SUMMARY.pkl')
        T3 = self.load_pickle(f'{self.saving_path}/EMB_4_medical_summarization_SUMMARY.pkl')

        Z = np.concatenate((self.XB, self.XD, self.XL, self.XM, self.XP), axis=1)
        ZS = np.concatenate((Z, self.XS), axis=1)
        ZST0 = np.concatenate((ZS, T0), axis=1)
        ZST1 = np.concatenate((ZS, T1), axis=1)
        ZST2 = np.concatenate((ZS, T2), axis=1)
        ZST3 = np.concatenate((ZS, T3), axis=1)
        
        Y = self.classify_los_3_classes(self.LoS)

        print('|--- Saving ...')
        self.save_list_as_pickle(Z, self.saving_path, 'Z')
        self.save_list_as_pickle(ZS, self.saving_path, 'ZS')
        self.save_list_as_pickle(T0, self.saving_path, 'T0')
        self.save_list_as_pickle(T1, self.saving_path, 'T1')
        self.save_list_as_pickle(T2, self.saving_path, 'T2')
        self.save_list_as_pickle(T3, self.saving_path, 'T3')
        self.save_list_as_pickle(ZST0, self.saving_path, 'ZST0')
        self.save_list_as_pickle(ZST1, self.saving_path, 'ZST1')
        self.save_list_as_pickle(ZST2, self.saving_path, 'ZST2')
        self.save_list_as_pickle(ZST3, self.saving_path, 'ZST3')
        self.save_list_as_pickle(Y, self.saving_path, 'Y')

    def save_list_as_pickle(self, L, given_path, file_name):
        if not os.path.exists(given_path):
            os.makedirs(given_path)
            print(f'\tDirectory created: {given_path}')
        print(f'\tSaving to {given_path}/{file_name}.pkl')
        with open(os.path.join(given_path, f'{file_name}.pkl'), 'wb') as file:
            pickle.dump(L, file)

    def load_pickle(self, thePath):
        with open(thePath, 'rb') as f:
            data = pickle.load(f)
        return data

if __name__ == '__main__':
    structured_path = '../../../Data/structured'
    unstructured_path = '../../../Data/unstructured/emb'
    saving_path = '../../../Data/XY3'
    models = ['bioclinicalbert', 'clinicalbert', 'gatortron']
    for the_model in models:
        _ = Unifying(structured_path, unstructured_path, saving_path, the_model)