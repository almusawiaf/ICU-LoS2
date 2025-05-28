import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import pickle

from TheMIMIC import *

def save_list_as_pickle(L, given_path, file_name):
    # Ensure the directory exists
    if not os.path.exists(given_path):
        os.makedirs(given_path)
        print(f'\tDirectory created: {given_path}')
    
    # Save the list as a pickle file
    print(f'\tSaving to {given_path}/{file_name}.pkl')
    with open(os.path.join(given_path, f'{file_name}.pkl'), 'wb') as file:
        pickle.dump(L, file)
        
def load_pickle(thePath):
    with open(thePath, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_embeddings_batch(texts, tokenizer, model, device):
    """
    Generate CLS token embeddings for a batch of texts using BioClinicalBERT.
    """
    inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().tolist()
    return cls_embeddings

if __name__ == "__main__":
    thePath     = os.getenv('thePath', '../../../Data/unstructured')
    emb_model  = os.getenv('emb_model', 'Bio_ClinicalBERT')
    batch_size  = int(os.getenv('batch_size', 32))

    if emb_model == "Bio_ClinicalBERT":
        print("Loading BioClinicalBERT model...")
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
    elif emb_model =="ClinicalBERT":
        print("Loading ClinicalBERT model...")
        tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        
    elif emb_model =="gatortron-base":
        print("Loading GatorTron model...")
        tokenizer = AutoTokenizer.from_pretrained('UFNLP/gatortron-base')
        model = AutoModel.from_pretrained('UFNLP/gatortron-base')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("|--- Model loaded.")
        
    # ===============================================================================  
    print('Loading the data...')
    
    
    # Input and output paths
    output_path = f'{thePath}/emb/ALL_Notes2/emb_{emb_model}.csv'

    VIY = load_pickle('../../../Data/structured/VIY.pkl')
    the_visits = VIY[:,0]

    MIMIC = TheMIMIC()
    NoteEvents = MIMIC.read_NOTEEVENTS()
    print('|--- Files reading completed ... ')

    NoteEvents['CATEGORY'] = NoteEvents['CATEGORY'].str.strip()
    category_distribution = NoteEvents['CATEGORY'].value_counts()
    
    print('|--- Preparing the data ... ')
    NoteEvents = NoteEvents.dropna(subset=['HADM_ID'])
    NoteEvents = NoteEvents[NoteEvents["HADM_ID"].isin(the_visits)]

    categories_to_keep = ["Nursing/other", "Nursing", "Physician", "Radiology"]
    NoteEvents2 = NoteEvents[NoteEvents['CATEGORY'].isin(categories_to_keep)].reset_index(drop=True)

    NoteEvents2['CHARTDATE'] = pd.to_datetime(NoteEvents2['CHARTDATE'], format='%Y-%m-%d')
    NoteEvents2['CHARTTIME'] = pd.to_datetime(NoteEvents2['CHARTTIME'], format='%Y-%m-%d %H:%M:%S')
    NoteEvents2.sort_values(by='CHARTTIME', ascending=True, inplace=True)

    NoteEvents2 = NoteEvents2.reset_index(drop=True)
    needed_cols = ['HADM_ID', 'CATEGORY', 'TEXT']
    data = NoteEvents2[needed_cols]

    print('   |--- Preparing Completed ... ')

    print(f"Dataset loaded. Total rows: {len(data)}")

    # ===============================================================================  
    # Ensure text fields are non-null and converted to strings
    print("Preprocessing text columns...")
    col = "TEXT"
    data[col] = data[col].fillna("").astype(str)

    # ===============================================================================  
    # Initialize storage for embeddings
    embeddings = {"HADM_ID": data["HADM_ID"], "CATEGORY": data["CATEGORY"]}

    # ===============================================================================  
    # Generate embeddings for TEXT
    print(f"Generating embeddings for '{col}' column...")
    emb_list = []
    for i in tqdm(range(0, len(data), batch_size), desc=f"Processing '{col}'"):
        batch_texts = data[col][i:i+batch_size].tolist()
        batch_embeddings = generate_embeddings_batch(batch_texts, tokenizer, model, device)
        emb_list.extend(batch_embeddings)
    
    # Store embeddings as a new column
    embeddings[f"EMB"] = emb_list

    # ===============================================================================  
    # Create new DataFrame with embeddings
    print("Creating final dataframe with embeddings...")
    result_df = pd.DataFrame(embeddings)

    # ===============================================================================  
    # Save to CSV
    print(f"Saving embeddings to {output_path}...")
    result_df.to_csv(output_path, index=False)
    print(f"Embeddings successfully saved to {output_path}.")
