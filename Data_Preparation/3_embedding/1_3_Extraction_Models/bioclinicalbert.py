import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os


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
    # Load BioClinicalBERT model
    print("Loading BioClinicalBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model loaded.")
        
    # ===============================================================================
    # File paths
    thePath     = os.getenv('thePath', '../../../Data/unstructured')
    smrz_model  = os.getenv('smrz_model', '1_t5_small2')
    # input_file  = os.getenv('input_file',  'ALL_first_last')
    
    input_path  = f'{thePath}/summarized/{smrz_model}.csv'
    output_path = f'{thePath}/emb/{smrz_model}.csv'
    
    # ===============================================================================
    # Load input data
    print("Loading input data...")
    data = pd.read_csv(input_path, dtype={"HADM_ID": str, "SUBJECT_ID": str})
    print(f"Dataset loaded. Total rows: {len(data)}")

    # ===============================================================================
    # Ensure text fields are non-null and converted to strings
    print("Preprocessing text columns...")
    data["TEXT"]    = data["TEXT"].fillna("").astype(str)
    data["SUMMARY"] = data["SUMMARY"].fillna("").astype(str)

    # ===============================================================================
    # Initialize storage for embeddings
    emb_text, emb_summary = [], []
    batch_size = 32  # Adjust based on GPU memory

    # ===============================================================================
    # Generate embeddings for 'text'
    print("Generating embeddings for 'text' column...")
    for i in tqdm(range(0, len(data), batch_size), desc="Processing 'text'"):
        batch_texts = data["TEXT"][i:i+batch_size].tolist()
        batch_embeddings = generate_embeddings_batch(batch_texts, tokenizer, model, device)
        emb_text.extend(batch_embeddings)

    # ===============================================================================
    # Generate embeddings for 'summary'
    print("Generating embeddings for 'summary' column...")
    for i in tqdm(range(0, len(data), batch_size), desc="Processing 'summary'"):
        batch_summaries = data["SUMMARY"][i:i+batch_size].tolist()
        batch_embeddings = generate_embeddings_batch(batch_summaries, tokenizer, model, device)
        emb_summary.extend(batch_embeddings)

    # ===============================================================================
    # Create new dataframe with embeddings
    print("Creating final dataframe with embeddings...")
    result_df = pd.DataFrame({
        "HADM_ID": data["HADM_ID"],
        "SUBJECT_ID": data["SUBJECT_ID"],
        "EMB_TEXT": emb_text,
        "EMB_SUMMARY": emb_summary
    })

    # ===============================================================================
    # Save to CSV
    print(f"Saving embeddings to {output_path}...")
    result_df.to_csv(output_path, index=False)
    print(f"Embeddings successfully saved to {output_path}.")
    # result_df