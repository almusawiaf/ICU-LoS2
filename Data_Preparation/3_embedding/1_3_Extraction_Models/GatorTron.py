from transformers import AutoTokenizer, AutoModel
import torch
import os

import pandas as pd
from tqdm import tqdm

def generate_embeddings_batch(texts, tokenizer, model, device):
    """
    Generate CLS token embeddings for a batch of texts using GatorTron.
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
    # Load GatorTron model
    print("Loading GatorTron model...")
    tokenizer = AutoTokenizer.from_pretrained('UFNLP/gatortron-base')
    model = AutoModel.from_pretrained('UFNLP/gatortron-base')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model loaded.")
        
    # ===============================================================================  
    # File paths
    thePath = os.getenv('thePath', '../../../Data/unstructured')

    # Input and output paths
    input_path = f'{thePath}/summarized/merged_summaries.csv'
    output_path = f'{thePath}/emb/merged_embeddings_gatortron.csv'

    # ===============================================================================  
    # Load input data
    print("Loading input data...")
    data = pd.read_csv(input_path, dtype={"HADM_ID": str, "SUBJECT_ID": str})
    print(f"Dataset loaded. Total rows: {len(data)}")

    # ===============================================================================  
    # Ensure text fields are non-null and converted to strings
    print("Preprocessing text columns...")
    text_columns = ["TEXT", "1_t5_small2_SUMMARY", "3_bart_large_cnn_SUMMARY", "4_medical_summarization_SUMMARY"]
    for col in text_columns:
        data[col] = data[col].fillna("").astype(str)

    # ===============================================================================  
    # Initialize storage for embeddings
    embeddings = {"HADM_ID": data["HADM_ID"], "SUBJECT_ID": data["SUBJECT_ID"]}
    batch_size = 32  # Adjust based on GPU memory

    # ===============================================================================  
    # Generate embeddings for each text column
    for col in text_columns:
        print(f"Generating embeddings for '{col}' column...")
        emb_list = []
        for i in tqdm(range(0, len(data), batch_size), desc=f"Processing '{col}'"):
            batch_texts = data[col][i:i+batch_size].tolist()
            batch_embeddings = generate_embeddings_batch(batch_texts, tokenizer, model, device)
            emb_list.extend(batch_embeddings)
        
        # Store embeddings as a new column
        embeddings[f"EMB_{col}"] = emb_list

    # ===============================================================================  
    # Create new DataFrame with embeddings
    print("Creating final dataframe with embeddings...")
    result_df = pd.DataFrame(embeddings)

    # ===============================================================================  
    # Save to CSV
    print(f"Saving embeddings to {output_path}...")
    result_df.to_csv(output_path, index=False)
    print(f"Embeddings successfully saved to {output_path}.")
