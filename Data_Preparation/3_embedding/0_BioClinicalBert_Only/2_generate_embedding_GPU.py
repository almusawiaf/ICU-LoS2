import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def generate_embeddings_batch(texts, tokenizer, model, device):
    """
    Generate embeddings for a batch of texts using BioClinicalBERT.
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
    # File paths
    # text_column = 'COMPLAINT'
    # text_column = 'CHIEF_COMPLAINT'
    # text_column = 'MEDICATIONS'
    # text_column = 'PHYSICAL_EXAMINATION'

    # text_column = 'TEXT'
    # input_file  = f'data/text/merged_texts.csv'
    # output_file =  f'data/emb/merged_texts.csv'

    input_file  = '../../data/text/summary_NOTEEVENTS_60_150_2.csv'
    output_file = '../../data/emb/summary_NOTEEVENTS_60_150_2.csv'
    text_column = 'SUMMARY'
    
    
    # Load BioClinicalBERT model
    print("Loading BioClinicalBERT...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model loaded.")

    # Load input data
    print("Loading input data...")
    data = pd.read_csv(input_file)
    print(f"Input data loaded. Total rows: {len(data)}")

    print("Ensuring consistent data types for the text column...")
    data[text_column] = data[text_column].fillna("").astype(str)

    # Generate embeddings in batches
    print("Generating embeddings...")
    batch_size = 32  # Adjust based on GPU memory
    embeddings = []

    for i in tqdm(range(0, len(data), batch_size), desc="Generating Embeddings"):
        batch_texts = data[text_column][i:i+batch_size].tolist()
        batch_embeddings = generate_embeddings_batch(batch_texts, tokenizer, model, device)
        embeddings.extend(batch_embeddings)

    # Save embeddings
    print("Saving embeddings to output file...")
    data['EMBEDDINGS'] = embeddings
    data.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")
