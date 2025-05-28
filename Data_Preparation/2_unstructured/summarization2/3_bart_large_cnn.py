import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import os

# Environment variables for file paths
input_file = os.getenv('input_file')
output_file = os.getenv('output_file')

# Device setup (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and tokenizer initialization
MODEL_NAME = "facebook/bart-large-cnn"
MAX_TOKENS = 1024  # Maximum token limit for input chunks

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

def summarize_text(text, max_tokens=MAX_TOKENS):
    """
    Summarizes long texts by chunking them into token-based segments
    and processing them through the BART summarization model.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Tokenize text and split into manageable chunks
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [
        tokens[i:i + max_tokens]
        for i in range(0, len(tokens), max_tokens)
    ]

    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        try:
            inputs = torch.tensor([chunk]).to(device)
            outputs = model.generate(inputs, max_length=200, min_length=60, do_sample=False)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            chunk_summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            chunk_summaries.append("")

    return " ".join(chunk_summaries)

def process_dataframe(df):
    """
    Processes the input DataFrame to generate summaries for the 'TEXT' column.
    Returns a new DataFrame with 'SUBJECT_ID', 'HADM_ID', and 'summarized TEXT'.
    """
    print("Generating summaries...")
    summaries = []
    for text in tqdm(df['TEXT'], desc="Summarizing"):
        summary = summarize_text(text, max_tokens=MAX_TOKENS)
        summaries.append(summary)

    # Create a new DataFrame with the required columns
    summarized_df = pd.DataFrame({
        'SUBJECT_ID': df['SUBJECT_ID'],
        'HADM_ID': df['HADM_ID'],
        'summarized TEXT': summaries
    })
    return summarized_df

def main():
    # Load the input CSV file
    print("Loading data...")
    df = pd.read_csv(input_file, dtype={'TEXT': str})
    print(f"Total records: {len(df)}")

    # Process the DataFrame to generate summaries
    summarized_df = process_dataframe(df)

    # Save the summarized DataFrame to a new CSV file
    summarized_df.to_csv(output_file, index=False)
    print(f"Summarized data saved to {output_file}")

if __name__ == "__main__":
    main()