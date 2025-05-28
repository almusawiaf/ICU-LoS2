import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import os

# Environment variables for file paths
input_file = os.getenv('input_file')
output_file = os.getenv('output_file')

# Device setup (GPU if available, else CPU)
device = 0 if torch.cuda.is_available() else -1

# Model and tokenizer initialization
MODEL_NAME = "Falconsai/medical_summarization"
SUMMARY_RATIO = 2  # Controls the compression ratio of the summary

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
summarizer = pipeline("summarization", model=MODEL_NAME, device=device)

def summarize_text(text, max_tokens=512):
    """
    Summarizes long medical texts by chunking them into sentence-based segments
    and processing them through the summarization model.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Sentence-based chunking
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        encoded_length = len(tokenizer.encode(current_chunk + sentence, add_special_tokens=False))
        if encoded_length < max_tokens:
            current_chunk += sentence + '. '
        else:
            if current_chunk:
                tokenized_chunk = tokenizer.encode(current_chunk, add_special_tokens=True, truncation=True, max_length=max_tokens)
                chunks.append(tokenizer.decode(tokenized_chunk))
            current_chunk = sentence + '. '

    if current_chunk:
        tokenized_chunk = tokenizer.encode(current_chunk, add_special_tokens=True, truncation=True, max_length=max_tokens)
        chunks.append(tokenizer.decode(tokenized_chunk))

    # Summarize each chunk with dynamically controlled compression
    chunk_summaries = []
    for chunk in chunks:
        try:
            input_length = len(tokenizer.encode(chunk, add_special_tokens=False))
            max_len = max(50, min(150, input_length // SUMMARY_RATIO))  # Dynamic max length
            min_len = max(25, max_len // 2)

            result = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
            summary = result[0]['summary_text']
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
        summary = summarize_text(text)
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