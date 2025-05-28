import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
from joblib import Parallel, delayed
import os

# Load input and output file paths from environment variables
input_file = os.getenv('input_file')
output_file = os.getenv('output_file')

# Configuration
min_len = 60  # Minimum length of the summary
max_len = 150  # Maximum length of the summary
max_tokens = 512  # Maximum token limit for model input
batch_size = 32  # Adjust batch size based on available memory

# Initialize tokenizer and model
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Enable mixed precision (FP16) for better performance on GPUs
if device.type == "cuda":
    model.half()  # Convert model to half precision

# Initialize the summarization pipeline
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    framework="pt",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
)

def preprocess_text(text):
    """
    Splits text into chunks of <= max_tokens for the model.
    """
    if pd.isna(text) or not isinstance(text, str):
        return []

    # Tokenize entire text first
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Split tokens into chunks of max_tokens length
    chunks = [tokens[i: i + max_tokens] for i in range(0, len(tokens), max_tokens)]

    # Decode chunks back into text
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def summarize_text(text):
    """
    Summarizes a single text by processing it in chunks.
    """
    chunks = preprocess_text(text)
    if not chunks:
        return ""

    summaries = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            batch_summaries = summarizer(
                ["summarize: " + chunk for chunk in batch],
                max_length=max_len,
                min_length=min_len,
                truncation=True,  # Ensures input remains within limit
                do_sample=False
            )
            summaries.extend([summary['summary_text'] for summary in batch_summaries])
        except Exception as e:
            print(f"Error summarizing batch: {e}")
            summaries.extend([""] * len(batch))  # Handle errors gracefully
    return " ".join(summaries)

def process_record(record):
    """
    Summarizes a single record.
    """
    text = record['TEXT']
    summary = summarize_text(text)
    return {
        "SUBJECT_ID": record['SUBJECT_ID'],
        "HADM_ID": record['HADM_ID'],
        "TEXT": summary
    }

def main(input_file, output_file):
    """
    Main function to summarize the text column in the input dataframe.
    """
    # Read the input dataframe
    print("Loading data...")
    df = pd.read_csv(input_file)
    
    # Process records in parallel for efficiency
    print("Generating summaries in parallel...")
    results = Parallel(n_jobs=8)(
        delayed(process_record)(row) for _, row in tqdm(df.iterrows(), total=len(df))
    )

    # Create a new dataframe with the summaries
    dataframe2 = pd.DataFrame(results)

    # Save the new dataframe to a CSV file
    dataframe2.to_csv(output_file, index=False)
    print(f"Summarized data saved to {output_file}")

# Run script
if __name__ == "__main__":
    main(input_file, output_file)
