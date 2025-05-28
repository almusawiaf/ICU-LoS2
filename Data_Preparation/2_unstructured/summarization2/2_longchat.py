import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

# Environment variables for file paths
input_file = os.getenv('input_file')
output_file = os.getenv('output_file')

# Initialize summarizer with LongChat
model_name = "lmsys/longchat-7b-16k"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=True)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_text(text, max_input_length=16000, max_output_length=512):
    """
    Summarizes a single text input using LongChat. Handles text exceeding max_input_length by chunking.

    Args:
        text: The input text to summarize.
        max_input_length: Maximum token length for input.
        max_output_length: Maximum token length for output summary.

    Returns:
        A summarized version of the input text.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    try:
        # Tokenize the text without truncation
        tokenized_text = tokenizer(text, truncation=False, return_tensors="pt")
        input_ids = tokenized_text["input_ids"][0]

        # Chunk input if it exceeds max_input_length
        if len(input_ids) > max_input_length:
            chunks = [input_ids[i:i + max_input_length] for i in range(0, len(input_ids), max_input_length)]
        else:
            chunks = [input_ids]

        # Generate summaries for each chunk
        summaries = []
        for chunk in chunks:
            inputs = {"input_ids": chunk.unsqueeze(0).to(device)}
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_output_length,
                min_length=100,  # Adjust as needed
                length_penalty=2.0,
                num_beams=4,
                no_repeat_ngram_size=3
            )
            summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

        # Combine summaries from all chunks
        return " ".join(summaries)
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return ""

def main():
    # Read CSV
    print("Loading data...")
    df = pd.read_csv(input_file).head(10)
    print(f"Total records: {len(df)}")

    # Initialize SUMMARY column
    df['TEXT'] = ""

    # Generate summaries
    print("Generating summaries...")
    summaries = []
    for text in tqdm(df['TEXT'], desc="Summarizing"):
        summary = summarize_text(text)
        summaries.append(summary)

    # Assign summaries to the new column
    df['TEXT'] = summaries

    # Create a new dataframe with the results
    dataframe2 = df[['SUBJECT_ID', 'HADM_ID', 'TEXT']]

    # Save to CSV
    dataframe2.to_csv(output_file, index=False)
    print(f"Summarized data saved to {output_file}")

if __name__ == "__main__":
    main()