import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
import os
import nltk
from nltk.corpus import stopwords

# Ensure stopwords and wordnet are downloaded
nltk.download('stopwords')

# Load Spacy model
nlp = spacy.load("en_core_sci_md")

# Define stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = ''.join([char for char in text if not char.isdigit()])  # Remove digits
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    text = text.strip()
    tokens = nlp(text)
    tokens = [token.lemma_ for token in tokens]  # Apply lemmatization
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    return " ".join(tokens)

def parallel_preprocessing(texts, n_jobs=8):
    docs = list(nlp.pipe(texts, n_process=n_jobs, disable=["parser", "ner"]))  # Optimize Spacy processing
    return [" ".join([token.lemma_ for token in doc if token.text not in stop_words]) for doc in docs]

def create_bow(df, vectorizer):
    vectorizer.fit(df["CLEAN_TEXT"])
    X_bow = vectorizer.transform(df["CLEAN_TEXT"])
    bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
    bow_df.insert(0, "HADM_ID", df["HADM_ID"])
    return bow_df

model_name = 'sci_md'
# Set source path safely
source_path = os.path.abspath("../../../Data/unstructured")

# Load dataset
df = pd.read_csv(f'{source_path}/text/ALL_first_last.csv')

# Preprocessing
df["TEXT"] = df["TEXT"].apply(lambda x: ''.join([char for char in str(x).lower() if not char.isdigit()]))
df["CLEAN_TEXT"] = parallel_preprocessing(df["TEXT"].tolist(), n_jobs=8)

# Create BoW representation
vectorizer = TfidfVectorizer(max_features=2000, analyzer="word")
bow_df = create_bow(df, vectorizer)

# Save output
bow_df.to_csv(f'{source_path}/emb/BoW/ALL_first_last_{model_name}_2000.csv', index=False)

print('Mission accomplished!')
