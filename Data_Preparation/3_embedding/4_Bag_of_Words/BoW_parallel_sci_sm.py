import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_sci_sm")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = ''.join([char for char in text if not char.isdigit()])  # Remove all digits
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    text = text.strip()
    tokens = nlp(text)
    tokens = [token.lemma_ for token in tokens if token.text not in stop_words]
    return " ".join(tokens)

def parallel_preprocessing(texts, n_jobs=8):
    docs = list(nlp.pipe(texts, n_process=n_jobs))
    return [" ".join([token.lemma_ for token in doc]) for doc in docs]

def create_bow(df, vectorizer):
    X_bow = vectorizer.fit_transform(df["CLEAN_TEXT"])
    bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
    bow_df.insert(0, "HADM_ID", df["HADM_ID"])
    return bow_df

source_path = '../../../Data/unstructured'
# df = pd.read_csv(f'{source_path}/summarized/ALL_first_last_SUMMARY_ONLY/1_t5_small2.csv').head(10)
df = pd.read_csv(f'{source_path}/text/ALL_first_last.csv')


df["TEXT"] = df["TEXT"].apply(lambda x: ''.join([char for char in str(x).lower() if not char.isdigit()]))
df["CLEAN_TEXT"] = parallel_preprocessing(df["TEXT"].tolist(), n_jobs=8)

vectorizer = TfidfVectorizer(max_features=2000)
bow_df = create_bow(df, vectorizer)

bow_df.to_csv(f'{source_path}/emb/BoW/ALL_first_last_sci_sm_2000.csv', index=False)

print('Mission accomplished!')