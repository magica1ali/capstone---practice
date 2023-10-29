#Load Modules 
import sys
import os
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import string
import spacy
from spacy.cli.download import download
from transformers import LayoutLMv3FeatureExtractor, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from PIL import Image
import glob2
import re
import fitz
import csv
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
import streamlit as st
import io

# Preload the SpaCy model at app startup
python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

#split text into sentences and add the document year to each sentence 
@st.cache_data
def spacyLayer(text,corpus):
    index_to_year = {}

    for i in range(len(corpus)):
        index_to_year[i] = corpus.index[i]

    # Create a new list to store sentences with updated indices
    sentences_with_years = []

    # Iterate through the sentences and rename the indices
    for index, sentence in enumerate(text):
        year = index_to_year.get(index, None)
    if year is not None:
        sentences_with_years.append(f"{year}: {sentence}")

# Initialize an empty list to store sentences with year appended
    sentences_with_years_appended = []

# Iterate through each document in sentences_with_years
    for document in sentences_with_years:
        # Split the document into sentence text and year
        year, sentence_text = document.split(": ", 1)

    # Parse the sentence using spaCy
    doc = nlp(sentence_text)

    # Iterate through each sentence in the document
    for sentence in doc.sents:
        # Append the sentence with year appended
        sentence_with_year = f"{sentence.text} ({year})"
        sentences_with_years_appended.append(sentence_with_year)
        
    filtered_sentences = [sentence for sentence in sentences_with_years_appended if len(sentence) >= 50]

    return filtered_sentences

#Extracts timestamps for topics over time visulization 
@st.cache_data
def datetime_layer(text):
    def extract_year(sentence):
    # Find the last four digits in the sentence
    for i in range(len(sentence) - 4, -1, -1):
        chunk = sentence[i:i+4]
        if chunk.isdigit() and len(chunk) == 4:
            # Parse the last four characters as an integer
            year = int(chunk)
            # Check if the year is valid (not 0 or less)
            if year > 0:
                return year
    # Return None if no valid year is found
    return None

    # Create a list of dictionaries with 'sentence' and 'date' attributes
    sentences_with_dates = []

    for sentence in text:
        year = extract_year(sentence)
    if year is not None:
        date_obj = datetime.date(year=year, month=1, day=1)
        sentences_with_dates.append({'sentence': sentence, 'date': date_obj})
        
    timestamps = [item['date'] for item in sentences_with_dates]
    
    timestamps = pd.to_datetime(timestamps)
    
    return timestamps



### Load Bertopic Model  - with predetermined parameters 
@st.cache_data
def bertopic_model(text):
  #parameters for bertopic
  embedding_model = SentenceTransformer("all-mpnet-base-v2")
  umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
  hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
  vectorizer_model = CountVectorizer(stop_words= stop_words)
  ctfidf_model = ClassTfidfTransformer()
  representation_model = KeyBERTInspired()

  #load model with parameters
  topic_model = BERTopic(
  embedding_model=embedding_model,          # Step 1 - Extract embeddings
  umap_model= umap_model,                    # Step 2 - Reduce dimensionality
  hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
  representation_model=representation_model # Step 6 - (Optional) Fine-tune topic represenations
)
  
  #Fit Model to text
  topics, probs = topic_model.fit_transform(text)
  # Get the document information
  document_info = topic_model.get_document_info(text)
  return topic_model


st.session_state['preprocessed_text']
st.session_state['corpus']

#run spaCyLayer, timestamp and berTopic functions 

corpus_split = spacyLayer(st.session_state['preprocessed_text'],st.session_state['corpus'])
timestamps = datetime_layer(corpus_split)
topic_model = bertopic_model(corpus_split)

st.write(topic_model.visualize_topics())