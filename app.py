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
import csv
import datetime
import collections
from collections import defaultdict
import matplotlib.pyplot as plt
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.manifold import TSNE
import requests





st.title("Streamlit App with Preloaded Models")

# Define variables
text_dict = {}  # Initialize the dictionary to store text data

# Create a file uploader in your Streamlit app
uploaded_files = st.file_uploader("Choose a PDF file", type='pdf', accept_multiple_files=True)

# Process each uploaded PDF file
for uploaded_file in uploaded_files:
    st.write("Ingested file: ", uploaded_file.name)

    try:
        file_data = uploaded_file.read()
        with fitz.open(stream=io.BytesIO(file_data)) as doc:
            concatenated_text = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()

                # Check if the page text contains "TABLE OF CONTENTS" (case-insensitive)
                if re.search(r'table of contents', page_text, re.IGNORECASE):
                    continue  # Skip this page
                concatenated_text += page_text

            # Display the concatenated text
            st.write("Wrote file:", uploaded_file.name)
            text_dict[uploaded_file.name] = concatenated_text


    except Exception as e:
        st.write(f"Error processing {uploaded_file.name}: {str(e)}")


extracted_sections = {}
original_texts = {}

def remove_outreach(text):
    # Find the index of "Outreach" in the text
    outreach_index = text.lower().find("outreach")

    if outreach_index != -1:
        # Extract content after "Outreach"
        cleaned_text = text[outreach_index:]
        return cleaned_text
    else:
        return text

def remove_rationale(text):
    # Define the regular expression pattern
    pattern = re.compile(r"rationale:(.*?)(comment:|$)", re.DOTALL)

    # Remove rationale sections
    text_without_rationale = re.sub(pattern, "comment:", text)

    return text_without_rationale

def remove_members(text):
    # Find the index of "Members" in the text
    members_index = text.lower().find("members")

    if members_index != -1:
        # Extract content after "Outreach"
        cleaned_text = text[members_index:]
        return cleaned_text
    else:
        return text

documents = text_dict.items()

# Define a regular expression pattern to match section headings
pattern = r'Part\s*[IVX]+\s+.*'  # Pattern to match headings like "PART I" or "PART II: Heading"

# Loop through the list of document texts
for document_name, document_text in documents:
    # Exception handling for reports from the 90s with a different format
    if re.search(r'199\d', document_name, re.IGNORECASE):
        pattern_part_iii = r'part iii \n'
        # Search for the pattern in the text
        match = re.search(pattern_part_iii, document_text.lower())
        start_index = match.start()
        end_index = document_text.lower().find('part iv', start_index)  # Case-insensitive search
        #1996 report
        if start_index != -1 and end_index != -1:
            #remove outreach, rationale, and comments sections using previously defined function
            recommendations_section = document_text[start_index:end_index].strip()
            recommendations_section = recommendations_section.lower()
            recommendations_section = remove_rationale(remove_outreach(recommendations_section))
            comment_pattern = r'comment:.*? [a-z]\.'
            cleaned_text = re.sub(comment_pattern, '', recommendations_section, flags=re.DOTALL)
            extracted_sections[document_name] = recommendations_section
            original_texts[document_name] = document_text
        #1998 report (didnt have a part iv)
        if start_index != -1 and end_index == -1:
          #remove outreach, rationale, and comments sections using previously defined function
            recommendations_section = document_text[start_index:len(document_text)].strip()
            recommendations_section = recommendations_section.lower()
            recommendations_section = remove_rationale(remove_outreach(recommendations_section))
            comment_pattern = r'comment:.*? [a-z]\.'
            cleaned_text = re.sub(comment_pattern, '', recommendations_section, flags=re.DOTALL)
            extracted_sections[document_name] = recommendations_section
            original_texts[document_name] = document_text
    elif re.search(r'1998', document_name, re.IGNORECASE):
      start_index = document_text.lower().find('part ii')  # Case-insensitive search
      end_index = document_text.lower().find('part iii', start_index)  # Case-insensitive search
      if start_index != -1 and end_index != -1:
          recommendations_section = document_text[start_index:end_index].strip()
          #remove members and outreach sections
          recommendations_section = remove_members(recommendations_section)
          recommendations_section = recommendations_section.lower()
          extracted_sections[document_name] = recommendations_section
          original_texts[document_name] = document_text
    # Exception handling for reports from 2002 (wild year for this report)
    elif '2002' in document_name:
        # Process the 2002 report differently (e.g., extract sections between "Recommendations and Rationale" and "VA Response to Recommendations")
        start_index = document_text.lower().find('recommendations and rationale')  # Case-insensitive search
        end_index = document_text.lower().find('va response to recommendations', start_index)  # Case-insensitive search
        if start_index != -1 and end_index != -1:
            recommendations_section = document_text[start_index:end_index].strip()
            recommendations_section = recommendations_section.lower()
            extracted_sections[document_name] = recommendations_section
            original_texts[document_name] = document_text

    else:
        # Find all matching headings in the document text for other files
        headings = re.findall(pattern, document_text, re.IGNORECASE)

        # Find the start and end index of the 'RECOMMENDATIONS' section for other files
        start_index = None
        end_index = None

        for i, heading in enumerate(headings):
            if ('recommendations' in heading.lower() and 'va response to recommendations' not in heading.lower()):  # Case-insensitive search
                start_index = document_text.lower().find(heading.lower())  # Case-insensitive search
                if i + 1 < len(headings):
                    end_index = document_text.lower().find(headings[i + 1].lower())  # Case-insensitive search
                recommendations_section = document_text[start_index:end_index].strip()
                recommendations_section = recommendations_section.lower()
                extracted_sections[document_name] = recommendations_section
                original_texts[document_name] = document_text
                break

corpus = pd.DataFrame.from_dict(extracted_sections, orient='index', columns=['recommendations'])
original_texts_df = pd.DataFrame.from_dict(original_texts, orient='index', columns=['original_text'])

# Join the 'corpus' DataFrame onto the 'original_texts_df' DataFrame using the index
corpus = corpus.join(original_texts_df)

# Function to remove specific words and patterns from a document
def remove_words_and_patterns(document, words_to_remove, patterns_to_remove):
    # Split the document into words
    words = document.split()

    # Clean the words by removing specified words and patterns
    cleaned_words = [word for word in words if word.lower() not in words_to_remove and not any(re.match(pattern, word) for pattern in patterns_to_remove)]

    # Join the cleaned words to form the cleaned document
    cleaned_document = " ".join(cleaned_words)
    return cleaned_document

# List of words to remove
words_to_remove = ['veteran','veterans' ,'woman',"women", 'va', 'committee', 'program', 'center', 'study', 'report', 'service', 'within',
                   'include', 'provide', 'ensure', 'develop', 'must', 'need', 'level','department','administration','affairs','veterans benefits administration'
                   ,'acwv']

# List of patterns to remove (e.g., '2.', '11.', '12.', etc.)
# ALSO INCLUDED TEXT TO REMOVE DATES
patterns_to_remove = [r'\d+\.', r'\d+\)',r'\b(?=199\d|20\d{2})\d{4}|january\s\d{1,2}\s|february\s\d{1,2}\s|march\s\d{1,2}\s|april\s\d{1,2}\s|may\s\d{1,2}\s|june\s\d{1,2}\s|july\s\d{1,2}\s|august\s\d{1,2}\s|september\s\d{1,2}\s|october\s\d{1,2}\s|november\s\d{1,2}\s|december\s\d{1,2}\s']

for i, row in corpus.iterrows():
    recommendations = row['recommendations'].lower()
    original_text = row['original_text'].lower()
    # Remove the specified words and patterns from the recommendations and convert it to lowercase
    cleaned_document = remove_words_and_patterns(recommendations, words_to_remove, patterns_to_remove)
    cleaned_document = remove_words_and_patterns(original_text, words_to_remove, patterns_to_remove)
    print(f"Cleaned document at index {i}")

# Load the tokenizer and model for summarization
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Function to summarize text
def summarize_text(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=300, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

corpus['recommendations_summary'] = corpus['recommendations'].apply(summarize_text)
corpus['orig_text_summary'] = corpus['original_text'].apply(summarize_text)

#ingest dictionary workbook
words_dict = {}  # Initialize an empty dictionary

response = requests.get('https://github.com/ls1495/capstone_project_2023/main/words_dict.csv')

if response.status_code == 200:
    # If the file is successfully downloaded, open it using a StringIO object
    from io import StringIO
    csv_data = StringIO(response.text)

    # Create a dictionary to store the data
    words_dict = {}

    # Read data from the CSV file
    csv_reader = csv.DictReader(csv_data)
    
    for row in csv_reader:
        title = row['Title']
        meaning = row['Meaning']
        words_dict[title] = meaning
else:
    print("Failed to download the reference file from GitHub")



#convert word dict dataframe to dictionary
words_dict = {key.lower(): value for key, value in words_dict.items()}

# create stopword list
#create custom stop words vector for
custom_stop_words = ['veteran','veterans' ,'woman',"women", 'va', 'committee', 'program', 'center', 'study', 'report', 'service', 'within',
                   'include', 'provide', 'ensure', 'develop', 'must', 'need', 'level','department','administration','affairs','veterans benefits administration'
                   ]

# Preload the SpaCy model at app startup
download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")

# Default English stop words
default_stop_words = set(CountVectorizer(stop_words="english").get_stop_words())

# Combine default English stop words with custom stop words
stop_words = list(default_stop_words) + custom_stop_words

#function to preprocess the text
def preprocess_text(text):

    # Convert the text to lowercase
    text = text.lower()

    # Remove punctuation
    my_punctuation = '”!"#$%&()*+,\'''/:;<=>?@[\\]’^_`{|}~“•'
    text = text.translate(str.maketrans("", "", my_punctuation))

    # Replace hyphens with spaces
    text = text.replace("-", " ")

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Rejoin the processed words into a single text
    processed_text = " ".join(words)

    # Lowercase the words
    processed_text = processed_text.lower()

    # Replace acronyms with

    return processed_text

#function to replace acronyms with plain text
def replace_words(text, acronym_dict):
    words = text.split()
    replaced_words = [acronym_dict.get(word, word) for word in words]
    replaced_text = ' '.join(replaced_words)
    replaced_text = replaced_text.lower()
    return replaced_text

def spell_check_and_correct(input_text):
    spell = SpellChecker()

    # Split the text into words
    words = input_text.split()

    # Find misspelled words
    misspelled = spell.unknown(words)

    # Correct misspelled words and return corrected text
    corrected_words = []
    for word in words:
        corrected_word = spell.correction(word)
        if word in misspelled and corrected_word is not None and corrected_word != word:
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)

    corrected_text = ' '.join(corrected_words)
    return corrected_text


preprocessed_text = []
spell_checked_text = []
translated_text = []

[preprocessed_text.append(preprocess_text(i)) for i in corpus['recommendations']]
[spell_checked_text.append(spell_check_and_correct(i)) for i in preprocessed_text]
[translated_text.append(replace_words(item, words_dict)) for item in spell_checked_text]
# List of words that should not be considered misspelled  #though document 10

# Load the spaCy language model
nlp = spacy.load("en_core_web_lg")  # You can choose a different model if needed

# Initialize the mapping of old indices to years
index_to_year = {}

for i in range(len(corpus)):
    index_to_year[i] = corpus.index[i]

# Create a new list to store sentences with updated indices
sentences_with_years = []

# Iterate through the sentences and rename the indices
for index, sentence in enumerate(translated_text):
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

# Filter sentences with less than 20 characters
filtered_sentences = [sentence for sentence in sentences_with_years_appended if len(sentence) >= 35]

#Resave sentences as translated_text
translated_text = filtered_sentences

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

for sentence in translated_text:
    year = extract_year(sentence)
    if year is not None:
        date_obj = datetime.date(year=year, month=1, day=1)
        sentences_with_dates.append({'sentence': sentence, 'date': date_obj})

timestamps = [item['date'] for item in sentences_with_dates]
timestamps = pd.to_datetime(timestamps)

# Step 1 - Extract embeddings
embedding_model = SentenceTransformer("all-mpnet-base-v2")

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

tsne_model = TSNE(n_components=5, perplexity=30, n_iter=300)

# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words= stop_words)

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()

# Step 6 - (Optional) Fine-tune topic representations with
# a `bertopic.representation` model
representation_model = KeyBERTInspired()

# All steps together
topic_model = BERTopic(
  embedding_model=embedding_model,          # Step 1 - Extract embeddings
  umap_model=umap_model,                    # Step 2 - Reduce dimensionality
  hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
  representation_model=representation_model # Step 6 - (Optional) Fine-tune topic represenations
)

#NEW Do this all with TSNE
# Step 2 - Extract embeddings and reduce dimensionality using TSNE
embeddings_tsne = embedding_model.encode(translated_text, show_progress_bar=True)

# Apply TSNE for dimensionality reduction
tsne_model = TSNE(n_components=3, perplexity=30, n_iter=300)
reduced_embeddings_tsne = tsne_model.fit_transform(embeddings_tsne)

# Step 3 - Use the reduced embeddings as input to BERTopic for clustering
topic_model_tsne = BERTopic(vectorizer_model=vectorizer_model)

# Fit BERTopic on the reduced embeddings
topics_tsne, probs_tsne = topic_model_tsne.fit_transform(reduced_embeddings_tsne)

#Fit Model to text
topics, probs = topic_model.fit_transform(translated_text)

# Get the document information
document_info = topic_model.get_document_info(translated_text)

document_info

topic_model.visualize_topics()

#Topic Hierachical Clustering
topic_model.visualize_hierarchy()

# Topic Heatmap - Evaluate for Document Matrix and Topic Similarity (future evaluation or removal)
topic_model.visualize_heatmap()

topics_over_time = topic_model.topics_over_time(docs=translated_text,
                                                timestamps=timestamps,
                                                global_tuning=True,
                                                evolution_tuning=True,
                                                nr_bins=15)

topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
