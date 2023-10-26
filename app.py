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
import datetime 



# Preload the SpaCy model at app startup
#download("en_core_web_lg")
#nlp = spacy.load("en_core_web_lg")

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

st.dataframe(corpus)
