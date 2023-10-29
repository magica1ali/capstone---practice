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

st.dataframe(corpus)

# Load the tokenizer and model for summarization

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Function to summarize text
@st.cache_data
def summarize_text(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=300, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

corpus['recommendations_summary'] = corpus['recommendations'].apply(summarize_text)
corpus['orig_text_summary'] = corpus['original_text'].apply(summarize_text)

st.dataframe(corpus)
