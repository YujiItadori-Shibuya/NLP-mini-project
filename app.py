import streamlit as st
from textblob import TextBlob
import re
from collections import defaultdict, Counter
import nltk
import os

# Create a callback function to update the text area. This is defined at the
# top level to be accessible by the button widgets.
def add_word_to_input(word_to_add):
    st.session_state.main_input += f" {word_to_add}"

# --- NLTK Corpus Download and Setup ---
# Check if the 'gutenberg' corpus is already downloaded.
try:
    # This will raise a LookupError if the corpus is not found
    nltk.data.find('corpora/gutenberg')
except LookupError:
    st.info("The 'gutenberg' corpus is being downloaded. This may take a moment.")
    try:
        nltk.download('gutenberg')
    except Exception as e:
        st.error(f"Failed to download 'gutenberg' corpus. Please try again. Error: {e}")
        st.stop() # Stop the app if download fails

# --- Language Model Training (N-gram) ---
# This is a simple 2-gram (bigram) model trained on a small corpus for demonstration.
CORPUS_SENTENCES = nltk.corpus.gutenberg.sents('austen-sense.txt')

# Create a bigram model for next word prediction
ngram_model = defaultdict(Counter)
for sentence in CORPUS_SENTENCES:
    # Add start and end tokens for better modeling
    clean_sentence = [word.lower() for word in sentence if word.isalpha()]
    bigrams = list(nltk.ngrams(['<start>'] + clean_sentence, 2))
    for first, second in bigrams:
        ngram_model[first][second] += 1

def predict_next_word(current_text, top_n=5):
    """
    Predicts the next most likely words based on a bigram model.
    """
    last_word = current_text.strip().lower().split()[-1] if current_text.strip() else '<start>'
    suggestions = []
    
    if last_word in ngram_model:
        # Get the most common words following the last word
        most_common = ngram_model[last_word].most_common(top_n)
        for word, count in most_common:
            # Avoid the start token in suggestions
            if word != '<start>':
                suggestions.append(word)
    
    return suggestions

# --- Autocorrect Logic ---
def autocorrect_sentence(text):
    """
    Corrects spelling mistakes in a given sentence using TextBlob.
    """
    blob = TextBlob(text)
    # The correct() method returns a new TextBlob with corrected words
    corrected_blob = blob.correct()
    return str(corrected_blob)

# --- Streamlit UI and Application Logic ---
st.set_page_config(
    page_title="Autocorrect & Autofill",
    page_icon="✍️"
)

st.title("✍️ Autocorrect & Autofill MODEL")
st.markdown("Demonstrating real-time autocorrect and next-word prediction using Python.")

# The single text input widget
user_input = st.text_area(
    "Start typing here:",
    placeholder="e.g., I havv a gret idea...",
    height=200,
    key="main_input"
)

# --- Dynamic Processing and Display ---

# Check if the user has typed anything
if user_input:
    # Display autocorrected text
    st.markdown("---")
    st.subheader("✅ Autocorrected Text")
    corrected_text = autocorrect_sentence(user_input)
    st.write(corrected_text)

    # Display autofill suggestions
    st.subheader("💡 Autofill Suggestions")
    suggestions = predict_next_word(user_input)
    
    if suggestions:
        # Format the suggestions as a clickable list
        st.write("Click a word to add it:")
        
        # Use columns for a nice layout
        cols = st.columns(len(suggestions))
        for i, word in enumerate(suggestions):
            with cols[i]:
                # Use a callback function with on_click and args to fix the API exception
                st.button(word, on_click=add_word_to_input, args=(word,))
    else:
        st.write("No suggestions yet. Keep typing!")