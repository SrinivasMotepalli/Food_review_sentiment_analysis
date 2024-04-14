import streamlit as st
import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from translate import Translator

# Load the trained model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Load the vectorizer
def load_vectorizer(vectorizer_path):
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

def load_function(function_path):
    with open(function_path, 'rb') as f:
        preprocess_and_vectorize_text = pickle.load(f)
    return preprocess_and_vectorize_text

# Define languages
languages = {
    'en': 'English(India)',
    'gu-IN': 'Gujarati(India)',
    'hi-IN': 'Hindi(India)',
    'kn-IN': 'Kannada(India)',
    'kok-IN': 'Konkani(India)',
    'mr-IN': 'Marathi(India)',
    'pa-IN': 'Punjabi(India)',
    'sa-IN': 'Sanskrit(India)',
    'ta-IN': 'Tamil(India)',
    'te-IN': 'Telugu(India)'
}

st.title('Multilingual Comment Analyzer')

# Load model and vectorizer
lmodel = load_model('logistic_regression_model.pkl')
vectorizer = load_vectorizer('tfidf_vectorizer.pkl')
preprocess_and_vectorize_text = load_function('')


# User input
st.subheader('Enter Sentence')
new_sentence = st.text_input('Enter a sentence:')

# Language translation
st.subheader('Translate to Language')
convert_lang = st.selectbox('Select language:', list(languages.keys()))

if new_sentence:
    translator = Translator(from_lang='en', to_lang=convert_lang)
    translation = translator.translate(new_sentence)
    st.write('Translated Sentence:', translation)

    # Sentiment analysis
    st.subheader('Sentiment Analysis')

    # Vectorize the preprocessed sentence
    vectorized_sentence = preprocess_and_vectorize_text(new_sentence, vectorizer)
    
    # Predict sentiment
    predicted_sentiment = lmodel.predict(vectorized_sentence)

    # Display sentiment
    sentiment = "Positive" if predicted_sentiment[0] == 1 else "Negative"
    st.write('Predicted Sentiment:', sentiment)
