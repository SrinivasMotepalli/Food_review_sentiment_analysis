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
import xgboost as xgb

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

# Define model paths
model_paths = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Multinomial Naive Bayes': 'multinomialnb_model.pkl',
    'Gradient Boosting': 'XGBoost_model.pkl',
    # Add more models as needed
}

# Define vectorizer paths
vectorizer_paths = {
    'Logistic Regression': 'tfidf_vectorizer.pkl',
    'Multinomial Naive Bayes': 'tfidf_vectorizer.pkl',
    'Gradient Boosting': 'tfidf_vectorizer.pkl',
    # Add more vectorizers as needed
}

st.title('Multilingual Food Reviews  Analyzer')

# User input
st.subheader('Enter Sentence')
new_sentence = st.text_input('Enter a sentence:')

st.subheader('Translate to Language')
convert_lang = st.selectbox('Select language:', list(languages.keys()))

# Model selection
selected_model = st.selectbox('Select Model:', list(model_paths.keys()))

if new_sentence:
    translator = Translator(from_lang='en', to_lang=convert_lang)
    translation = translator.translate(new_sentence)
    st.write('Translated Sentence:', translation)

    # Load selected model and vectorizer
    model = load_model(model_paths[selected_model])
    vectorizer = load_vectorizer(vectorizer_paths[selected_model])

    # Sentiment analysis
    st.subheader('Sentiment Analysis')

    # Vectorize the preprocessed sentence
    X_new = vectorizer.transform([new_sentence])
    
    # Predict sentiment
    predicted_sentiment = model.predict(X_new)

    # Display sentiment
    sentiment = "Positive" if predicted_sentiment[0] == 1 else "Negative"
    st.write('Predicted Sentiment:', sentiment)
