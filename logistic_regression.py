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

# Define model paths
model1_path = 'logistic_regression_model.pkl'
model2_path = 'multinomialnb_model.pkl'
model3_path = 'gradient_boosting_model.pkl'
model4_path = 'DecisionTree_model.pkl'
model5_path = 'xgboost_model.pkl'

# Define vectorizer path
vectorizer_path = 'tfidf_vectorizer.pkl'

st.title('Multilingual Food Reviews Sentiment Analyzer')

# User input
st.subheader('Enter Sentence')
new_sentence = st.text_input('Enter a sentence:')

st.subheader('Translate to Language')
convert_lang = st.selectbox('Select language:', list(languages.keys()))

# Model selection
selected_model = st.selectbox('Select Model:', ['Logistic Regression', 'Multinomial Naive Bayes', 'Gradient Boosting','Decision Tree','XGBoost'])
model_path = None

if selected_model == 'Logistic Regression':
    model_path = model1_path
elif selected_model == 'Multinomial Naive Bayes':
    model_path = model2_path
elif selected_model == 'Gradient Boosting':
    model_path = model3_path
elif selected_model == 'Decision Tree':
    model_path = model4_path
elif selected_model == 'XGBoost':
    model_path = model5_path

if new_sentence:
    translator = Translator(from_lang='en', to_lang=convert_lang)
    translation = translator.translate(new_sentence)
    st.write('Translated Sentence:', translation)

    # Load selected model and vectorizer
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)

    # Sentiment analysis
    st.subheader('Sentiment Analysis')

    # Vectorize the preprocessed sentence
    X_new = vectorizer.transform([new_sentence])
    
    # Predict sentiment
    predicted_sentiment = model.predict(X_new)

    # Display sentiment
    sentiment = "Positive" if predicted_sentiment[0] == 1 else "Negative"
    st.write('Predicted Sentiment:', sentiment)
