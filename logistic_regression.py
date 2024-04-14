import streamlit as st
import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from translate import Translator

# Load preprocessed dataset
df_preprocessed = pd.read_csv('preprocessed_dataset.csv')

# Split data into features and target
X = df_preprocessed['Text']
y = df_preprocessed['Sentiment']

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
lmodel = LogisticRegression()
lmodel.fit(X_train, y_train)

# Define a function to preprocess and vectorize text
def preprocess_and_vectorize_text(text):
    # Lowercasing
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Removing punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    # Vectorize the preprocessed sentence
    vectorized_sentence = vectorizer.transform([preprocessed_text])
    return vectorized_sentence

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
    vectorized_sentence = preprocess_and_vectorize_text(new_sentence)
    
    # Predict sentiment
    predicted_sentiment = lmodel.predict(vectorized_sentence)

    # Display sentiment
    sentiment = "Positive" if predicted_sentiment[0] == 1 else "Negative"
    st.write('Predicted Sentiment:', sentiment)


