import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle



with open('logistic_regression_model.pkl', 'rb') as f:
    lmodel = pickle.load(f)

def preprocess_and_vectorize_text(text, vectorizer):
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

# Example usage:
new_sentence = input("Enter a sentence: ")
vectorized_sentence = preprocess_and_vectorize_text(new_sentence, vectorizer)
predicted_sentiment = lmodel.predict(vectorized_sentence)
print("Predicted Sentiment:", "Positive" if predicted_sentiment[0] == 1 else "Negative")
