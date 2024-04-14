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


from translate import Translator
print("INDIAN LANGUAGE TRANSLATOR")
print("lang_code language \n en English(India) \n gu-IN Gujarati(India) \n hi-IN Hindi(India) \n kn-IN Kannada(India) \n kok-IN Konkani(India) \n mr-IN Marathi(India) \n pa-IN Punjabi(India) \n sa-IN Sanskrit(India) \n ta-IN Tamil(India) \n te-IN Telugu(India)")
say_lang=input("ENTER THE LANGUAGE IN WHICH YOU ARE FAMILIAR WITH (ENTER THE LANG_CODE) :")
convert_lang=input("ENTER THE LANGUAGE YOU WANT TO CONVERT INTO (ENTER THE LANG_CODE) :")
translator=Translator(from_lang = say_lang,to_lang=convert_lang)
sentence=input("ENTER THE SENTENCE YOU WANT TO CONVERT INTO :")
translation=translator.translate(sentence)
print(translation)

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
