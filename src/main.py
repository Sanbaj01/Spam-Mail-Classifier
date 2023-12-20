import numpy as np
import pandas as pd
import re
import contractions
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import os

import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import pickle

ps = PorterStemmer()

directory = '/home/sanbaj/Python Projects/spam_mail_classification/email-spam-classification/dataset/'
file_name = 'SPAM_MAILS_CSV.csv'

df = pd.read_csv(os.path.join(directory,file_name), encoding='ISO-8859-1')
df = df.drop_duplicates(keep='first')

# Data Preprocessing
def transform_text(text: object, max_repeating_chars: int = 5) -> str:
    """
    Transform the input text by applying various text cleaning steps.

    Parameters:
    - text (str): The input text to be transformed.
    - max_repeating_chars (int): The maximum number of consecutive repeating characters to allow.
                                 Defaults to 5.

    Returns:
    - str: The cleaned and transformed text.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    
    # Remove HTML tags and content
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove numbers (including digits and numbers with commas or periods)
    text = re.sub(r'\d+[\d,\.]*', ' ', text)

    # Remove newline characters
    text = text.replace('\n', ' ')

    # Remove punctuations and unwanted characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra spaces and strip leading/trailing spaces
    text = ' '.join(text.split())
    
    # Tokenize the text
    words = text.split()
        
    # Remove sequences of repeating characters
    # words = [re.sub(r'(\w)\1{%d,}' % (max_repeating_chars - 1), r'\1', word) for word in words]
    cleaned_words = []

    for word in words:
        cleaned_word = re.sub(r'(\w)\1{%d,}' % (max_repeating_chars - 1), r'\1', word)
        cleaned_words.append(cleaned_word)

    # Remove single-character words and stopwords
    stop_words = set(stopwords.words("english"))
    # words = [word for word in cleaned_words if len(word) > 2 and word not in stop_words]
    filtered_words = []

    for word in cleaned_words:
        if len(word) > 2 and word not in stop_words:
            filtered_words.append(word)
    
    # Rejoin the words into a single string
    cleaned_text = ' '.join(filtered_words)
    
    return cleaned_text

# Stemming the emails
def stem_email(email):
    stemmed_words = []
    for word in email.split(" "):
        stemmed_word = ps.stem(word)
        stemmed_words.append(stemmed_word)
    stemmed_email = " ".join(stemmed_words)
    
    return stemmed_email

df['stemmed_mail'] = df['transformed_text'].apply(stem_email)



def spam_words(df:pd.DataFrame)->list:
    spam_words = []
    for email in df[df['target']==1]['transformed_text'].tolist():
        for word in email.split():
            spam_words.append(word)

    words = Counter(spam_words).most_common(500)
    spam_words = [word[0] for word in words]

    return spam_words

# Model Building
X = df['stemmed_mail']
Y = df['target']

tfidf = TfidfVectorizer(max_features=30000)
tfidf.fit_transform(X)

X_tfidf_vectorized = tfidf.fit_transform(X).toarray()

# Train_Test_Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf_vectorized, Y, test_size=0.2, random_state=32)


mnb = MultinomialNB()
mnb_model = mnb.fit(X_train, y_train)
y_pred = mnb_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(precision_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# pickle.dump(tfidf, open('../models/TfidfVectorizer.pkl','wb'))
# pickle.dump(mnb, open('../models/mnb_model.pkl','wb'))