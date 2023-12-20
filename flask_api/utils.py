import os
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from langdetect import detect
from collections import Counter
from nltk.stem import PorterStemmer

def spam_word_list(df:pd.DataFrame)->list:
    """functions to return list of spam words
    Args: DataFrame

    Returns: List of Spam Mails for most common 500 words.
    """
    spam_words = []
    for email in df[df['target']==1]['transformed_text'].tolist():
        # print(type(email))
        if isinstance(email, str):        
            for word in email.split():
                spam_words.append(word)

    words = Counter(spam_words).most_common(500)
    spam_words = [word[0] for word in words]

    return spam_words


def transform_text(email:str)->str:
    stop_words = set(stopwords.words("english"))

    email = email.lower()
    tokenized_email_words = word_tokenize(email)
    
    words = []
    for word in tokenized_email_words:
        if word.isalnum():
            words.append(word)
    
    without_stopwords = [word for word in words if word not in stop_words] 
    ps = PorterStemmer()
    transform_stemmed_text = [ps.stem(word) for word in without_stopwords]
    
    preprocessed_email = " ".join(transform_stemmed_text)

    return preprocessed_email
