import os
import re
import sys
import string
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from langdetect import detect
from collections import Counter

from utils import spam_word_list, transform_text

app = Flask(__name__)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "dataset/cleaned.csv")
model_path = os.path.join(base_dir, "models/mnb_model.pkl")
etc_model_path = os.path.join(base_dir, "models/etc.pkl")
vectorizer_path = os.path.join(base_dir, "models/TfidfVectorizer.pkl")

df = pd.read_csv(dataset_path)
spam_word_list = set(spam_word_list(df))

# Load the pre-trained model and TF-IDF vectorizer
with open(vectorizer_path, 'rb') as file:
    tfidf = pickle.load(file)

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(etc_model_path, 'rb') as file:
    etc_model = pickle.load(file)

@app.route('/', methods=['POST'])
def classify_email():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid JSON data"}), 400

        email_data = request.get_json()
        email = email_data.get("email", "")
        
        if not email:
            return jsonify({"error": "Email content is missing"}), 400

        cleaned_email = transform_text(email)
        transformed_email = tfidf.transform([cleaned_email])
        prediction = model.predict(transformed_email)
        common_words = set([word for word in spam_word_list if word in email.split()])
        language = detect(email)

        output = "Spam" if prediction[0] == 1 else "Not Spam"
        result = {
            "classification": output,
            "common_spam_words": list(common_words),
            "email_length": len(email),
            "language": language
        }

        return jsonify(result)
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True) 






