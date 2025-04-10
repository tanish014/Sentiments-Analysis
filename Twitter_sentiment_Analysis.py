import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

# Load your model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# Streamlit UI
st.title("Twitter Sentiment Analysis")
user_input = st.text_input("Enter a tweet:")

if st.button("Analyze"):
    preprocessed = preprocess_text(user_input)
    vectorized_input = vectorizer.transform([preprocessed])
    prediction = model.predict(vectorized_input)[0]

    st.write("### Sentiment:")
    if prediction == 0:
        st.error("Negative")
    elif prediction == 1:
        st.success("Positive")
    else:
        st.info("Neutral")
