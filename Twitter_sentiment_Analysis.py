import streamlit as st
import re
import nltk
import pickle
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    return ""

# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered")
st.title("📊 Twitter Sentiment Analysis")
st.markdown("Enter a tweet to analyze its sentiment (positive or negative).")

# Input box
user_input = st.text_input("Enter tweet text:", "")

if user_input:
    preprocessed = preprocess_text(user_input)
    
    if preprocessed.strip() == "":
        st.warning("The text is too short or became empty after preprocessing.")
    else:
        vectorized = vectorizer.transform([preprocessed]).toarray()
        prediction = model.predict(vectorized)[0]

        # Display result
        if prediction == 'positive':
            st.success("✅ Sentiment: Positive")
        elif prediction == 'negative':
            st.error("❌ Sentiment: Negative")
        else:
            st.info("ℹ️ Sentiment: Neutral")
            dataset['Sentiment'].value_counts(normalize=True)

