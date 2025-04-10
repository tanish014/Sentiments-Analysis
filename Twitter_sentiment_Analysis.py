import numpy as np
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
column_names = ["PostID", "Post Description", "Date", "Language Code", "Full Language", "Sentiment"]
dataset = pd.read_csv("1000_dataset_english (2).csv", names=column_names, skiprows=1)

# Drop NaN
dataset = dataset.dropna(subset=['Post Description', 'Sentiment'])

# Filter only positive and negative sentiments
dataset = dataset[dataset['Sentiment'].isin(['positive', 'negative'])]

# Preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # remove URLs
        text = re.sub(r'[^a-z\s]', '', text)  # remove special chars
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    return ""

dataset['Processed_Text'] = dataset['Post Description'].apply(preprocess_text)

# Remove empty text
dataset = dataset[dataset['Processed_Text'].str.strip() != ""]

# Balance classes
positive = dataset[dataset['Sentiment'] == 'positive']
negative = dataset[dataset['Sentiment'] == 'negative']
min_len = min(len(positive), len(negative))

positive_bal = resample(positive, replace=False, n_samples=min_len, random_state=42)
negative_bal = resample(negative, replace=False, n_samples=min_len, random_state=42)

balanced_dataset = pd.concat([positive_bal, negative_bal]).sample(frac=1, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(balanced_dataset['Processed_Text'])
y = balanced_dataset['Sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=200, max_depth=25, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model & vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("✅ Model and vectorizer saved successfully!")
