import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Download stopwords
nltk.download('stopwords')

# Load dataset
dataset = pd.read_csv("1000_dataset_english (2).csv", encoding="utf-8")

# Display first few rows
print(dataset.head())

# Check missing values
print(dataset.isnull().sum())


# Manually specify the column names
column_names = ["PostID", "Post Description", "Date", "Language Code", "Full Language", "Sentiment"]

# Read CSV properly
dataset = pd.read_csv("1000_dataset_english (2).csv", names=column_names, skiprows=1)

# Display the first few rows
print(dataset.head())

print(dataset.columns)
print(dataset.shape)

import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if not already available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):  # Ensure text is string
        text = text.lower()  # Lowercase
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
        text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
        return text
    return ""

# Apply processing to 'Post Description'
dataset['Processed_Text'] = dataset['Post Description'].apply(preprocess_text)

# Verify
print(dataset[['Post Description', 'Processed_Text']].head())  # Check processed text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2

# Vectorize text using TF-IDF
# Remove neutral sentiment from the dataset
dataset = dataset[dataset['Sentiment'] != 'neutral']

# Now continue with feature extraction
vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1,3), stop_words='english')
X = vectorizer.fit_transform(dataset['Processed_Text']).toarray()
y = dataset['Sentiment']

# Check label distribution
print(y.value_counts())  # Should now show only "positive" and "negative"


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Random Forest model with optimized hyperparameters
model = RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=5, min_samples_leaf=2, 
                               random_state=42, class_weight="balanced_subsample", bootstrap=True)  #class_weight: helps with imbalanced data
 # Enable bootstrapping

model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print(dataset['Sentiment'].value_counts(normalize=True))  # Show class balance

import pickle

# Save the trained model
pickle.dump(model, open('model.pkl', 'wb'))

# Save the TF-IDF vectorizer
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model and Vectorizer Saved!")

# Select random samples from the dataset for testing
sample_data = dataset.sample(n=5, random_state=42)  # Pick 5 random samples
sample_texts = sample_data['Processed_Text'].tolist()

# Transform input text using the same vectorizer
sample_features = vectorizer.transform(sample_texts).toarray()

# Predict sentiment
predictions = model.predict(sample_features)

# Display results
for text, sentiment in zip(sample_texts, predictions):
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")

import numpy as np

# Get feature importance
feature_importances = model.feature_importances_
top_n = 20  # Number of top features to display

# Get top features
top_features_idx = np.argsort(feature_importances)[::-1][:top_n]
top_features = [vectorizer.get_feature_names_out()[i] for i in top_features_idx]

print("Top Important Features in Sentiment Classification:")
print(top_features)
