from flask import Flask, request, jsonify, render_template
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon (only first time needed)
nltk.download("vader_lexicon", quiet=True)

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    
    if compound >= 0.05:
        sentiment = "Positive Sentiment"
        color = "green"
    elif compound <= -0.05:
        sentiment = "Negative Sentiment"
        color = "red"
    else:
        sentiment = "Neutral Sentiment"
        color = "gray"
    
    # Return JSON response
    return jsonify({
        "sentiment": sentiment,
        "color": color
    })

if __name__ == "__main__":
    app.run(debug=True)
