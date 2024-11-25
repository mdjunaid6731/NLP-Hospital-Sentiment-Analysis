from flask import Flask, request, jsonify, render_template
import re
import pandas as pd
import pickle
import base64
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"] 
    prediction, graph = get_prediction_and_graph(text)
    return jsonify({"prediction": prediction, "graph": graph})

@app.route("/predict_real_time_feedbacks", methods=["GET"])
def predict_real_time_feedbacks():
    # Google Reviews Scraper API endpoint
    url = "https://google-reviews-scraper.p.rapidapi.com/getReviews"
    # Parameters for the request
    querystring = {"searchId": "U2FsdGVkX19DQKaHczi72qIWRpZ%2BnF4PapaG0l3YRy%2BpuVl098YG7ZyIj6Zemn87n2UiXP%2FN98lxFAi9K8nWJA%3D%3D", "sort": "<REQUIRED>", "nextpage": "false"}
    # Headers for the request
    headers = {
        "X-RapidAPI-Key": "1d16c5feb2msh0ba848dfaaf7a4fp17baffjsnd59ec1270efe",
        "X-RapidAPI-Host": "google-reviews-scraper.p.rapidapi.com"
    }
    # Send request to Google Reviews Scraper API
    response = requests.get(url, headers=headers, params=querystring)
    
    if response.status_code == 200:
        data = response.json()
        # Extracting comments
        if "data" in data and isinstance(data["data"], list):
            comments = [review.get("comment") for review in data["data"]]
            sentiments = [get_prediction_and_graph(comment)[0] for comment in comments]
            positive_reviews = sentiments.count("Positive")
            negative_reviews = sentiments.count("Negative")
            return jsonify({"positive_reviews": positive_reviews, "negative_reviews": negative_reviews})
        else:
            return jsonify({"error": "Failed to fetch reviews"})
    else:
        return jsonify({"error": "Failed to fetch reviews. Status code: {}".format(response.status_code)})

def get_prediction_and_graph(text):
    # Load model and vectorizer
    model = pickle.load(open("Models/model_xgb.pkl", "rb"))
    vectorizer = pickle.load(open("Models/countVectorizer.pkl", "rb"))
    stemmer = PorterStemmer()
    # Preprocess text
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)
    # Vectorize text
    X = vectorizer.transform([review]).toarray()
    # Make prediction
    prediction = model.predict(X)[0]
    # Prepare graph
    if prediction == 1:
        color = 'green'
    else:
        color = 'red'
    plt.figure(figsize=(5,5))
    plt.bar(['Negative', 'Positive'], [1 - prediction, prediction], color=color)
    plt.xlabel('Sentiment')
    plt.ylabel('Probability')
    plt.title('Sentiment Distribution')
    graph_buffer = BytesIO()
    plt.savefig(graph_buffer, format='png')
    plt.close()
    graph_buffer.seek(0)
    graph_encoded = base64.b64encode(graph_buffer.getvalue()).decode("ascii")
    return ("Positive" if prediction == 1 else "Negative", graph_encoded)

if __name__ == "__main__":
    app.run(debug=True)
