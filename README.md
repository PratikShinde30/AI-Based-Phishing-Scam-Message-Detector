# AI-Based-Phishing-Scam-Message-Detector
AI-Based Phishing / Scam Message Detector

This project is a Machine Learning-based application that detects whether an SMS or text message is Spam / Suspicious or Legitimate.

The model uses TF-IDF Vectorization to convert text messages into numerical features and Multinomial Naive Bayes for classification.

The training script (train_model.py) loads and preprocesses the SMS dataset, trains the machine learning model, evaluates its performance using Accuracy, Precision, Recall, and F1-score, and saves the trained model.

The Streamlit application (app.py) provides a simple user interface where users can enter a message and receive a prediction along with the estimated spam probability and model performance metrics.

Technologies Used

Python

Pandas

Scikit-learn

TF-IDF Vectorizer

Multinomial Naive Bayes

Streamlit

Joblib
