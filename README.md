📧 Email Spam Detection using Machine Learning
📌 Problem Statement

Spam emails pose serious security threats including phishing, scams, and malware. This project builds a machine learning–based spam detection system that classifies emails as Spam or Ham using natural language processing techniques.

🎯 Objectives

Clean and preprocess raw email text data

Extract meaningful features using NLP techniques

Train and compare multiple ML models

Evaluate performance using industry-standard metrics

Optimize the best-performing model

Explore deployment strategies for real-world email filtering

🛠️ Tech Stack

Programming: Python

Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, NLTK

Models:

Multinomial Naive Bayes

Support Vector Machine (SVM)

Decision Tree

Techniques:

TF-IDF Vectorization

Text Cleaning & Tokenization

Hyperparameter Tuning

Cross-Validation

🔍 Dataset Overview

Spam: ~13.41%
Ham: ~86.59%

Highly imbalanced dataset handled using appropriate evaluation metrics.

📊 Model Performance
Model	Accuracy	Precision	Recall	F1-Score
Decision Tree	~95%	Moderate	Moderate	Moderate
SVM	~97%	High	High	High
Multinomial Naive Bayes	98.49%	Very High	Excellent	Best

✔ Best Model: Multinomial Naive Bayes
🏆 Key Insights

Spam messages commonly contain words like “free”, “call”, “txt”, “now”

TF-IDF effectively captures textual importance

Naive Bayes outperformed complex models due to its probabilistic nature and efficiency with text data

🚀 Future Improvements

Deep learning (LSTM, Transformers)

Handling adversarial spam techniques

Real-time deployment using Flask / FastAPI

Integration with email clients
