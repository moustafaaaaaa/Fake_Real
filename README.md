📰 Fake or Real News Detection
Detect whether a news article is Fake or Real using supervised machine learning algorithms. This project demonstrates the application of text preprocessing, feature extraction with TF-IDF, and classification using Multinomial Naive Bayes and K-Nearest Neighbors (KNN), with detailed evaluation.

🚀 Overview
Fake news has become one of the major challenges in digital media. In this project, I used natural language processing (NLP) and machine learning (ML) techniques to detect whether a piece of news is real or fake based on its content.

The project uses a labeled dataset of real and fake news articles and applies:

Text Cleaning & Preprocessing

TF-IDF Vectorization

Model Training with Multinomial Naive Bayes and KNN

Evaluation Metrics: Precision, Recall, F1-score, and Accuracy

📊 Model Performance
✅ Multinomial Naive Bayes
Accuracy: 95%

Class	Precision	Recall	F1-Score	Support
Real (0)	0.95	0.96	0.96	1024
Fake (1)	0.95	0.95	0.95	956
Macro Avg	0.95	0.95	0.95	1980
Weighted Avg	0.95	0.95	0.95	1980

✅ K-Nearest Neighbors (KNN)
Accuracy: 97%

Class	Precision	Recall	F1-Score	Support
Real (0)	0.99	0.96	0.97	1024
Fake (1)	0.96	0.99	0.97	956
Macro Avg	0.97	0.97	0.97	1980
Weighted Avg	0.97	0.97	0.97	1980

<img width="558" height="557" alt="لقطة شاشة 2025-07-27 202617" src="https://github.com/user-attachments/assets/ac03c960-b3d6-40a6-8c00-43e41e1e32b7" />


🛠️ Tech Stack
Python

Scikit-learn

Pandas / NumPy

Matplotlib / Seaborn

NLP: Spacy

Jupyter Notebook
