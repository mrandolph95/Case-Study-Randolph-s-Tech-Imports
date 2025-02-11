# Randolph's Tech Imports: iPhone Review Sentiment Prediction

### Overview

This project focuses on predicting the sentiment of iPhone reviews, classifying them into three categories: high, neutral, and negative. The sentiment prediction is achieved using a TensorFlow-based machine learning model, which outperforms previous models built with Naive Bayes and scikit-learn.

### Installation

To run this project, you need to install the following dependencies:

  - TensorFlow
  - Flask
  - Pandas
  - scikit-learn

### Training

The model was trained on a dataset from Kaggle, consisting of iPhone reviews. The data was cleaned and preprocessed, with reviews being categorized into three classes: high, negative, and neutral.

Key steps in the training process:

  - Data cleaning: Removing unnecessary characters and formatting issues.
  - Tokenization: Converting text data into numerical features.
  - Model: A deep learning model built using TensorFlow.

### Results

The performance of the model was evaluated using accuracy metrics:

  - Naive Bayes model: 0.68 accuracy
  - Naive Bayes using scikit-learn: 0.72 accuracy
  - TensorFlow model: 0.84 accuracy (best-performing model)

