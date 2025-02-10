import re
import numpy as np
import pandas as pd
import joblib

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

reviews = pd.read_csv('iphone.csv')
reviews.head()

column_name_mapping = {'productAsin':'Product_Number','country':'Country','date':'Date',
                       'isVerified':'Verified','ratingScore':'Rating_Score','reviewTitle':'Review_Title',
                       'reviewDescription':'Review_Description','reviewUrl':'Review_Url',
                       'reviewedIn':'Reviewer_Location','variant':'Product_Type','variantAsin':'Product_Type_Number'}

reviews.rename(columns=column_name_mapping, inplace=True)

emojis = (r'[\U0001F600-\U0001F64F' 
        r'\U0001F300-\U0001F5FF'  
        r'\U0001F680-\U0001F6FF'  
        r'\U0001F700-\U0001F77F'  
        r'\U0001F780-\U0001F7FF'  
        r'\U0001F800-\U0001F8FF'  
        r'\U0001F900-\U0001F9FF'  
        r'\U0001FA00-\U0001FA6F'  
        r'\U0001FA70-\U0001FAFF'  
        r'\U00002764\ufe0f'  
        r']+')

# Removes emoji characters from Review_Description column
reviews['Review_Description'] = reviews['Review_Description'].apply(lambda x: re.sub(emojis, '', str(x)) if isinstance(x, str) else x)

# Asigns rating score a category of High, Low, and Neutral
reviews['Rating_Category'] = np.where(reviews['Rating_Score'] >= 4, 'High', np.where(reviews['Rating_Score'] <=2, 'Low', 'Neutral'))

# Asigns rating category a number
rating_mapping = {'High':1,'Low':2,'Neutral':3}
reviews['Numerical_Rating_Category'] = reviews['Rating_Category'].map(rating_mapping)

reviews = reviews.dropna(subset=['Review_Description']) # Drop rows with null values
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
X = vectorizer.fit_transform(reviews['Review_Description'])
y = reviews['Numerical_Rating_Category']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

x_train_dense = X_train.toarray()
x_test_dense = X_test.toarray()

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train) # Converts labels to numbers
y_test_encoded = label_encoder.fit_transform(y_test)

model = Sequential([
    Dense(16, activation='relu', input_shape=(x_train_dense.shape[1],)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

history = model.fit(x_train_dense, y_train_encoded,
                   epochs=20,
                   batch_size=32,
                   validation_data=(x_test_dense, y_test_encoded))

def predict_customer_sentiment(review_text):
    review_tfidf = vectorizer.transform([review_text])
    
    sentiment_labels = {1:'Positive', 2:'Negative', 3:'Neutral'}
    
    sentiment_probs = model.predict(review_tfidf.toarray())
    sentiment_class = sentiment_probs.argmax(axis=1)[0]
    
    sentiment_class += 1
    
    return sentiment_labels[sentiment_class]


# API

joblib.dump(vectorizer, 'vectorizer.pkl')
model.save('sentiment_model.h5')



