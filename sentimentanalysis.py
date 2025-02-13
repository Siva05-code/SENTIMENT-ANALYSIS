#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Importing libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[13]:


nltk.download('stopwords')


# In[14]:


import nltk
nltk.download('stopwords', download_dir="/Users/sivakarthick/HUB/AmazonReviews")


# In[15]:


nltk.data.path.append("/Users/sivakarthick/HUB/AmazonReviews")
stop_words = set(stopwords.words('english'))


# In[16]:


df = pd.read_csv("Reviews.csv") 


# In[17]:


# Basic info
print(df.head())  
print('\n')
print(df.info())


# In[18]:


# Selecting only relevant columns
df = df[['Text', 'Score']]


# In[19]:


#Ratings to binary sentiment labels:
# Score 4, 5 -> Positive (1), Score 1, 2, 3 -> Negative (0)
df['Sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)


# In[20]:


def preprocess_text(t):
    t = t.lower() 
    t = re.sub(r'\d+', '', t)  
    t = t.translate(str.maketrans('', '', string.punctuation))  
    t = t.strip()
    words = t.split()
    words = [i for i in words if i not in stopwords.words('english')]
    return " ".join(words)


# In[37]:


df['Cleaned_Text'] = df['Text'].apply(preprocess_text)


# In[38]:


# Text into numerical form using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)  
X = tfidf.fit_transform(df['Cleaned_Text']).toarray()
y = df['Sentiment']


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression()
model.fit(X_train, y_train)


# In[40]:


y_pred = model.predict(X_test)


# In[41]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[42]:


plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[43]:


new_reviews_df = pd.read_csv("Test.csv")  

print(new_reviews_df.columns)

review_column = "Text"    
product_id_col = "ProductId"  
user_id_col = "UserId"  

new_reviews_df["Cleaned_Review"] = new_reviews_df[review_column].apply(preprocess_new_text)

vectorized_reviews = tfidf.transform(new_reviews_df["Cleaned_Review"]).toarray()

predicted_sentiments = model.predict(vectorized_reviews)

new_reviews_df["Predicted_Sentiment"] = predicted_sentiments
new_reviews_df["Sentiment_Label"] = new_reviews_df["Predicted_Sentiment"].map({1: "Positive", 0: "Negative"})

output_df = new_reviews_df[[product_id_col, user_id_col, "Predicted_Sentiment", "Sentiment_Label"]]

output_df.to_csv("Result.csv", index=False)

print("Predictions are saved to 'Result.csv'!")

