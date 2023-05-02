# -*- coding: utf-8 -*-

pip install snscrape

import pandas as pd
import snscrape.modules.twitter as sntwitter
import os

#Natural Language Toolkit
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import string
import re
import textblob
from textblob import TextBlob

import warnings

# Using OS library to call CLI commands in Python
os.system("snscrape --jsonl --max-results 10000 --since 2023-03-29 twitter-search 'Piala Dunia U20' > text-PDU20-tweets.json")

# creates a pandas dataframe
tweets_df_PDU20 = pd.read_json('text-PDU20-tweets.json', lines=True)
tweets_df_PDU20.head()

"""Load the Data"""

df_PDU20 = tweets_df_PDU20[['date', 'rawContent','renderedContent','user','replyCount','retweetCount','likeCount','lang','place','hashtags','viewCount']].copy()
print(df_PDU20.shape)

"""Twitter Data Cleaning , Preprocessing and Exploratory Data Analysis"""

df_wcu20 = df_PDU20.drop_duplicates('renderedContent')

print(df_wcu20.shape)
df_wcu20.head()

df_wcu20.info()

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

plt.figure(figsize=(10, 4))
ax = sns.heatmap(df_wcu20.isnull(), cbar=True, cmap="plasma", yticklabels=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.xlabel("Nama Kolom")
plt.title("Nilai yang Hilang pada Kolom")
plt.show()

import plotly.graph_objects as go

tweet_loc = df_wcu20['place'].value_counts()
print(tweet_loc.head(10))

"""Twitter Data Cleaning and Preprocessing"""

def preprocessing(text):

  #Hapus Mention
  text = re.sub(r'@\w+', '', text)

  #Hapus Hashtags
  text = re.sub(r'#\w+', '', text)

  #Hapus Link
  text = re.sub('http://\S+|https://\S+', '', text)
  text = re.sub('http[s]?://\S+', '', text)
  text = re.sub(r"http\S+", "", text)

  #Hapus karakter baris baru
  text = re.sub('[\r\n]+', ' ', text)

  #Hapus Double Space pada Karakter
  text = re.sub('\s+',' ', text)

  #Convert HTML references
  text = re.sub('&amp', 'and', text)
  text = re.sub('&lt', '<', text)
  text = re.sub('&gt', '>', text)
  
  #Konversi ke lowercase
  text = text.lower()

  return text

df_wcu20['processed_text'] = df_wcu20['renderedContent'].apply(preprocessing)

print(df_wcu20['processed_text'].head())

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_bigram(df_wcu20['processed_text'], 20)
print(common_words)

df_wcu20_2 = pd.DataFrame(common_words, columns = ['TweetText' , 'count'])

import cufflinks as cf

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

"""Sentiment Analysis"""

nltk.downloader.download('vader_lexicon')

df_wcu20_2.head(10)

df_wcu20.head()

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ininsiasi SentimentIntensityAnalyzer baru
wcu20_sentiment = SentimentIntensityAnalyzer()

# Generate skor Sentimen
sentiment_scores = df_wcu20['processed_text'].apply(wcu20_sentiment.polarity_scores)

df_wcu20['hashtags'].value_counts()

hashtags_counts = df_wcu20['hashtags'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
sns.color_palette("bright")
explode = [0.1, 0, 0, 0, 0, 0, 0, 0, 0 ,0]

plt.pie(hashtags_counts.values, labels=hashtags_counts.index, explode=explode, autopct='%1.1f%%')
plt.title('Jumlah Presentase Hastags')
plt.show()

lang_counts = df_wcu20['lang'].value_counts().head()
sns.barplot(x=lang_counts.index, y=lang_counts.values)
sns.set(rc={'figure.figsize':(6,3)})

plt.title('Jumlah Bahasa')
plt.xlabel('Bahasa')
plt.ylabel('Jumlah')
plt.show()

df_wcu20['date_column'] = df_wcu20['date'].dt.date

# Kelompokkan data berdasarkan hari dan hitung jumlah tweet per hari
daily_counts = df_wcu20.groupby(df_wcu20['date'].dt.date).count()

# Buat LineChartnya
plt.figure(figsize=(8,6))
plt.plot(daily_counts.index, daily_counts.values)
plt.title('Jumlah Tweet per Harinya')
plt.xlabel('Tanggal')
plt.xticks(rotation=90)
plt.ylabel('Jumlah Tweet')
plt.show()

# Buat Scatter Plot
sns.scatterplot(x=df_wcu20['retweetCount'], y=df_wcu20['likeCount'])

sns.set(rc={'figure.figsize':(6,3)})
plt.title('Jumlah Retweet vs Jumlah Like')
plt.xlabel('Jumlah Retweet')
plt.ylabel('Jumlah Like')
plt.show()

!pip install textblob

from textblob import TextBlob

# Tentukan fungsi untuk melakukan analisis sentimen pada tweet menggunakan TextBlob
def analisis_sentimen(tweet):
    blob = TextBlob(tweet)
    
    # Gunakan TextBlob untuk menghitung polaritas sentimen tweet
    polarity = blob.sentiment.polarity

    return polarity

# Terapkan fungsi analisis sentimen ke setiap tweet di DataFrame
df_wcu20['sentiment'] = df_wcu20['processed_text'].apply(analisis_sentimen)
print(df_wcu20.head())

def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positif'
    elif polarity < 0:
        return 'Negatif'
    else:
        return 'Netral'

# Apply the classify_sentiment function to each sentiment polarity value in the DataFrame
df_wcu20['sentiment_type'] = df_wcu20['sentiment'].apply(classify_sentiment)

sentiment_counts = df_wcu20['sentiment_type'].value_counts()

sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
sns.set(rc={'figure.figsize':(6,4)})
plt.title('Hasil Analisis Sentimen')
plt.xlabel('Jenis Sentimen')
plt.ylabel('Jumlah Tweet')

plt.show()

"""Word Cloud"""

# display only the tweet and sentiment score columns
positive_tweets = df_wcu20[df_wcu20['sentiment'] > 0]
positive_tweets.head(10)

negative_tweets = df_wcu20[df_wcu20['sentiment'] < 0]
negative_tweets.head(10)

from wordcloud import WordCloud, STOPWORDS
from PIL import Image

# concatenate all the very positive tweets into a single string
all_tweets = ' '.join(positive_tweets['processed_text'])

# generate the word cloud
wordcloud = WordCloud(width=1000, height=1000, background_color='white', colormap='Blues').generate(all_tweets)

# plot the word cloud
plt.figure(figsize=(10, 10), facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# concatenate all the very positive tweets into a single string
all_tweets=" "
all_tweets = ' '.join(negative_tweets['processed_text'])

# generate the word cloud
wordcloud = WordCloud(width=1000, height=1000, background_color='white', colormap='Blues').generate(all_tweets)

# plot the word cloud
plt.figure(figsize=(10, 10), facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

"""Cek Akurasi"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df_wcu20_2 = df_wcu20[['processed_text', 'sentiment_type']]

X_train, X_test, y_train, y_test = train_test_split(df_wcu20_2['processed_text'], df_wcu20_2['sentiment_type'], random_state=0)

# Convert text into numerical vectors using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train with SVM classifier
clf = SVC()
clf.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_vec)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)