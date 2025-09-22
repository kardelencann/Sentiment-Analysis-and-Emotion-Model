from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV,cross_val_predict
from sklearn.preprocessing import LabelEncoder
from textblob import Word ,TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: '%2f' % x)

df = pd.read_csv("amazon_reviews.csv")
df.head()

df["reviewText"] = df["reviewText"].str.lower()
df["reviewText"] = df["reviewText"].str.replace(r"[^\w\s]","", regex=True) #noktalama işaretlerini kaldırır.
df["reviewText"] = df["reviewText"].str.replace(r"\d", " ", regex=True) #sayıları kaldırır.

import nltk
#nltk.download('stopwords')

sw = stopwords.words('english')
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


temp_df = pd.Series(' '.join(df["reviewText"]).split()).value_counts()
drop = temp_df[temp_df <=1]
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in drop))

df["reviewText"].apply(lambda x: TextBlob(x).words).head()