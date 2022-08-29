import os
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.warn("ignore")
# data_dir = os.path.join(os.path.dirname(__file__), "Kaggle\data")

movies_df = pd.read_csv(
    r"C:\Users\MPM\Documents\Movie_recommendation\Kaggle\data\tmdb_5000_movies.csv")

credit_df = pd.read_csv(
    r"C:\Users\MPM\Documents\Movie_recommendation\Kaggle\data\tmdb_5000_credits.csv")

merged_df = pd.merge(movies_df, credit_df, on='title')
# required columns
# genere,id,keywords,title,overview,cast,crew

merged_df = merged_df[["movie_id", "title",
                       "overview", "genres", "cast", "crew", "keywords"]]

merged_df.dropna(inplace=True)


def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


merged_df["genres"] = merged_df["genres"].apply(convert)
merged_df["keywords"] = merged_df["keywords"].apply(convert)


def convert_2(obj):
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            l.append(i['name'])
        else:
            break
    return l


merged_df["cast"] = merged_df["cast"].apply(convert_2)


def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            l.append(i['name'])
    return l


merged_df["crew"] = merged_df["crew"].apply(fetch_director)
merged_df["overview"] = merged_df["overview"].apply(lambda x: x.split())
merged_df["genres"] = merged_df["genres"].apply(
    lambda x: [i.replace(" ", "") for i in x])

merged_df["keywords"] = merged_df["keywords"].apply(
    lambda x: [i.replace(" ", "") for i in x])

merged_df["cast"] = merged_df["cast"].apply(
    lambda x: [i.replace(" ", "") for i in x])
merged_df["crew"] = merged_df["crew"].apply(
    lambda x: [i.replace(" ", "") for i in x])

merged_df["tags"] = merged_df["overview"]+merged_df["genres"] + \
    merged_df["keywords"]+merged_df["cast"]+merged_df["crew"]

new_df = merged_df[["movie_id", "title", "tags"]]
new_df["tags"] = merged_df["tags"].apply(lambda x: " ".join(x))
new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())


# stemming , base words
ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


new_df["tags"] = new_df["tags"].apply(stem)
# bag of words


cv = CountVectorizer(max_features=5000, stop_words="english")
vector = cv.fit_transform(new_df["tags"]).toarray()
# cosine distance
similarity = cosine_similarity(vector)

# index fetrching


def recommend(movie):
    movie_index = new_df[new_df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


pickle.dump(new_df.to_dict(), open("movies_dict.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))
