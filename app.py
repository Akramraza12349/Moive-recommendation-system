import streamlit as st
import pandas as pd
import pickle
import requests


def fetch_poster(movie_id):
    response = requests.get(
        'https://api.themoviedb.org/3/movie/{}?api_key=c40dcf83395bc4f6a36c969475c44de5'.format(movie_id))
    data = response.json()

    return "https://image.tmdb.org/t/p/w500/"+data["poster_path"]


def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True, key=lambda x: x[1])[1:6]
    recommended_movie = []
    recomended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id

        # fetch poster
        recommended_movie.append(movies.iloc[i[0]].title)
        recomended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movie, recomended_movies_posters


similarity = pickle.load(open("similarity.pkl", "rb"))
movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)
movies_list = movies["title"].values
st.title("Movie recommender system")
selected_movie_name = st.selectbox("Movies", movies_list)

if st.button("Recommend"):
    recommendations, poster = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommendations[0])
        st.image(poster[0])
    with col2:
        st.text(recommendations[1])
        st.image(poster[1])
    with col3:
        st.text(recommendations[2])
        st.image(poster[2])
    with col4:
        st.text(recommendations[3])
        st.image(poster[3])
    with col5:
        st.text(recommendations[4])
        st.image(poster[4])
