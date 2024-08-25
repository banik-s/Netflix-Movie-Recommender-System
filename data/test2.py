import numpy as np
import pandas as pd
import pickle
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the NLP model and TFIDF vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('transform.pkl', 'rb'))

# Function to create a similarity matrix
def create_similarity():
    data = pd.read_csv('main_data.csv')
    # Creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # Creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data, similarity

# Function to recommend movies based on a title
def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].str.lower().unique():
        return 'Sorry! Try another movie name'
    else:
        idx = data[data['movie_title'].str.lower() == m].index[0]
        lst = list(enumerate(similarity[idx]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        
        # Exclude the input movie from the recommendations
        lst = [item for item in lst if data['movie_title'].iloc[item[0]].lower() != m]
        
        lst = lst[:10]  # Get the top 10 recommendations
        recommendations = []
        for i in range(len(lst)):
            a = lst[i][0]
            movie_title = data['movie_title'].iloc[a]
            recommendations.append(movie_title)
        return recommendations

# Function to fetch movie details and reviews
def fetch_movie_details(movie_title):
    api_key = 'c2cde6c4e3f448b03d1ad15d56baecb8'  # Replace with your TMDb API key
    search_response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}")
    search_data = search_response.json()

    if 'results' in search_data and len(search_data['results']) > 0:
        movie_id = search_data['results'][0]['id']
        details_response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&append_to_response=credits")
        movie_data = details_response.json()

        reviews_response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={api_key}")
        reviews_data = reviews_response.json()

        return movie_data, reviews_data
    else:
        return None, None

# Function to fetch recommended movie details including titles and posters
def fetch_recommendation_details(recommendations):
    api_key = 'c2cde6c4e3f448b03d1ad15d56baecb8'  # Replace with your TMDb API key
    movie_details = []
    for movie_title in recommendations:
        search_response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}")
        search_data = search_response.json()
        if 'results' in search_data and len(search_data['results']) > 0:
            movie_id = search_data['results'][0]['id']
            details_response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}")
            movie_data = details_response.json()
            movie_details.append((movie_data['title'], movie_data['poster_path']))
    return movie_details

# Streamlit application
st.title("Movie Recommendation System")

# Inject HTML and CSS for full-page background
st.markdown("""
    <style>
    .full-page-background {
        background: url('https://pbs.twimg.com/media/GR2vyoTWgAAEpBY?format=jpg&name=4096x4096') no-repeat center center fixed;
        background-size: cover;
        height: 100vh;
        padding: 20px;
    }
    </style>
    <div class="full-page-background">
        
    </div>
    """, unsafe_allow_html=True)

# Initialize session state for movie selection
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

# Input movie title if no movie is selected
if st.session_state.selected_movie is None:
    input_movie_title = st.text_input("Enter a movie title to get recommendations:")
    if input_movie_title:
        st.session_state.selected_movie = input_movie_title.lower()

if st.session_state.selected_movie:
    # Fetch and display movie details and reviews for the selected movie
    movie_data, reviews_data = fetch_movie_details(st.session_state.selected_movie)

    if movie_data:
        st.subheader(f"Details for {st.session_state.selected_movie.title()}")
        st.image(f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}", width=300)
        st.write(f"**Title:** {movie_data['title']}")
        st.write(f"**Overview:** {movie_data['overview']}")
        st.write(f"**Release Date:** {movie_data['release_date']}")
        st.write(f"**Genres:** {[genre['name'] for genre in movie_data['genres']]}")
        st.write(f"**Rating:** {movie_data['vote_average']} ({movie_data['vote_count']} votes)")

        st.write("**Director(s):**")
        directors = [crew_member for crew_member in movie_data['credits']['crew'] if crew_member['job'] == 'Director']
        director_cols = st.columns(len(directors))
        for idx, director in enumerate(directors):
            with director_cols[idx]:
                st.image(f"https://image.tmdb.org/t/p/w500{director['profile_path']}", width=100)
                st.write(f"{director['name']}")

        st.write("**Cast:**")
        cast_members = movie_data['credits']['cast'][:5]  # Limit to top 5 cast members
        cast_cols = st.columns(len(cast_members))
        for idx, cast in enumerate(cast_cols):
            with cast:
                st.image(f"https://image.tmdb.org/t/p/w500{cast_members[idx]['profile_path']}", width=100)
                st.write(f"{cast_members[idx]['name']} as {cast_members[idx]['character']}")

        if reviews_data:
            st.subheader("Reviews")
            reviews = reviews_data.get('results', [])[:5]  # Fetch up to 5 reviews
            reviews_table = pd.DataFrame([{
                "Author": review['author'],
                "Content": review['content'][:500] + '...',
                "Rating": 'Good' if clf.predict(vectorizer.transform([review['content']])) else 'Bad'
            } for review in reviews])

            # Set index to start from 1
            reviews_table.index = np.arange(1, len(reviews_table) + 1)
            st.table(reviews_table)


    else:
        st.error("No details found for the movie. Please try a different title.")

    # Get movie recommendations
    recommendations = rcmd(st.session_state.selected_movie)
    if isinstance(recommendations, str):
        st.error(recommendations)
        st.session_state.selected_movie = None  # Reset to allow new input
    else:
        st.write(f"Top 10 movie recommendations based on '{st.session_state.selected_movie}':")
        movie_details = fetch_recommendation_details(recommendations)

        # Display recommendations with thumbnails as clickable images in two columns
        col1, col2 = st.columns(2)
        for i, (movie_title, poster_path) in enumerate(movie_details):
            with (col1 if i % 2 == 0 else col2):
                if st.button(movie_title):
                    st.session_state.selected_movie = movie_title.lower()
                    st.experimental_rerun()  # Immediately rerun the app with the new selected movie
                st.image(f"https://image.tmdb.org/t/p/w500{poster_path}", width=250)

# Reset button to start over
if st.button("Reset"):
    st.session_state.selected_movie = None
