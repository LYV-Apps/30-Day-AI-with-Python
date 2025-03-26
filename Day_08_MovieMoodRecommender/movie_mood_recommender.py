import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.metrics.pairwise import cosine_similarity
import requests
import zipfile
import io
import time

def slow_print(text, delay=0.1):
    print(text)
    time.sleep(delay)

# Mood to genre mapping
MOOD_TO_GENRES = {
    "happy": ["Comedy"],
    "sad": ["Drama"],
    "excited": ["Action", "Adventure", "Thriller"],
    "relaxed": ["Romance", "Family"]
}

# Sample movies for user to rate (movieId and title from MovieLens ml-latest-small)
SAMPLE_MOVIES = {
    1: "Toy Story (1995)",
    356: "Forrest Gump (1994)",
    318: "Shawshank Redemption, The (1994)",
    296: "Pulp Fiction (1994)",
    593: "Silence of the Lambs, The (1991)"
}

# Function to download and extract the MovieLens dataset
def download_movielens_data():
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    response = requests.get(url)
    response.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall('movielens_data')
    
    ratings_path = 'movielens_data/ml-latest-small/ratings.csv'
    movies_path = 'movielens_data/ml-latest-small/movies.csv'
    
    return ratings_path, movies_path

# Function to get user ratings
def get_user_ratings():
    slow_print(f"\n{Fore.BLUE}Rate the following movies from 1 to 5 (or 0 if you haven’t seen it)[5 being the best]:")
    user_ratings = {}
    for movie_id, title in SAMPLE_MOVIES.items():
        while True:
            rating = input(f"{Fore.MAGENTA}{title}: ")
            try:
                rating = float(rating)
                if rating < 0 or rating > 5:
                    slow_print(f"{Fore.RED}Please enter a number between 0 and 5.")
                else:
                    user_ratings[movie_id] = rating
                    break
            except ValueError:
                slow_print(f"{Fore.RED}Please enter a valid number (0-5).")
    return user_ratings

# Function to find similar users
def get_similar_users(user_ratings, user_item_matrix, n=5):
    """Find similar users based on cosine similarity"""
    # Create a vector for the new user's ratings, aligned with the matrix columns
    user_vector = np.zeros(len(user_item_matrix.columns))
    for movie_id, rating in user_ratings.items():
        if movie_id in user_item_matrix.columns:
            user_vector[user_item_matrix.columns.get_loc(movie_id)] = rating
    
    # Reshape for cosine similarity
    user_vector = user_vector.reshape(1, -1)
    
    # Calculate similarity with all users
    similarity_scores = cosine_similarity(user_item_matrix.values, user_vector)
    
    # Create DataFrame with user IDs and similarity scores
    similarity_df = pd.DataFrame({
        'userId': user_item_matrix.index,
        'similarity': similarity_scores.flatten()
    })
    
    # Sort by similarity and get top n
    similar_users = similarity_df.sort_values('similarity', ascending=False).head(n)
    return similar_users

# Function to recommend movies
def recommend_movies(user_ratings, user_item_matrix, movies_df, mood, n_similar_users=5, n_recommendations=10):
    """Recommend movies based on similar users' ratings and mood"""
    similar_users = get_similar_users(user_ratings, user_item_matrix, n_similar_users)
    
    # Movies the user hasn’t rated (exclude ones they rated > 0)
    rated_movies = [movie_id for movie_id, rating in user_ratings.items() if rating > 0]
    unrated_movies = [col for col in user_item_matrix.columns if col not in rated_movies]
    
    recommendations = {}
    for _, row in similar_users.iterrows():
        similar_user = row['userId']
        similarity = row['similarity']
        similar_user_ratings = user_item_matrix.loc[similar_user]
        
        for movie in unrated_movies:
            if movie in similar_user_ratings.index and similar_user_ratings[movie] > 0:
                if movie not in recommendations:
                    recommendations[movie] = 0
                recommendations[movie] += similar_user_ratings[movie] * similarity
    
    # Convert to DataFrame and sort
    recommendations_df = pd.DataFrame({
        'movieId': list(recommendations.keys()),
        'weighted_rating': list(recommendations.values())
    }).sort_values('weighted_rating', ascending=False)
    
    # Merge with movie information
    recommendations_df = pd.merge(recommendations_df, movies_df, on='movieId')
    
    # Filter by mood-related genres
    mood_genres = MOOD_TO_GENRES.get(mood.lower(), [])
    if mood_genres:
        recommendations_df = recommendations_df[
            recommendations_df['genres'].apply(
                lambda x: any(genre in x for genre in mood_genres)
            )
        ]
    
    return recommendations_df[['title', 'genres', 'weighted_rating']].head(n_recommendations)

# Main function
def run_recommender():
    # Load the data
    ratings_path, movies_path = download_movielens_data()
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    # Create user-item matrix
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    
    # Get user ratings
    user_ratings = get_user_ratings()
    
    # Get user mood
    slow_print(f"{Fore.MAGENTA}\nHow are you feeling? Choose a mood:")
    slow_print(f"{Fore.BLUE}1. Happy")
    slow_print(f"{Fore.YELLOW}2. Sad")
    slow_print(f"{Fore.GREEN}3. Excited")
    slow_print(f"{Fore.CYAN}4. Relaxed")
    mood_choice = input(f"\n{Fore.LIGHTBLUE_EX}Enter the number of your mood (1-4): ")
    mood_map = {"1": "happy", "2": "sad", "3": "excited", "4": "relaxed"}
    mood = mood_map.get(mood_choice, "happy")  # Default to "happy" if invalid
    
    # Generate recommendations
    recommendations = recommend_movies(user_ratings, user_item_matrix, movies, mood)
    
    # Display recommendations
    slow_print(f"\n{Fore.GREEN}Top movie recommendations for you feeling {mood}:")
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        slow_print(f"{Fore.BLACK}{i}. {row['title']} ({row['genres']}) - Score: {row['weighted_rating']:.2f}")

# Run the recommender
if __name__ == "__main__":
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
    slow_print(f"{Fore.GREEN}*    🤖  30 DAY AI  🤖    *{Style.RESET_ALL}")
    slow_print(f"{Fore.YELLOW}*    WELCOME TO DAY 8!    *{Style.RESET_ALL}")
    slow_print(f"{Fore.RED}*  MOVIE MOOD RECOMMENDER  *{Style.RESET_ALL}")
    slow_print(f"{Fore.BLUE}*           🎬✅          *{Style.RESET_ALL}")
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")

    run_recommender()