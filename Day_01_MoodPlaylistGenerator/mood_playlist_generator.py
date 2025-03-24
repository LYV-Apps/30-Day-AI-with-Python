from nltk.sentiment.vader import SentimentIntensityAnalyzer
from thirty_days_pyai_helpers.print import slow_print, slow_print_header, print_intro

# Initialize NLTK VADER SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Predefined mood-to-playlist mapping (Spotify public playlist URLs)
mood_playlists = {
    "positive": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",  # Happy Hits! playlist
    "negative": "https://open.spotify.com/playlist/37i9dQZF1DX3YSRoSdA634",  # Life Sucks playlist
    "neutral": "https://open.spotify.com/playlist/37i9dQZF1E8PoczrwEPaxO"   # Whatever playlist
}

def analyze_mood(text):
    """Analyze the sentiment of input text."""
    scores = sid.polarity_scores(text)
    compound_score = scores["compound"]

    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

def get_playlist(mood):
    """Get a Spotify playlist based on mood."""
    return mood_playlists.get(mood, "No playlist found for this mood.")

def main():
    print_intro(1, "Mood Playlist Generator", "Feel the beat!")

    slow_print_header("❔ How are you feeling today? 🤔")

    # Get user input
    user_input = input(f"""
    (example: Today I'm feeling happy) 
    """)

    # Analyze mood
    mood = analyze_mood(user_input)
    slow_print(" ")
    slow_print("****************************")
    slow_print(" ")
    slow_print("🤖💬")     
    slow_print(f" We detect that {user_input} is a {mood} emotion.")

    # Suggest Playlist
    playlist_url = get_playlist(mood)
    slow_print(" ")
    slow_print(f" Open the link in your browser to listen to your playlist: ")
    slow_print(f"🌐 {playlist_url}")
    slow_print(" ")
    slow_print("🦾🤖💬")
    slow_print(f"Have a great day!")
    slow_print(" ")
    slow_print("****************************")

if __name__ == "__main__":
    main()