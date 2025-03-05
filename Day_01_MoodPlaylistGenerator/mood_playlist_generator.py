import nltk
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

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

def slow_print(text, delay=0.05):
    print(text)
    time.sleep(delay)

def main():
    # Arcade-style welcome message with color and emojis
    slow_print(f"{Fore.CYAN}{'*' * 28}{Style.RESET_ALL}")
    slow_print(f"{Fore.GREEN}*    🤖  30 DAY AI  🤖    *{Style.RESET_ALL}")
    slow_print(f"{Fore.YELLOW}*   WELCOME TO DAY 1!      *{Style.RESET_ALL}")
    slow_print(f"{Fore.RED}* MOOD PLAYLIST GENERATOR  *{Style.RESET_ALL}")
    slow_print(f"{Fore.BLUE}*  🎶  FEEL THE BEAT!  🎶  *{Style.RESET_ALL}")
    slow_print(f"{Fore.CYAN}{'*' * 28}{Style.RESET_ALL}")

    # Get user input
    user_input = input(f"""
    {Fore.YELLOW}❔ How are you feeling today? {Style.RESET_ALL}
    {Fore.MAGENTA}(example: Today I'm feeling happy) {Style.RESET_ALL}
    """)

    # Analyze mood
    mood = analyze_mood(user_input)
    slow_print(" ")
    slow_print("****************************")
    slow_print(" ")
    slow_print("🤖💬")     
    slow_print(f"{Fore.GREEN} We detect that {user_input} is a {mood} emotion.{Style.RESET_ALL}")

    # Suggest Playlist
    playlist_url = get_playlist(mood)
    slow_print(" ")
    slow_print(f"{Fore.CYAN} Open the link in your browser to listen to your playlist: {Style.RESET_ALL}")
    slow_print(f"🌐 {playlist_url}")
    slow_print(" ")
    slow_print("🦾🤖💬")
    slow_print(f"Have a great day!")
    slow_print(" ")
    slow_print("****************************")

if __name__ == "__main__":
    main()