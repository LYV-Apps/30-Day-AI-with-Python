from colorama import Fore, Style
import nltk
import os
import time
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pytwitter import Api
from textblob import TextBlob
from typing import Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables from .env file
load_dotenv()

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize lemmatizer for light preprocessing
lemmatizer = WordNetLemmatizer()

# Initialize PyTwitter API
bearer_token = os.getenv("X_BEARER_TOKEN")
api = Api(bearer_token=os.getenv("X_BEARER_TOKEN"))

def slow_print(text, delay=0.1):
    print(text)
    time.sleep(delay)

def preprocess_tweet(tweet):
    tokens = word_tokenize(tweet.lower())
    lemmatized = [
        lemmatizer.lemmatize(token)
        for token in tokens 
        if  token.isalnum() or token in ["n't", "not", "wouldn't"]
        ]
    return ' '.join(lemmatized)

def analyze_sentiment_vader(text: str) -> Dict[str, float]:
    sentiment = analyzer.polarity_scores(text)
    return sentiment

def classify_sentiment(compound_score: float) -> str:
    if compound_score <= -0.05:
        return "Negative"
    elif -0.05 < compound_score < 0.05:
        return "Neutral"
    else:
        return "Positive"

def extract_text_from_x_post(url):
    if not bearer_token:
        slow_print(f"\n{Fore.RED}Error: X_BEARER_TOKEN environment variable is not set.{Style.RESET_ALL}")
        return None

    try:
        slow_print(f"\n{Fore.LIGHTGREEN_EX}Fetching tweet...{Style.RESET_ALL}")
        tweet = api.get_tweet(url.split("/")[-1])

        if tweet.data:
            tweet_text = tweet.data.text
            slow_print(f"\n{Fore.BLUE}Tweet Text: {tweet_text}")
            # Analyze sentiment on original text
            original_sentiment = analyze_sentiment_vader(tweet_text)
            original_classification = classify_sentiment(original_sentiment['compound'])

            # Lightly preprocess the text for additional analysis
            preprocessed_text = preprocess_tweet(tweet_text)
            slow_print(f"\n{Fore.CYAN}Preprocessed Text: {preprocessed_text}")

            # Analyze sentiment on preprocessed text
            preprocessed_sentiment = analyze_sentiment_vader(preprocessed_text)
            preprocessed_classification = classify_sentiment(preprocessed_sentiment['compound'])

            return tweet_text, original_sentiment, preprocessed_sentiment, original_classification, preprocessed_classification
        else: 
            slow_print(f"\n{Fore.RED}Error: Tweet not found.{Style.RESET_ALL}")
            return None
    
    except Exception as e:
        slow_print(f"\n{Fore.RED}Error extracting post: {e}{Style.RESET_ALL}")
        return None
    
def get_textblob_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'


if __name__ == "__main__":
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
    slow_print(f"{Fore.GREEN}*    🤖  30 DAY AI  🤖    *{Style.RESET_ALL}")
    slow_print(f"{Fore.YELLOW}*    WELCOME TO DAY 9!    *{Style.RESET_ALL}")
    slow_print(f"{Fore.RED}*   X Sentiment Analyzer  *{Style.RESET_ALL}")
    slow_print(f"{Fore.BLUE}*           🆇〠          *{Style.RESET_ALL}")
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
    
    # Test with an X post URL
    slow_print(f"\n{Fore.GREEN}Enter an X.com post URL to analyze (e.g., https://x.com/username/status/123456789):")
    url = input(f"{Fore.BLACK}URL: ")
   
    result = extract_text_from_x_post(url)
    if result:
        tweet_text, original_sentiment, preprocessed_sentiment, original_classification, preprocessed_classification = result
        slow_print(f"\n{Fore.GREEN}Sentiment Analysis Results:")
        slow_print(f"\n{Fore.BLUE}Original Text Sentiment: {original_sentiment}")
        slow_print(f"Original Text Classification: {original_classification}")
        slow_print(f"\n{Fore.YELLOW}Preprocessed Text Sentiment: {preprocessed_sentiment}")
        slow_print(f"Preprocessed Text Classification: {preprocessed_classification}")
    else:
        slow_print(f"\n{Fore.RED}Failed to retrieve tweet or analyze sentiment.")