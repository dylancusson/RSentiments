import praw
import time
import os
import csv
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER sentiment lexicon
nltk.download("vader_lexicon")

# Load environment variables
load_dotenv()

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Reddit API Authentication
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
)

# CSV file setup
CSV_FILE = "reddit_comments_sentiment.csv"

# Ensure CSV file has headers
def setup_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["scraped_at", "subreddit", "post_title", "post_id", "comment_id", "comment_text", "sentiment_score", "sentiment_label"])

# Function to fetch ONLY top-level comments with exponential backoff
def fetch_top_level_comments_with_backoff(post, max_comments=20):
    comments = []
    backoff = 1  # Initial wait time in seconds

    while True:
        try:
            post.comments.replace_more(limit=None)  # Expand all top-level comments
            top_level_comments = [comment for comment in post.comments if isinstance(comment, praw.models.Comment)]
            return top_level_comments[:max_comments]  # Return only top 20 comments
        except praw.exceptions.RedditAPIException:
            print(f"Rate limit hit! Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)  # Exponential backoff, max 60 sec
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

# Function to perform sentiment analysis
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)["compound"]  # Get sentiment score (-1 to 1)
    if sentiment_score >= 0.05:
        sentiment_label = "positive"
    elif sentiment_score <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    
    return sentiment_score, sentiment_label

# Function to save comments & sentiment analysis to CSV
def save_to_csv(subreddit, post, comments):
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp
        for comment in comments:
            sentiment_score, sentiment_label = analyze_sentiment(comment.body)
            writer.writerow([scraped_at, subreddit, post.title, post.id, comment.id, comment.body, sentiment_score, sentiment_label])

# Main function to fetch top posts and comments from multiple subreddits
def fetch_top_posts(subreddits, post_limit=10):
    for subreddit_name in subreddits:
        print(f"\n📌 Scraping subreddit: r/{subreddit_name}")
        subreddit = reddit.subreddit(subreddit_name)

        for post in subreddit.top(time_filter="day", limit=post_limit):
            print(f"🔍 Fetching post: {post.title}")
            comments = fetch_top_level_comments_with_backoff(post)

            print(f"✅ Fetched {len(comments)} top-level comments for post: {post.title}")

            # Save to CSV
            save_to_csv(subreddit_name, post, comments)

            time.sleep(2)  # Small delay between post requests

# Setup CSV and run the script
setup_csv()
subreddits_to_scrape = ["politics", "conservative", "democrats", "50501", "libertarian", "walkaway", "worldnews"]
fetch_top_posts(subreddits=subreddits_to_scrape)

print(f"All done! Data saved to {CSV_FILE}")

