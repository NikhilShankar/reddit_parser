import praw
import mysql.connector
from mysql.connector import Error
import datetime
import time
import os
from dotenv import load_dotenv
import logging
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RedditScraper:
    def __init__(self, subreddit_name: str = "DecidingToBeBetter"):
        """Initialize Reddit scraper with database and API connections."""
        self.reddit = None
        self.db_connection = None
        self.subreddit_name = subreddit_name
        
    def setup_reddit_connection(self):
        """Set up Reddit API connection using PRAW."""
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'MindfulnessScaper/1.0')
            )
            logger.info("Reddit API connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {e}")
            return False
    
    def setup_database_connection(self):
        """Set up MySQL database connection."""
        try:
            self.db_connection = mysql.connector.connect(
                host='localhost',
                port=3306,
                database='reddit_mindfulness',
                user='root',
                password='admin123',
                charset='utf8mb4',
                use_unicode=True
            )
            logger.info("Database connection established successfully")
            return True
        except Error as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def close_connections(self):
        """Close database connection."""
        if self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
            logger.info("Database connection closed")
    
    def safe_get_attribute(self, obj, attr: str, default=None):
        """Safely get attribute from Reddit object, handling deleted/removed content."""
        try:
            value = getattr(obj, attr, default)
            # Handle cases where author is deleted
            if attr == 'author' and value:
                return str(value) if str(value) != '[deleted]' else None
            return value
        except Exception:
            return default
    
    def convert_utc_timestamp(self, utc_timestamp: float) -> datetime.datetime:
        """Convert UTC timestamp to datetime object."""
        return datetime.datetime.fromtimestamp(utc_timestamp)
    
    def insert_post(self, post) -> bool:
        """Insert a post into the database."""
        try:
            cursor = self.db_connection.cursor()
            
            # Prepare post data
            post_data = {
                'id': post.id,
                'title': post.title[:1000] if post.title else None,  # Limit title length
                'author': self.safe_get_attribute(post, 'author'),
                'content': post.selftext if hasattr(post, 'selftext') else None,
                'url': post.url,
                'score': self.safe_get_attribute(post, 'score', 0),
                'upvote_ratio': self.safe_get_attribute(post, 'upvote_ratio'),
                'num_comments': self.safe_get_attribute(post, 'num_comments', 0),
                'created_utc': self.convert_utc_timestamp(post.created_utc),
                'subreddit': str(post.subreddit),
                'is_self': self.safe_get_attribute(post, 'is_self', False),
                'selftext': post.selftext if hasattr(post, 'selftext') else None,
                'permalink': self.safe_get_attribute(post, 'permalink')
            }
            
            insert_query = """
            INSERT IGNORE INTO posts 
            (id, title, author, content, url, score, upvote_ratio, num_comments, 
             created_utc, subreddit, is_self, selftext, permalink)
            VALUES (%(id)s, %(title)s, %(author)s, %(content)s, %(url)s, %(score)s, 
                    %(upvote_ratio)s, %(num_comments)s, %(created_utc)s, %(subreddit)s, 
                    %(is_self)s, %(selftext)s, %(permalink)s)
            """
            
            cursor.execute(insert_query, post_data)
            self.db_connection.commit()
            cursor.close()
            
            logger.info(f"Inserted post: {post.id} - {post.title[:50]}...")
            return True
            
        except Error as e:
            logger.error(f"Error inserting post {post.id}: {e}")
            return False
    
    def insert_comment(self, comment, post_id: str) -> bool:
        """Insert a comment into the database."""
        try:
            cursor = self.db_connection.cursor()
            
            # Skip deleted or removed comments
            if not hasattr(comment, 'body') or comment.body in ['[deleted]', '[removed]']:
                return False
            
            # Determine parent type and ID
            parent_type = 'post'
            parent_id = post_id
            
            if hasattr(comment, 'parent_id') and comment.parent_id:
                parent_id_str = str(comment.parent_id)
                if parent_id_str.startswith('t1_'):  # Comment parent
                    parent_type = 'comment'
                    parent_id = parent_id_str[3:]  # Remove 't1_' prefix
                elif parent_id_str.startswith('t3_'):  # Post parent
                    parent_type = 'post'
                    parent_id = parent_id_str[3:]  # Remove 't3_' prefix
            
            comment_data = {
                'id': comment.id,
                'post_id': post_id,
                'author': self.safe_get_attribute(comment, 'author'),
                'body': comment.body,
                'score': self.safe_get_attribute(comment, 'score', 0),
                'created_utc': self.convert_utc_timestamp(comment.created_utc),
                'parent_type': parent_type,
                'parent_id': parent_id,
                'permalink': self.safe_get_attribute(comment, 'permalink')
            }
            
            insert_query = """
            INSERT IGNORE INTO comments 
            (id, post_id, author, body, score, created_utc, parent_type, parent_id, permalink)
            VALUES (%(id)s, %(post_id)s, %(author)s, %(body)s, %(score)s, 
                    %(created_utc)s, %(parent_type)s, %(parent_id)s, %(permalink)s)
            """
            
            cursor.execute(insert_query, comment_data)
            self.db_connection.commit()
            cursor.close()
            
            logger.debug(f"Inserted comment: {comment.id}")
            return True
            
        except Error as e:
            logger.error(f"Error inserting comment {comment.id}: {e}")
            return False
    
    def process_comment_tree(self, comment_forest, post_id: str):
        """Recursively process all comments in a comment tree."""
        comment_count = 0
        
        for comment in comment_forest:
            # Skip MoreComments objects
            if isinstance(comment, praw.models.MoreComments):
                continue
                
            # Insert the comment
            if self.insert_comment(comment, post_id):
                comment_count += 1
            
            # Process replies recursively
            if hasattr(comment, 'replies') and comment.replies:
                comment_count += self.process_comment_tree(comment.replies, post_id)
        
        return comment_count
    
    def scrape_subreddit(self, limit: int = 1000, sort_method: str = 'hot'):
        """
        Scrape posts and comments from the mindfulness subreddit.
        
        Args:
            limit: Number of posts to scrape
            sort_method: 'hot', 'new', 'top', 'rising'
        """
        if not self.reddit or not self.db_connection:
            logger.error("Reddit API or database connection not established")
            return
        
        try:
            subreddit = self.reddit.subreddit(self.subreddit_name)
            logger.info(f"Starting to scrape r/{self.subreddit_name} - {sort_method} posts (limit: {limit})")
            
            # Get posts based on sort method
            if sort_method == 'hot':
                posts = subreddit.hot(limit=limit)
            elif sort_method == 'new':
                posts = subreddit.new(limit=limit)
            elif sort_method == 'top':
                posts = subreddit.top(limit=limit)
            elif sort_method == 'rising':
                posts = subreddit.rising(limit=limit)
            else:
                posts = subreddit.hot(limit=limit)
            
            total_posts = 0
            total_comments = 0
            
            for post in posts:
                try:
                    # Insert post
                    if self.insert_post(post):
                        total_posts += 1
                        
                        # Get all comments for this post
                        post.comments.replace_more(limit=None)  # Load all comments
                        comment_count = self.process_comment_tree(post.comments, post.id)
                        total_comments += comment_count
                        
                        logger.info(f"Post {post.id}: {comment_count} comments processed")
                        
                        # Small delay to be respectful to Reddit's servers
                        time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing post {post.id}: {e}")
                    continue
            
            logger.info(f"Scraping completed! Total posts: {total_posts}, Total comments: {total_comments}")
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
    
    def run(self, limit: int = 100, sort_method: str = 'hot'):
        """Main method to run the scraper."""
        logger.info("Starting Reddit scraper for r/mindfulness")
        
        # Setup connections
        if not self.setup_reddit_connection():
            return
        
        if not self.setup_database_connection():
            return
        
        try:
            # Run the scraping
            self.scrape_subreddit(limit=limit, sort_method=sort_method)
        finally:
            # Clean up connections
            self.close_connections()
            logger.info("Scraper finished")

def main():
    """Main function to run the scraper."""
    # Configuration
    POST_LIMIT = 100000  # Number of posts to scrape
    SORT_METHOD = 'hot'  # 'hot', 'new', 'top', 'rising'
    SUBREDDIT_NAME = 'mindfulness'
    scraper = RedditScraper(SUBREDDIT_NAME)
    scraper.run(limit=POST_LIMIT, sort_method=SORT_METHOD)

if __name__ == "__main__":
    main()