import mysql.connector
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditDataExtractor:
    def __init__(self, db_config: Dict):
        """Initialize with database configuration."""
        self.db_config = db_config
        self.connection = None
        
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def analyze_corpus(self) -> Dict:
        """Analyze the corpus to understand data distribution."""
        if not self.connection:
            self.connect()
            
        cursor = self.connection.cursor()
        
        analysis = {}
        
        # Basic statistics
        cursor.execute("SELECT COUNT(*) FROM posts")
        analysis['total_posts'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM comments")
        analysis['total_comments'] = cursor.fetchone()[0]
        
        # Content length analysis
        cursor.execute("""
            SELECT 
                AVG(CHAR_LENGTH(COALESCE(title, '') + COALESCE(selftext, ''))) as avg_post_length,
                MAX(CHAR_LENGTH(COALESCE(title, '') + COALESCE(selftext, ''))) as max_post_length,
                MIN(CHAR_LENGTH(COALESCE(title, '') + COALESCE(selftext, ''))) as min_post_length
            FROM posts 
            WHERE title IS NOT NULL
        """)
        post_stats = cursor.fetchone()
        analysis['post_length'] = {
            'avg': post_stats[0],
            'max': post_stats[1], 
            'min': post_stats[2]
        }
        
        cursor.execute("""
            SELECT 
                AVG(CHAR_LENGTH(body)) as avg_comment_length,
                MAX(CHAR_LENGTH(body)) as max_comment_length,
                MIN(CHAR_LENGTH(body)) as min_comment_length
            FROM comments 
            WHERE body IS NOT NULL AND body != '[deleted]' AND body != '[removed]'
        """)
        comment_stats = cursor.fetchone()
        analysis['comment_length'] = {
            'avg': comment_stats[0],
            'max': comment_stats[1],
            'min': comment_stats[2]
        }
        
        # Comments per post distribution
        cursor.execute("""
            SELECT 
                p.id,
                p.num_comments as reported_comments,
                COUNT(c.id) as actual_comments
            FROM posts p
            LEFT JOIN comments c ON p.id = c.post_id
            GROUP BY p.id
            ORDER BY actual_comments DESC
            LIMIT 10
        """)
        top_commented_posts = cursor.fetchall()
        analysis['top_commented_posts'] = top_commented_posts
        
        # Score distribution
        cursor.execute("""
            SELECT 
                AVG(score) as avg_score,
                MAX(score) as max_score,
                MIN(score) as min_score,
                STDDEV(score) as std_score
            FROM posts
        """)
        post_score_stats = cursor.fetchone()
        analysis['post_scores'] = {
            'avg': post_score_stats[0],
            'max': post_score_stats[1],
            'min': post_score_stats[2],
            'std': post_score_stats[3]
        }
        
        cursor.execute("""
            SELECT 
                AVG(score) as avg_score,
                MAX(score) as max_score,
                MIN(score) as min_score,
                STDDEV(score) as std_score
            FROM comments
            WHERE body IS NOT NULL AND body != '[deleted]' AND body != '[removed]'
        """)
        comment_score_stats = cursor.fetchone()
        analysis['comment_scores'] = {
            'avg': comment_score_stats[0],
            'max': comment_score_stats[1],
            'min': comment_score_stats[2],
            'std': comment_score_stats[3]
        }
        
        cursor.close()
        return analysis
    
    def extract_posts_with_comments(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
        """Extract all posts with their associated comments."""
        if not self.connection:
            self.connect()
            
        cursor = self.connection.cursor(dictionary=True)
        
        # Get posts
        query = """
            SELECT 
                id, title, author, selftext, url, score, 
                upvote_ratio, num_comments, created_utc, permalink
            FROM posts 
            WHERE title IS NOT NULL
            ORDER BY score DESC, created_utc DESC
        """
        
        params = []
        if limit:
            query += f" LIMIT {limit}"
            if offset > 0:
                query += f" OFFSET {offset}"
                
        cursor.execute(query)
        posts = cursor.fetchall()
        
        logger.info(f"Extracted {len(posts)} posts")
        
        # Get comments for each post
        for i, post in enumerate(posts):
            if i % 50 == 0:  # Progress indicator
                logger.info(f"Processing post {i+1}/{len(posts)}")
                
            cursor.execute("""
                SELECT 
                    id, author, body, score, created_utc, 
                    parent_type, parent_id, permalink
                FROM comments 
                WHERE post_id = %s 
                AND body IS NOT NULL 
                AND body NOT IN ('[deleted]', '[removed]')
                ORDER BY score DESC, created_utc ASC
            """, (post['id'],))
            
            post['comments'] = cursor.fetchall()
        
        cursor.close()
        return posts
    
    def get_high_value_comments(self, min_score: int = 5, limit: Optional[int] = None) -> List[Dict]:
        """Extract high-value standalone comments."""
        if not self.connection:
            self.connect()
            
        cursor = self.connection.cursor(dictionary=True)
        
        query = """
            SELECT 
                c.id, c.author, c.body, c.score, c.created_utc, 
                c.parent_type, c.parent_id, c.permalink, c.post_id,
                p.title as post_title, p.author as post_author
            FROM comments c
            JOIN posts p ON c.post_id = p.id
            WHERE c.body IS NOT NULL 
            AND c.body NOT IN ('[deleted]', '[removed]')
            AND c.score >= %s
            ORDER BY c.score DESC, c.created_utc DESC
        """
        
        params = [min_score]
        if limit:
            query += " LIMIT %s"
            params.append(limit)
            
        cursor.execute(query, params)
        comments = cursor.fetchall()
        
        cursor.close()
        return comments

class HierarchicalChunker:
    def __init__(self, max_tokens_l1: int = 1200, max_tokens_l2: int = 600, max_tokens_l3: int = 400):
        """Initialize with token limits for each level."""
        self.max_tokens_l1 = max_tokens_l1
        self.max_tokens_l2 = max_tokens_l2
        self.max_tokens_l3 = max_tokens_l3
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)."""
        return len(text) // 4
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove markdown formatting
        text = text.replace('**', '').replace('*', '')
        text = text.replace('\n\n', '\n').replace('\r', '')
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def create_level1_chunks(self, posts_data: List[Dict]) -> List[Dict]:
        """Create Level 1 chunks: Post + top comments."""
        chunks = []
        
        for post in posts_data:
            # Start with post content
            title = self.clean_text(post.get('title', ''))
            selftext = self.clean_text(post.get('selftext', ''))
            
            content = f"Title: {title}\n\n"
            if selftext:
                content += f"Post: {selftext}\n\n"
            
            # Add metadata
            content += f"Author: {post.get('author', 'Unknown')}\n"
            content += f"Score: {post.get('score', 0)}\n"
            content += f"Comments: {post.get('num_comments', 0)}\n\n"
            
            # Add top comments (sorted by score)
            comments = sorted(post.get('comments', []), key=lambda x: x.get('score', 0), reverse=True)
            
            content += "Top Community Responses:\n"
            current_tokens = self.estimate_tokens(content)
            
            for i, comment in enumerate(comments[:10]):  # Max 10 top comments
                comment_text = self.clean_text(comment.get('body', ''))
                comment_addition = f"\n[Comment {i+1}] {comment.get('author', 'Unknown')} (Score: {comment.get('score', 0)}): {comment_text}\n"
                
                if current_tokens + self.estimate_tokens(comment_addition) > self.max_tokens_l1:
                    break
                    
                content += comment_addition
                current_tokens += self.estimate_tokens(comment_addition)
            
            chunk = {
                'id': f"l1_{post['id']}",
                'level': 1,
                'content': content,
                'metadata': {
                    'post_id': post['id'],
                    'content_type': 'post_with_comments',
                    'title': title,
                    'author': post.get('author'),
                    'score': post.get('score', 0),
                    'created_utc': post.get('created_utc'),
                    'num_comments': len(comments),
                    'permalink': post.get('permalink')
                }
            }
            chunks.append(chunk)
            
        return chunks
    
    def create_level2_chunks(self, posts_data: List[Dict]) -> List[Dict]:
        """Create Level 2 chunks: Individual comments with context."""
        chunks = []
        
        for post in posts_data:
            post_title = self.clean_text(post.get('title', ''))
            
            for comment in post.get('comments', []):
                comment_text = self.clean_text(comment.get('body', ''))
                
                if not comment_text or len(comment_text) < 20:  # Skip very short comments
                    continue
                
                # Build context
                content = f"Post Context: {post_title}\n\n"
                
                # Add parent comment if it's a reply
                if comment.get('parent_type') == 'comment':
                    # Find parent comment
                    parent_comment = next(
                        (c for c in post.get('comments', []) if c['id'] == comment.get('parent_id')), 
                        None
                    )
                    if parent_comment:
                        parent_text = self.clean_text(parent_comment.get('body', ''))[:200]  # Truncate if long
                        content += f"Replying to: {parent_text}\n\n"
                
                content += f"Comment: {comment_text}\n\n"
                content += f"Author: {comment.get('author', 'Unknown')}\n"
                content += f"Score: {comment.get('score', 0)}"
                
                # Check token limit
                if self.estimate_tokens(content) > self.max_tokens_l2:
                    # Truncate comment if too long
                    available_tokens = self.max_tokens_l2 - self.estimate_tokens(content.replace(comment_text, ''))
                    max_comment_chars = available_tokens * 4
                    comment_text = comment_text[:max_comment_chars] + "..."
                    content = content.replace(comment.get('body', ''), comment_text)
                
                chunk = {
                    'id': f"l2_{comment['id']}",
                    'level': 2,
                    'content': content,
                    'metadata': {
                        'post_id': post['id'],
                        'comment_id': comment['id'],
                        'content_type': 'comment_with_context',
                        'author': comment.get('author'),
                        'score': comment.get('score', 0),
                        'created_utc': comment.get('created_utc'),
                        'parent_type': comment.get('parent_type'),
                        'permalink': comment.get('permalink')
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def create_level3_chunks(self, high_value_comments: List[Dict]) -> List[Dict]:
        """Create Level 3 chunks: High-value standalone comments."""
        chunks = []
        
        for comment in high_value_comments:
            comment_text = self.clean_text(comment.get('body', ''))
            post_title = self.clean_text(comment.get('post_title', ''))
            
            content = f"Context: {post_title}\n\n"
            content += f"High-Value Response: {comment_text}\n\n"
            content += f"Author: {comment.get('author', 'Unknown')}\n"
            content += f"Community Score: {comment.get('score', 0)}"
            
            # Ensure within token limit
            if self.estimate_tokens(content) > self.max_tokens_l3:
                available_tokens = self.max_tokens_l3 - self.estimate_tokens(content.replace(comment_text, ''))
                max_comment_chars = available_tokens * 4
                comment_text = comment_text[:max_comment_chars] + "..."
                content = content.replace(comment.get('body', ''), comment_text)
            
            chunk = {
                'id': f"l3_{comment['id']}",
                'level': 3,
                'content': content,
                'metadata': {
                    'post_id': comment['post_id'],
                    'comment_id': comment['id'],
                    'content_type': 'high_value_comment',
                    'author': comment.get('author'),
                    'score': comment.get('score', 0),
                    'created_utc': comment.get('created_utc'),
                    'post_title': post_title,
                    'permalink': comment.get('permalink')
                }
            }
            chunks.append(chunk)
        
        return chunks

def main_batch_processing():
    """Alternative main function for batch processing large datasets."""
    
    db_config = {
        'host': 'localhost',
        'port': 3306,
        'database': 'reddit_mindfulness',
        'user': 'root',
        'password': 'admin123',
        'charset': 'utf8mb4',
        'use_unicode': True
    }
    
    extractor = RedditDataExtractor(db_config)
    chunker = HierarchicalChunker()
    
    all_chunks = []
    batch_size = 100  # Process 100 posts at a time
    offset = 0
    
    try:
        # Get total post count first
        extractor.connect()
        cursor = extractor.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM posts WHERE title IS NOT NULL")
        total_posts = cursor.fetchone()[0]
        cursor.close()
        
        logger.info(f"Total posts to process: {total_posts}")
        
        # Process in batches
        while offset < total_posts:
            logger.info(f"Processing batch: {offset}-{offset+batch_size}")
            
            # Extract batch
            posts_data = extractor.extract_posts_with_comments(limit=batch_size, offset=offset)
            
            if not posts_data:
                break
                
            # Create chunks for this batch
            l1_chunks = chunker.create_level1_chunks(posts_data)
            l2_chunks = chunker.create_level2_chunks(posts_data)
            
            all_chunks.extend(l1_chunks)
            all_chunks.extend(l2_chunks)
            
            logger.info(f"Batch {offset//batch_size + 1}: +{len(l1_chunks)} L1 chunks, +{len(l2_chunks)} L2 chunks")
            
            offset += batch_size
        
        # Process high-value comments once
        logger.info("Processing high-value comments...")
        high_value_comments = extractor.get_high_value_comments(min_score=3)
        l3_chunks = chunker.create_level3_chunks(high_value_comments)
        all_chunks.extend(l3_chunks)
        
        # Final summary
        l1_count = len([c for c in all_chunks if c['level'] == 1])
        l2_count = len([c for c in all_chunks if c['level'] == 2])
        l3_count = len([c for c in all_chunks if c['level'] == 3])
        
        print(f"\n=== Final Chunking Results ===")
        print(f"Level 1 chunks (Post + Comments): {l1_count}")
        print(f"Level 2 chunks (Individual Comments): {l2_count}")
        print(f"Level 3 chunks (High-Value Comments): {l3_count}")
        print(f"Total chunks: {len(all_chunks)}")
        
        # Save chunks
        with open('hierarchical_chunks_full.json', 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info("All chunks saved to hierarchical_chunks_full.json")
        
    finally:
        extractor.disconnect()

def main():
    """Main function to demonstrate the data extraction and chunking."""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 3306,
        'database': 'reddit_mindfulness',
        'user': 'root',
        'password': 'admin123',
        'charset': 'utf8mb4',
        'use_unicode': True
    }
    
    # Initialize extractor
    extractor = RedditDataExtractor(db_config)
    
    try:
        # Analyze corpus
        logger.info("Analyzing corpus...")
        analysis = extractor.analyze_corpus()
        print("=== Corpus Analysis ===")
        print(json.dumps(analysis, indent=2, default=str))
        
        # Check if we should use batch processing
        total_posts = analysis.get('total_posts', 0)
        if total_posts > 200:
            print(f"\nDetected {total_posts} posts. Recommend using batch processing.")
            print("Run main_batch_processing() instead for better memory management.")
            
            # Ask user preference
            response = input("Process all posts now? (y/n): ").lower().strip()
            if response != 'y':
                logger.info("Exiting. Run main_batch_processing() when ready.")
                return
        
        # Extract all data (remove limit to get all posts)
        logger.info("Extracting all posts with comments...")
        posts_data = extractor.extract_posts_with_comments()  # Process all posts
        
        logger.info("Extracting high-value comments...")
        high_value_comments = extractor.get_high_value_comments(min_score=3)  # Removed limit, lowered min_score
        
        # Initialize chunker
        chunker = HierarchicalChunker()
        
        # Create chunks
        logger.info("Creating Level 1 chunks...")
        l1_chunks = chunker.create_level1_chunks(posts_data)
        
        logger.info("Creating Level 2 chunks...")
        l2_chunks = chunker.create_level2_chunks(posts_data)
        
        logger.info("Creating Level 3 chunks...")
        l3_chunks = chunker.create_level3_chunks(high_value_comments)
        
        # Summary
        print(f"\n=== Chunking Results ===")
        print(f"Level 1 chunks (Post + Comments): {len(l1_chunks)}")
        print(f"Level 2 chunks (Individual Comments): {len(l2_chunks)}")
        print(f"Level 3 chunks (High-Value Comments): {len(l3_chunks)}")
        print(f"Total chunks: {len(l1_chunks) + len(l2_chunks) + len(l3_chunks)}")
        
        # Show sample chunks
        print(f"\n=== Sample Level 1 Chunk ===")
        if l1_chunks:
            print(f"ID: {l1_chunks[0]['id']}")
            print(f"Content: {l1_chunks[0]['content'][:500]}...")
            print(f"Metadata: {l1_chunks[0]['metadata']}")
        
        # Save chunks to JSON for inspection
        all_chunks = l1_chunks + l2_chunks + l3_chunks
        with open('hierarchical_chunks.json', 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info("Chunks saved to hierarchical_chunks.json")
        
    finally:
        extractor.disconnect()

if __name__ == "__main__":
    # Choose processing method
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        main_batch_processing()
    else:
        main()