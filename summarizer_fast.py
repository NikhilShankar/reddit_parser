import mysql.connector
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from umap import UMAP
from hdbscan import HDBSCAN
import re
import logging
from typing import List, Dict, Tuple
import json
import pickle
from datetime import datetime
import random
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastMindfulnessTopicDiscovery:
    def __init__(self, db_config: Dict):
        """Initialize fast topic discovery with optimized parameters."""
        self.db_config = db_config
        self.connection = None
        self.embedding_model = None
        self.topic_model = None
        self.documents = []
        self.embeddings = None
        self.topics = None
        self.probabilities = None

    def connect_to_database(self):
        """Connect to MySQL database."""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            logger.info("Successfully connected to database")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def close_connection(self):
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")

    def extract_sample_content(self, top_comments: int = 10000, min_word_count: int = 5) -> List[Dict]:
        """Extract ALL posts plus top comments for comprehensive analysis."""
        if not self.connection:
            if not self.connect_to_database():
                return []

        cursor = self.connection.cursor(dictionary=True)

        logger.info("Extracting ALL posts and top comments for comprehensive analysis...")

        # Strategy: ALL posts + top N comments by score

        # Get ALL posts (no limit)
        all_posts_query = """
            SELECT
                id, title, selftext, author, score, created_utc, num_comments,
                'post' as content_type
            FROM posts
            WHERE title IS NOT NULL
            ORDER BY score DESC, created_utc DESC
        """
        cursor.execute(all_posts_query)
        all_posts = cursor.fetchall()
        logger.info(f"Extracted ALL {len(all_posts)} posts")

        # Get top N comments by score
        top_comments_query = """
            SELECT 
                c.id, c.body as content, c.author, c.score, c.created_utc, c.post_id,
                p.title as post_title, 'comment' as content_type,
                CHAR_LENGTH(TRIM(c.body)) as text_length
            FROM comments c
            JOIN posts p ON c.post_id = p.id
            WHERE c.body IS NOT NULL 
            AND c.body NOT IN ('[deleted]', '[removed]')
            AND CHAR_LENGTH(TRIM(c.body)) >= 20
            ORDER BY c.score DESC
            LIMIT %s
        """
        cursor.execute(top_comments_query, (top_comments,))
        top_comments_data = cursor.fetchall()
        logger.info(f"Extracted top {len(top_comments_data)} comments by score")

        # Process and combine content
        all_content = []

        # Process ALL posts first
        logger.info("Processing all posts...")
        for post in all_posts:
            text = (post.get('title', '') + ' ' + post.get('selftext', '')).strip()

            if len(text.split()) >= min_word_count:
                content_item = {
                    'id': f"post_{post['id']}",
                    'text': text,
                    'content_type': 'post',
                    'author': post.get('author'),
                    'score': post.get('score', 0),
                    'created_utc': post.get('created_utc'),
                    'post_id': post['id'],
                    'title': post.get('title', ''),
                    'word_count': len(text.split())
                }
                all_content.append(content_item)

        # Process top comments
        logger.info("Processing top comments...")
        for comment in top_comments_data:
            text = comment.get('content', '').strip()

            if len(text.split()) >= min_word_count:
                content_item = {
                    'id': f"comment_{comment['id']}",
                    'text': text,
                    'content_type': 'comment',
                    'author': comment.get('author'),
                    'score': comment.get('score', 0),
                    'created_utc': comment.get('created_utc'),
                    'post_id': comment.get('post_id'),
                    'post_title': comment.get('post_title', ''),
                    'word_count': len(text.split())
                }
                all_content.append(content_item)

        cursor.close()

        logger.info(f"Total content: {len(all_content)} items")
        logger.info(f"Posts: {len([c for c in all_content if c['content_type'] == 'post'])}")
        logger.info(f"Comments: {len([c for c in all_content if c['content_type'] == 'comment'])}")

        return all_content

    def clean_text_fast(self, text: str) -> str:
        """Fast text cleaning optimized for speed."""
        if not text:
            return ""

        # Basic cleaning only - speed over perfection
        text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
        text = re.sub(r'/[ur]/\w+', '', text)  # Remove Reddit mentions
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()

        return text

    def prepare_documents_fast(self, content_data: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Fast document preparation."""
        logger.info("Fast cleaning and preparing documents...")

        documents = []
        metadata = []

        for item in content_data:
            cleaned_text = self.clean_text_fast(item['text'])

            if cleaned_text and len(cleaned_text.split()) >= 5:
                documents.append(cleaned_text)
                metadata.append(item)

        logger.info(f"Prepared {len(documents)} documents for topic modeling")
        return documents, metadata

    def load_embedding_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load faster, smaller embedding model."""
        logger.info(f"Loading fast embedding model: {model_name}")
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Fast embedding model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

    def create_fast_topic_model(self):
        """Create optimized BERTopic model for speed."""
        logger.info("Configuring fast BERTopic model...")

        if not self.embedding_model:
            if not self.load_embedding_model():
                return None

        # Optimized UMAP for speed
        umap_model = UMAP(
            n_neighbors=10,  # Reduced from 15
            n_components=3,  # Reduced from 5
            min_dist=0.0,
            metric='cosine',
            random_state=42,
            low_memory=True  # Speed optimization
        )

        # Optimized HDBSCAN for speed
        hdbscan_model = HDBSCAN(
            min_cluster_size=15,  # Increased for faster clustering
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=False,  # Disable for speed
            memory='memory.pkl'  # Cache for speed
        )

        # Optimized vectorizer for speed
        vectorizer_model = CountVectorizer(
            max_features=5000,  # Reduced from 12000
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,  # Increased for speed
            max_df=0.9  # More aggressive filtering
        )

        # Create fast BERTopic model
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics="auto",
            calculate_probabilities=False,  # Disabled for speed
            verbose=True
        )

        self.topic_model = topic_model
        logger.info("Fast BERTopic model configured successfully")
        return topic_model

    def fit_topic_model_fast(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """Fit topic model with speed optimizations."""
        if not self.topic_model:
            if not self.create_fast_topic_model():
                return None, None

        logger.info(f"Fast fitting topic model on {len(documents)} documents...")
        logger.info("Estimated time: 10-30 minutes with optimizations")

        try:
            # Fit the model - no probabilities for speed
            topics = self.topic_model.fit_transform(documents)

            # Ensure topics is a flat array
            if isinstance(topics, tuple):
                topics = topics[0]  # fit_transform returns (topics, probabilities) sometimes

            # Convert to numpy array if it's not already
            topics = np.array(topics) if not isinstance(topics, np.ndarray) else topics

            # Flatten if needed
            if topics.ndim > 1:
                topics = topics.flatten()

            self.topics = topics
            self.probabilities = None  # Not calculated for speed

            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            logger.info(f"Discovered {len(topic_info)} topics (including outliers)")
            logger.info(f"Topics array shape: {topics.shape}, dtype: {topics.dtype}")

            return topics, None

        except Exception as e:
            logger.error(f"Error fitting topic model: {e}")
            return None, None

    def analyze_topics_fast(self, metadata: List[Dict]) -> Dict:
        """Fast topic analysis with application focus."""
        if not self.topic_model:
            logger.error("Topic model not fitted yet")
            return {}

        logger.info("Fast analyzing discovered topics...")

        # Get topic information
        topic_info = self.topic_model.get_topic_info()

        # Calculate content type breakdown
        posts_count = len([doc for doc in metadata if doc['content_type'] == 'post'])
        comments_count = len([doc for doc in metadata if doc['content_type'] == 'comment'])

        # Get topics (excluding outliers, topic -1)
        topics_analysis = {}

        for _, row in topic_info.iterrows():
            topic_id = row['Topic']

            if topic_id == -1:  # Skip outliers
                continue

            # Get top words for this topic
            topic_words = self.topic_model.get_topic(topic_id)
            top_words = [word for word, _ in topic_words[:12]]

            # Get representative documents
            representative_docs = self.topic_model.get_representative_docs(topic_id)

            # Fast tier categorization
            doc_count = row['Count']
            if doc_count >= 50:
                tier = "Tier 1 - Major User Need"
                analysis_depth = "comprehensive"
            elif doc_count >= 20:
                tier = "Tier 2 - Secondary User Need"
                analysis_depth = "moderate"
            else:
                tier = "Tier 3 - Niche User Need"
                analysis_depth = "brief"

            topics_analysis[topic_id] = {
                'topic_id': topic_id,
                'count': doc_count,
                'tier': tier,
                'analysis_depth': analysis_depth,
                'top_words': top_words,
                'representative_docs': representative_docs[:3],
                'topic_label': f"Topic {topic_id}: {', '.join(top_words[:4])}",
                'application_priority': self._calculate_app_priority_fast(doc_count, top_words)
            }

        # Fast statistics calculation
        tier_1_topics = [t for t in topics_analysis.values() if t['tier'].startswith('Tier 1')]
        tier_2_topics = [t for t in topics_analysis.values() if t['tier'].startswith('Tier 2')]
        tier_3_topics = [t for t in topics_analysis.values() if t['tier'].startswith('Tier 3')]

        analysis_summary = {
            'total_topics': len(topics_analysis),
            'total_documents': len(self.topics),
            'content_breakdown': {
                'posts_count': posts_count,
                'comments_count': comments_count,
                'total_content': posts_count + comments_count
            },
            'sample_note': 'Analysis based on ALL posts + top comments',
            'outliers_count': int(np.sum(self.topics == -1)),  # Fixed: numpy array comparison
            'tier_distribution': {
                'tier_1_major_needs': len(tier_1_topics),
                'tier_2_secondary_needs': len(tier_2_topics),
                'tier_3_niche_needs': len(tier_3_topics)
            },
            'application_insights': {
                'primary_user_needs': [t['topic_label'] for t in tier_1_topics[:8]],
                'secondary_opportunities': [t['topic_label'] for t in tier_2_topics[:8]],
                'niche_features': [t['topic_label'] for t in tier_3_topics[:5]]
            },
            'topics': topics_analysis
        }

        return analysis_summary

    def _calculate_app_priority_fast(self, doc_count: int, words: List[str]) -> str:
        """Fast application priority calculation."""
        high_priority_keywords = [
            'anxiety', 'stress', 'sleep', 'focus', 'beginner', 'daily', 'work',
            'habit', 'routine', 'time', 'busy', 'quick', 'easy', 'help', 'start'
        ]

        priority_score = sum(1 for word in words if word.lower() in high_priority_keywords)

        if doc_count >= 50 and priority_score >= 2:
            return "High Priority - Major User Pain Point"
        elif doc_count >= 30 and priority_score >= 1:
            return "Medium Priority - Common User Need"
        elif doc_count >= 15:
            return "Low Priority - Niche User Interest"
        else:
            return "Research Only - Very Specific Use Case"

    def save_results_fast(self, analysis: Dict, metadata: List[Dict], filename_prefix: str = "mindfulness_topics_fast"):
        """Save results with fast processing indicator."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Add processing metadata
        analysis['processing_info'] = {
            'method': 'fast_sampling',
            'sample_size': len(metadata),
            'processing_time': 'optimized_for_speed',
            'timestamp': timestamp
        }

        # Save topic analysis as JSON
        analysis_file = f"{filename_prefix}_analysis_{timestamp}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Fast topic analysis saved to {analysis_file}")

        # Save the fitted model
        model_file = f"{filename_prefix}_model_{timestamp}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.topic_model, f)
        logger.info(f"Fast topic model saved to {model_file}")

        # Save document-topic mappings
        if self.topics is not None and metadata:
            doc_topics = []
            for i, (topic_id, doc_meta) in enumerate(zip(self.topics, metadata)):
                doc_topics.append({
                    **doc_meta,
                    'topic_id': int(topic_id) if hasattr(topic_id, 'item') else int(topic_id)  # Handle numpy scalars
                })

            mappings_file = f"{filename_prefix}_document_mappings_{timestamp}.json"
            with open(mappings_file, 'w', encoding='utf-8') as f:
                json.dump(doc_topics, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Fast document-topic mappings saved to {mappings_file}")

        return analysis_file, model_file

    def print_topic_summary_fast(self, analysis: Dict):
        """Print fast topic discovery summary."""
        print("\n" + "=" * 80)
        print("🚀 COMPREHENSIVE MINDFULNESS TOPIC DISCOVERY")
        print("=" * 80)

        print(f"\n⚡ COMPREHENSIVE PROCESSING SUMMARY:")
        content_breakdown = analysis.get('content_breakdown', {})
        print(f"   • ALL posts included: {content_breakdown.get('posts_count', 'N/A')} posts")
        print(f"   • Top comments included: {content_breakdown.get('comments_count', 'N/A')} comments")
        print(
            f"   • Total documents analyzed: {content_breakdown.get('total_content', analysis.get('total_documents', 'N/A'))}")
        print(f"   • Total Topics Discovered: {analysis['total_topics']}")
        print(f"   • Processing Method: ALL posts + top comments")
        print(f"   • Outliers (uncategorized): {analysis['outliers_count']}")

        print(f"\n🎯 APPLICATION DEVELOPMENT TIERS:")
        tier_dist = analysis['tier_distribution']
        print(f"   • Tier 1 (Major User Needs): {tier_dist['tier_1_major_needs']} topics")
        print(f"   • Tier 2 (Secondary Needs): {tier_dist['tier_2_secondary_needs']} topics")
        print(f"   • Tier 3 (Niche Needs): {tier_dist['tier_3_niche_needs']} topics")

        print(f"\n🚀 TOP APPLICATION OPPORTUNITIES:")
        app_insights = analysis['application_insights']
        print(f"\n   PRIMARY USER NEEDS (Tier 1):")
        for i, need in enumerate(app_insights['primary_user_needs'], 1):
            print(f"   {i:2}. {need}")

        print(f"\n   SECONDARY OPPORTUNITIES (Tier 2):")
        for i, opp in enumerate(app_insights['secondary_opportunities'], 1):
            print(f"   {i:2}. {opp}")

        print(f"\n📋 SAMPLE TOPIC BREAKDOWN:\n")

        # Show top topics from each tier
        sorted_topics = sorted(
            analysis['topics'].items(),
            key=lambda x: (x[1]['tier'], -x[1]['count'])
        )

        current_tier = None
        topic_count = 0
        for topic_id, topic_data in sorted_topics:
            if current_tier != topic_data['tier']:
                current_tier = topic_data['tier']
                print(f"\n🏷️  {current_tier.upper()}")
                print("-" * 60)
                topic_count = 0

            if topic_count < 3:  # Show top 3 per tier for speed
                print(f"\n📌 {topic_data['topic_label']}")
                print(f"    Documents: {topic_data['count']} | Priority: {topic_data['application_priority']}")
                print(f"    Key concepts: {', '.join(topic_data['top_words'][:6])}")
                print(f"    Sample: \"{topic_data['representative_docs'][0][:100]}...\"")
                topic_count += 1

        print(f"\n" + "=" * 80)
        print("⚡ FAST PROCESSING COMPLETE!")
        print("✅ High-quality sample analyzed for rapid insights")
        print("🎯 Ready for application development planning")
        print("📄 Use summary document generator for detailed analysis")
        print("=" * 80)


def main():
    """Main function to run fast topic discovery."""

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

    # Fast processing configuration
    TOP_COMMENTS = 10000  # Process ALL posts + top 10K comments

    # Initialize fast topic discovery
    topic_discovery = FastMindfulnessTopicDiscovery(db_config)

    try:
        start_time = datetime.now()
        logger.info("🚀 Starting FAST topic discovery process...")

        # Step 1: Extract ALL posts + top comments
        logger.info("Step 1: Extracting ALL posts + top comments...")
        content_data = topic_discovery.extract_sample_content(top_comments=10000, min_word_count=5)

        if not content_data:
            logger.error("No content extracted from database")
            return

        # Step 2: Prepare documents
        logger.info("Step 2: Fast preparing documents...")
        documents, metadata = topic_discovery.prepare_documents_fast(content_data)

        if not documents:
            logger.error("No documents prepared for modeling")
            return

        # Step 3: Fast fit topic model
        logger.info("Step 3: Fast fitting topic model...")
        topics, probabilities = topic_discovery.fit_topic_model_fast(documents)

        if topics is None:
            logger.error("Fast topic modeling failed")
            return

        # Step 4: Fast analyze results
        logger.info("Step 4: Fast analyzing topics...")
        analysis = topic_discovery.analyze_topics_fast(metadata)

        # Step 5: Display and save results
        logger.info("Step 5: Saving fast results...")
        topic_discovery.print_topic_summary_fast(analysis)

        analysis_file, model_file = topic_discovery.save_results_fast(analysis, metadata)

        end_time = datetime.now()
        processing_time = end_time - start_time

        print(f"\n✅ FAST topic discovery completed successfully!")
        print(f"⏱️  Total processing time: {processing_time}")
        print(f"📁 Results saved:")
        print(f"   • Analysis: {analysis_file}")
        print(f"   • Model: {model_file}")
        print(f"\n🎯 Next step: Use these results with the summary document generator")
        print(f"💡 Note: This is a speed-optimized analysis of your top content")

    except Exception as e:
        logger.error(f"Fast topic discovery failed: {e}")
        raise
    finally:
        topic_discovery.close_connection()


if __name__ == "__main__":
    main()