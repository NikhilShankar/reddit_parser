import mysql.connector
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import re
import logging
from typing import List, Dict, Tuple
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MindfulnessTopicDiscovery:
    def __init__(self, db_config: Dict):
        """Initialize topic discovery with database configuration."""
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

    def extract_all_content(self, min_word_count: int = 5) -> List[Dict]:
        """Extract all posts and comments from database with metadata."""
        if not self.connection:
            if not self.connect_to_database():
                return []

        cursor = self.connection.cursor(dictionary=True)

        # Extract posts
        logger.info("Extracting posts...")
        posts_query = """
            SELECT 
                id, title, selftext, author, score, created_utc, num_comments,
                'post' as content_type
            FROM posts 
            WHERE title IS NOT NULL 
            AND CHAR_LENGTH(TRIM(COALESCE(title, '') + ' ' + COALESCE(selftext, ''))) > 0
        """
        cursor.execute(posts_query)
        posts = cursor.fetchall()
        logger.info(f"Extracted {len(posts)} posts")

        # Extract comments
        logger.info("Extracting comments...")
        comments_query = """
            SELECT 
                c.id, c.body as content, c.author, c.score, c.created_utc, c.post_id,
                p.title as post_title, 'comment' as content_type
            FROM comments c
            JOIN posts p ON c.post_id = p.id
            WHERE c.body IS NOT NULL 
            AND c.body NOT IN ('[deleted]', '[removed]')
            AND CHAR_LENGTH(TRIM(c.body)) > 0
        """
        cursor.execute(comments_query)
        comments = cursor.fetchall()
        logger.info(f"Extracted {len(comments)} comments")

        # Combine and process all content
        all_content = []

        # Process posts
        for post in posts:
            # Combine title and selftext
            text = (post.get('title', '') + ' ' + post.get('selftext', '')).strip()

            # Filter by word count
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

        # Process comments
        for comment in comments:
            text = comment.get('content', '').strip()

            # Filter by word count
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
        logger.info(f"Total content items after filtering: {len(all_content)}")

        return all_content

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for topic modeling."""
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '', text)  # Remove username mentions
        text = re.sub(r'/r/\w+', '', text)  # Remove subreddit mentions
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold formatting
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic formatting

        # Remove excessive whitespace and newlines
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Basic cleaning
        text = text.strip()

        return text

    def prepare_documents(self, content_data: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Prepare documents for topic modeling."""
        logger.info("Cleaning and preparing documents...")

        documents = []
        metadata = []

        for item in content_data:
            cleaned_text = self.clean_text(item['text'])

            if cleaned_text and len(cleaned_text.split()) >= 5:  # Double-check word count
                documents.append(cleaned_text)
                metadata.append(item)

        logger.info(f"Prepared {len(documents)} documents for topic modeling")
        return documents, metadata

    #try this as well : all-mpnet-base-v2
    def load_embedding_model(self, model_name: str = "all-MiniLM-L12-v2"):
        """Load the same embedding model used in the chatbot."""
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

    def create_topic_model(self):
        """Create and configure BERTopic model."""
        logger.info("Configuring BERTopic model...")

        # Use the same embedding model as the chatbot for consistency
        if not self.embedding_model:
            if not self.load_embedding_model():
                return None

        # Configure UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        # Configure HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=5,  # Minimum 5 posts/comments per topic (as requested)
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # Configure vectorizer for topic words - increased for 100K posts
        vectorizer_model = CountVectorizer(
            max_features=12000,  # Increased from 1000 to 12000 for rich vocabulary
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better topic descriptions
            min_df=3,  # Word must appear in at least 3 documents (increased for larger dataset)
            max_df=0.95  # Exclude words that appear in >95% of documents
        )

        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics="auto",  # Let algorithm decide
            calculate_probabilities=True,
            verbose=True
        )

        self.topic_model = topic_model
        logger.info("BERTopic model configured successfully")
        return topic_model

    def fit_topic_model(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """Fit the topic model on documents."""
        if not self.topic_model:
            if not self.create_topic_model():
                return None, None

        logger.info(f"Fitting topic model on {len(documents)} documents...")
        logger.info("This may take several minutes depending on your dataset size...")

        try:
            # Fit the model and get topics and probabilities
            topics, probabilities = self.topic_model.fit_transform(documents)

            self.topics = topics
            self.probabilities = probabilities

            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            logger.info(f"Discovered {len(topic_info)} topics (including outliers)")

            return topics, probabilities

        except Exception as e:
            logger.error(f"Error fitting topic model: {e}")
            return None, None

    def analyze_topics(self) -> Dict:
        """Analyze and summarize discovered topics with application development focus."""
        if not self.topic_model:
            logger.error("Topic model not fitted yet")
            return {}

        logger.info("Analyzing discovered topics for application development insights...")

        # Get topic information
        topic_info = self.topic_model.get_topic_info()

        # Get topics (excluding outliers, topic -1)
        topics_analysis = {}

        for _, row in topic_info.iterrows():
            topic_id = row['Topic']

            if topic_id == -1:  # Skip outliers
                continue

            # Get top words for this topic
            topic_words = self.topic_model.get_topic(topic_id)
            top_words = [word for word, _ in topic_words[:15]]  # Increased for better analysis

            # Get representative documents
            representative_docs = self.topic_model.get_representative_docs(topic_id)

            # Categorize topic tier based on document count for application focus
            doc_count = row['Count']
            if doc_count >= 100:
                tier = "Tier 1 - Major User Need"
                analysis_depth = "comprehensive"
            elif doc_count >= 30:
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
                'representative_docs': representative_docs[:5],  # More samples for analysis
                'topic_label': f"Topic {topic_id}: {', '.join(top_words[:4])}",
                'application_priority': self._calculate_app_priority(doc_count, top_words)
            }

        # Calculate comprehensive statistics for application development
        tier_1_topics = [t for t in topics_analysis.values() if t['tier'].startswith('Tier 1')]
        tier_2_topics = [t for t in topics_analysis.values() if t['tier'].startswith('Tier 2')]
        tier_3_topics = [t for t in topics_analysis.values() if t['tier'].startswith('Tier 3')]

        analysis_summary = {
            'total_topics': len(topics_analysis),
            'total_documents': len(self.topics),
            'outliers_count': sum(1 for t in self.topics if t == -1),
            'tier_distribution': {
                'tier_1_major_needs': len(tier_1_topics),
                'tier_2_secondary_needs': len(tier_2_topics),
                'tier_3_niche_needs': len(tier_3_topics)
            },
            'application_insights': {
                'primary_user_needs': [t['topic_label'] for t in tier_1_topics[:10]],
                'secondary_opportunities': [t['topic_label'] for t in tier_2_topics[:10]],
                'niche_features': [t['topic_label'] for t in tier_3_topics[:5]]
            },
            'topics': topics_analysis
        }

        return analysis_summary

    def _calculate_app_priority(self, doc_count: int, words: List[str]) -> str:
        """Calculate application development priority based on topic characteristics."""
        # High-priority keywords that indicate strong user needs
        high_priority_keywords = [
            'anxiety', 'stress', 'sleep', 'focus', 'beginner', 'daily', 'work',
            'habit', 'routine', 'time', 'busy', 'quick', 'easy', 'help'
        ]

        # Count high-priority keywords in topic
        priority_score = sum(1 for word in words if word.lower() in high_priority_keywords)

        if doc_count >= 100 and priority_score >= 2:
            return "High Priority - Major User Pain Point"
        elif doc_count >= 50 and priority_score >= 1:
            return "Medium Priority - Common User Need"
        elif doc_count >= 20:
            return "Low Priority - Niche User Interest"
        else:
            return "Research Only - Very Specific Use Case"

    def save_results(self, analysis: Dict, metadata: List[Dict], filename_prefix: str = "mindfulness_topics"):
        """Save topic analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save topic analysis as JSON
        analysis_file = f"{filename_prefix}_analysis_{timestamp}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Topic analysis saved to {analysis_file}")

        # Save the fitted model
        model_file = f"{filename_prefix}_model_{timestamp}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.topic_model, f)
        logger.info(f"Topic model saved to {model_file}")

        # Save document-topic mappings
        if self.topics is not None and metadata:
            doc_topics = []
            for i, (topic_id, doc_meta) in enumerate(zip(self.topics, metadata)):
                doc_topics.append({
                    **doc_meta,
                    'topic_id': int(topic_id),
                    'topic_probability': float(
                        self.probabilities[i][topic_id]) if self.probabilities is not None else None
                })

            mappings_file = f"{filename_prefix}_document_mappings_{timestamp}.json"
            with open(mappings_file, 'w', encoding='utf-8') as f:
                json.dump(doc_topics, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Document-topic mappings saved to {mappings_file}")

        return analysis_file, model_file

    def print_topic_summary(self, analysis: Dict):
        """Print a comprehensive summary focused on application development insights."""
        print("\n" + "=" * 80)
        print("🧘 MINDFULNESS APPLICATION DEVELOPMENT - TOPIC ANALYSIS")
        print("=" * 80)

        print(f"\n📊 DATASET SUMMARY:")
        print(f"   • Total Topics Discovered: {analysis['total_topics']}")
        print(f"   • Total Documents Analyzed: {analysis['total_documents']}")
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

        print(f"\n📋 DETAILED TOPIC BREAKDOWN:\n")

        # Sort topics by tier and document count
        sorted_topics = sorted(
            analysis['topics'].items(),
            key=lambda x: (x[1]['tier'], -x[1]['count'])
        )

        current_tier = None
        for topic_id, topic_data in sorted_topics:
            # Print tier header when it changes
            if current_tier != topic_data['tier']:
                current_tier = topic_data['tier']
                print(f"\n🏷️  {current_tier.upper()}")
                print("-" * 60)

            print(f"\n📌 {topic_data['topic_label']}")
            print(f"    Documents: {topic_data['count']} | Priority: {topic_data['application_priority']}")
            print(f"    Key concepts: {', '.join(topic_data['top_words'][:8])}")
            print(f"    Analysis depth: {topic_data['analysis_depth']}")
            print(f"    Sample content: \"{topic_data['representative_docs'][0][:120]}...\"")

        print(f"\n" + "=" * 80)
        print("🎯 NEXT STEPS FOR APPLICATION DEVELOPMENT:")
        print("1. Step 2: Refine topics into hierarchical categories")
        print("2. Step 3: Generate detailed user need analysis for each tier")
        print("3. Step 4: Create comprehensive application requirements document")
        print("=" * 80)


def main():
    """Main function to run topic discovery."""

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

    # Initialize topic discovery
    topic_discovery = MindfulnessTopicDiscovery(db_config)

    try:
        # Step 1: Extract all content from database
        logger.info("Step 1: Extracting content from database...")
        content_data = topic_discovery.extract_all_content(min_word_count=5)

        if not content_data:
            logger.error("No content extracted from database")
            return

        # Step 2: Prepare documents for topic modeling
        logger.info("Step 2: Preparing documents...")
        documents, metadata = topic_discovery.prepare_documents(content_data)

        if not documents:
            logger.error("No documents prepared for modeling")
            return

        # Step 3: Fit topic model
        logger.info("Step 3: Fitting topic model (this may take a while)...")
        topics, probabilities = topic_discovery.fit_topic_model(documents)

        if topics is None:
            logger.error("Topic modeling failed")
            return

        # Step 4: Analyze results
        logger.info("Step 4: Analyzing topics...")
        analysis = topic_discovery.analyze_topics()

        # Step 5: Display and save results
        logger.info("Step 5: Saving results...")
        topic_discovery.print_topic_summary(analysis)

        analysis_file, model_file = topic_discovery.save_results(analysis, metadata)

        print(f"\n✅ Topic discovery completed successfully!")
        print(f"📁 Results saved:")
        print(f"   • Analysis: {analysis_file}")
        print(f"   • Model: {model_file}")
        print(f"\n🎯 Next step: Review topics and create topic-specific summaries")

    except Exception as e:
        logger.error(f"Topic discovery failed: {e}")
        raise
    finally:
        topic_discovery.close_connection()


if __name__ == "__main__":
    main()