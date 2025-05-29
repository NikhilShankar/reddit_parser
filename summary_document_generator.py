import json
import pickle
import mysql.connector
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict, Counter
import re
import os
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MindfulnessSummaryGenerator:
    def __init__(self, db_config: Dict, ollama_model: str = "gemma:2b"):
        """Initialize summary generator."""
        self.db_config = db_config
        self.ollama_model = ollama_model
        self.connection = None
        self.topic_analysis = None
        self.document_mappings = None
        self.bertopic_model = None

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

    def find_latest_files(self, prefix: str = "mindfulness_topics_fast") -> Dict[str, str]:
        """Automatically find the latest topic discovery files."""
        files = {
            'analysis': None,
            'mappings': None,
            'model': None
        }

        # Find analysis file
        analysis_pattern = f"{prefix}_analysis_*.json"
        analysis_files = glob.glob(analysis_pattern)
        if analysis_files:
            files['analysis'] = max(analysis_files, key=os.path.getctime)
            logger.info(f"Found analysis file: {files['analysis']}")

        # Find mappings file
        mappings_pattern = f"{prefix}_document_mappings_*.json"
        mappings_files = glob.glob(mappings_pattern)
        if mappings_files:
            files['mappings'] = max(mappings_files, key=os.path.getctime)
            logger.info(f"Found mappings file: {files['mappings']}")

        # Find model file
        model_pattern = f"{prefix}_model_*.pkl"
        model_files = glob.glob(model_pattern)
        if model_files:
            files['model'] = max(model_files, key=os.path.getctime)
            logger.info(f"Found model file: {files['model']}")

        return files

    def load_topic_analysis(self, analysis_file: str, mappings_file: str, model_file: str = None):
        """Load BERTopic analysis results."""
        try:
            # Load topic analysis
            with open(analysis_file, 'r', encoding='utf-8') as f:
                self.topic_analysis = json.load(f)
            logger.info(f"Loaded topic analysis from {analysis_file}")

            # Load document mappings
            with open(mappings_file, 'r', encoding='utf-8') as f:
                self.document_mappings = json.load(f)
            logger.info(f"Loaded document mappings from {mappings_file}")

            # Optionally load the model
            if model_file and os.path.exists(model_file):
                try:
                    with open(model_file, 'rb') as f:
                        self.bertopic_model = pickle.load(f)
                    logger.info(f"Loaded BERTopic model from {model_file}")
                except Exception as e:
                    logger.warning(f"Could not load model file: {e}")

            return True
        except Exception as e:
            logger.error(f"Failed to load topic analysis: {e}")
            return False

    def get_database_metadata(self) -> Dict:
        """Get metadata about the database and subreddits."""
        if not self.connection:
            if not self.connect_to_database():
                return {}

        cursor = self.connection.cursor(dictionary=True)

        try:
            # Get subreddit information
            cursor.execute("SELECT DISTINCT subreddit FROM posts WHERE subreddit IS NOT NULL")
            subreddits = [row['subreddit'] for row in cursor.fetchall()]

            # Get date range
            cursor.execute("SELECT MIN(created_utc) as earliest, MAX(created_utc) as latest FROM posts")
            date_range = cursor.fetchone()

            # Get total counts
            cursor.execute("SELECT COUNT(*) as post_count FROM posts")
            post_count = cursor.fetchone()['post_count']

            cursor.execute(
                "SELECT COUNT(*) as comment_count FROM comments WHERE body NOT IN ('[deleted]', '[removed]')")
            comment_count = cursor.fetchone()['comment_count']

            # Get top authors by activity
            cursor.execute("""
                SELECT author, COUNT(*) as activity_count 
                FROM (
                    SELECT author FROM posts WHERE author IS NOT NULL
                    UNION ALL
                    SELECT author FROM comments WHERE author IS NOT NULL AND body NOT IN ('[deleted]', '[removed]')
                ) combined
                GROUP BY author
                ORDER BY activity_count DESC
                LIMIT 10
            """)
            top_authors = cursor.fetchall()

        except Exception as e:
            logger.error(f"Error getting database metadata: {e}")
            return {}
        finally:
            cursor.close()

        return {
            'subreddits': subreddits,
            'date_range': date_range,
            'post_count': post_count,
            'comment_count': comment_count,
            'total_content': post_count + comment_count,
            'top_authors': top_authors
        }

    def get_topic_documents(self, topic_id: int, limit: int = 10) -> List[Dict]:
        """Get documents for a specific topic."""
        topic_docs = [
            doc for doc in self.document_mappings
            if doc.get('topic_id') == topic_id
        ]

        # Sort by score and return top documents
        topic_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
        return topic_docs[:limit]

    def test_ollama_connection(self) -> bool:
        """Test if Ollama is available."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate_summary_with_ollama(self, content: str, prompt_type: str, topic_title: str = "") -> str:
        """Generate summary using Ollama."""
        if not self.test_ollama_connection():
            logger.warning("Ollama not available, using basic summary")
            return self._generate_basic_summary(content, prompt_type)

        prompts = {
            'topic_summary': f"""Analyze the following mindfulness community discussions about "{topic_title}" and create a comprehensive summary for application development.

Focus on:
1. Key user pain points and needs
2. Specific techniques and approaches mentioned
3. Common challenges users face
4. Feature requirements that emerge from discussions
5. User experience insights
6. Practical recommendations

Community discussions:
{content}

Create a detailed summary (2-3 paragraphs) that helps understand user needs for building mindfulness applications:""",

            'executive_summary': f"""Analyze this comprehensive mindfulness community data and create an executive summary for application development.

Data includes topics, user discussions, and community insights about mindfulness practices.

{content}

Create an executive summary (3-4 paragraphs) covering:
1. Overall landscape of user needs
2. Primary pain points and opportunities
3. Key insights for application development
4. Strategic recommendations

Executive Summary:""",

            'tier_analysis': f"""Analyze this tier of mindfulness topics and summarize the application development implications.

{content}

Provide analysis (2-3 paragraphs) covering:
1. Common themes in this tier
2. User needs and pain points
3. Feature opportunities
4. Development priorities

Analysis:"""
        }

        prompt = prompts.get(prompt_type, prompts['topic_summary'])

        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "max_tokens": 800
                }
            }

            response = requests.post(url, json=data, timeout=120)
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                logger.warning(f"Ollama request failed, using basic summary")
                return self._generate_basic_summary(content, prompt_type)

        except Exception as e:
            logger.warning(f"Ollama generation failed: {e}, using basic summary")
            return self._generate_basic_summary(content, prompt_type)

    def _generate_basic_summary(self, content: str, prompt_type: str) -> str:
        """Generate basic summary when Ollama is not available."""
        if prompt_type == 'executive_summary':
            return "This document analyzes community discussions to understand user needs for mindfulness application development. The analysis reveals key pain points, feature opportunities, and user experience insights derived from real community conversations."
        elif prompt_type == 'tier_analysis':
            return "This tier represents significant areas of user interest within the mindfulness community. The discussions reveal specific user needs and potential application features that could address common challenges."

        return "This topic represents a significant area of user interest and discussion within the mindfulness community. Further analysis of the community conversations reveals specific user needs and potential application features."

    def create_hierarchical_structure(self) -> Dict:
        """Create hierarchical organization of topics."""
        if not self.topic_analysis:
            return {}

        # Keywords for categorizing topics into hierarchies
        categories = {
            'Meditation Practices': [
                'meditation', 'breathe', 'breathing', 'mindful', 'awareness', 'focus', 'concentration',
                'mantra', 'visualization', 'body', 'scan', 'walking', 'sitting', 'guided', 'practice'
            ],
            'Mental Health & Wellness': [
                'anxiety', 'stress', 'depression', 'mental', 'health', 'emotion', 'feeling',
                'therapy', 'healing', 'trauma', 'self', 'care', 'wellbeing', 'wellness'
            ],
            'Daily Life Integration': [
                'daily', 'routine', 'habit', 'work', 'life', 'time', 'schedule', 'busy',
                'morning', 'evening', 'commute', 'relationship', 'family', 'integration'
            ],
            'Beginner & Learning': [
                'beginner', 'start', 'learn', 'new', 'first', 'help', 'guide', 'how',
                'question', 'advice', 'tip', 'recommend', 'getting', 'started'
            ],
            'Advanced Practices': [
                'advanced', 'deeper', 'spiritual', 'retreat', 'teacher', 'master',
                'experience', 'insight', 'wisdom', 'enlightenment', 'deep'
            ],
            'Specific Techniques': [
                'technique', 'method', 'exercise', 'approach', 'way', 'type',
                'step', 'instruction', 'process', 'tool', 'app', 'resource'
            ]
        }

        hierarchy = defaultdict(list)

        for topic_id, topic_data in self.topic_analysis['topics'].items():
            words = topic_data['top_words']

            # Find best category match
            best_category = 'Uncategorized'
            best_score = 0

            for category, keywords in categories.items():
                score = sum(1 for word in words if word.lower() in keywords)
                if score > best_score:
                    best_score = score
                    best_category = category

            hierarchy[best_category].append({
                'topic_id': topic_id,
                'topic_data': topic_data,
                'category_match_score': best_score
            })

        # Sort topics within each category by document count
        for category in hierarchy:
            hierarchy[category].sort(key=lambda x: x['topic_data']['count'], reverse=True)

        return dict(hierarchy)

    def generate_comprehensive_document(self, output_filename: str = None) -> str:
        """Generate the complete summary document."""
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"mindfulness_application_analysis_{timestamp}.md"

        logger.info("Generating comprehensive summary document...")

        # Get database metadata
        db_metadata = self.get_database_metadata()

        # Create hierarchical structure
        hierarchy = self.create_hierarchical_structure()

        # Start building the document
        doc_lines = []

        # Header and metadata
        doc_lines.append("# Mindfulness Application Development Analysis")
        doc_lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        doc_lines.append("")
        doc_lines.append("---")
        doc_lines.append("")

        # Data source information
        doc_lines.append("## 📊 Data Source Information")
        doc_lines.append("")
        doc_lines.append("### Subreddits Analyzed")
        if db_metadata.get('subreddits'):
            for subreddit in db_metadata['subreddits']:
                doc_lines.append(f"- r/{subreddit}")
        else:
            doc_lines.append("- r/mindfulness (primary)")
        doc_lines.append("")

        doc_lines.append("### Dataset Overview")
        doc_lines.append(f"- **Total Posts in Database**: {db_metadata.get('post_count', 'N/A'):,}")
        doc_lines.append(f"- **Total Comments in Database**: {db_metadata.get('comment_count', 'N/A'):,}")
        doc_lines.append(f"- **Total Content Items**: {db_metadata.get('total_content', 'N/A'):,}")

        # Add content breakdown from analysis
        content_breakdown = self.topic_analysis.get('content_breakdown', {})
        if content_breakdown:
            doc_lines.append(f"- **Posts Analyzed**: {content_breakdown.get('posts_count', 'N/A'):,}")
            doc_lines.append(f"- **Comments Analyzed**: {content_breakdown.get('comments_count', 'N/A'):,}")
            doc_lines.append(
                f"- **Analysis Method**: ALL posts + top {content_breakdown.get('comments_count', 'N/A')} comments")

        doc_lines.append(f"- **Topics Discovered**: {self.topic_analysis.get('total_topics', 'N/A')}")

        if db_metadata.get('date_range'):
            earliest = db_metadata['date_range']['earliest']
            latest = db_metadata['date_range']['latest']
            if earliest and latest:
                doc_lines.append(f"- **Date Range**: {earliest} to {latest}")
        doc_lines.append("")

        # Executive Summary
        doc_lines.append("## 🎯 Executive Summary")
        doc_lines.append("")

        # Generate executive summary content
        content_info = self.topic_analysis.get('content_breakdown', {})
        exec_summary_content = f"""
        Analysis of {self.topic_analysis.get('total_topics', 'multiple')} topics from mindfulness community discussions.
        Processed {content_info.get('posts_count', 'all')} posts and {content_info.get('comments_count', 'top')} comments.
        Tier 1 topics: {self.topic_analysis.get('tier_distribution', {}).get('tier_1_major_needs', 'N/A')}
        Tier 2 topics: {self.topic_analysis.get('tier_distribution', {}).get('tier_2_secondary_needs', 'N/A')}
        Tier 3 topics: {self.topic_analysis.get('tier_distribution', {}).get('tier_3_niche_needs', 'N/A')}
        """

        exec_summary = self.generate_summary_with_ollama(exec_summary_content, 'executive_summary')
        doc_lines.append(exec_summary)
        doc_lines.append("")

        # Application Development Priorities
        doc_lines.append("## 🚀 Application Development Priorities")
        doc_lines.append("")

        app_insights = self.topic_analysis.get('application_insights', {})

        doc_lines.append("### Primary User Needs (Tier 1)")
        primary_needs = app_insights.get('primary_user_needs', [])
        for i, need in enumerate(primary_needs, 1):
            doc_lines.append(f"{i}. {need}")
        doc_lines.append("")

        doc_lines.append("### Secondary Opportunities (Tier 2)")
        secondary_opps = app_insights.get('secondary_opportunities', [])
        for i, opp in enumerate(secondary_opps, 1):
            doc_lines.append(f"{i}. {opp}")
        doc_lines.append("")

        # Hierarchical Topic Analysis
        doc_lines.append("## 📋 Detailed Topic Analysis by Category")
        doc_lines.append("")

        for category, topics in hierarchy.items():
            if not topics:
                continue

            doc_lines.append(f"### {category}")
            doc_lines.append("")

            # Generate category-level analysis
            category_content = f"Category: {category} with {len(topics)} topics"
            for topic in topics[:3]:  # Top 3 topics in category for summary
                topic_data = topic['topic_data']
                category_content += f"\nTopic: {topic_data['topic_label']} ({topic_data['count']} documents)"
                category_content += f"\nKey words: {', '.join(topic_data['top_words'][:5])}"

            category_analysis = self.generate_summary_with_ollama(category_content, 'tier_analysis')
            doc_lines.append(category_analysis)
            doc_lines.append("")

            # Individual topic details
            for topic in topics:
                topic_data = topic['topic_data']
                topic_id = topic['topic_id']

                doc_lines.append(f"#### {topic_data['topic_label']}")
                doc_lines.append("")
                doc_lines.append(f"**Tier**: {topic_data['tier']}")
                doc_lines.append(f"**Documents**: {topic_data['count']}")
                doc_lines.append(f"**Application Priority**: {topic_data['application_priority']}")
                doc_lines.append(f"**Analysis Depth**: {topic_data['analysis_depth']}")
                doc_lines.append("")

                doc_lines.append("**Key Concepts**:")
                concepts = topic_data.get('top_words', [])[:10]
                doc_lines.append(f"{', '.join(concepts)}")
                doc_lines.append("")

                # Generate detailed topic analysis for comprehensive and moderate tiers
                if topic_data['analysis_depth'] in ['comprehensive', 'moderate']:
                    topic_docs = self.get_topic_documents(int(topic_id), limit=5)
                    if topic_docs:
                        sample_content = "\n".join([doc.get('text', '')[:300] for doc in topic_docs[:3]])
                        topic_summary = self.generate_summary_with_ollama(
                            sample_content, 'topic_summary', topic_data['topic_label']
                        )
                        doc_lines.append("**User Needs Analysis**:")
                        doc_lines.append(topic_summary)
                        doc_lines.append("")

                doc_lines.append("**Representative Community Discussions**:")
                rep_docs = topic_data.get('representative_docs', [])[:3]
                for i, doc in enumerate(rep_docs, 1):
                    doc_lines.append(f"{i}. \"{doc[:150]}...\"")
                doc_lines.append("")
                doc_lines.append("---")
                doc_lines.append("")

        # Feature Development Recommendations
        doc_lines.append("## 💡 Feature Development Recommendations")
        doc_lines.append("")

        doc_lines.append("### High Priority Features")
        tier_1_topics = [t for t in self.topic_analysis['topics'].values() if t['tier'].startswith('Tier 1')]
        for topic in sorted(tier_1_topics, key=lambda x: x['count'], reverse=True)[:5]:
            doc_lines.append(f"- **{topic['topic_label']}**: Based on {topic['count']} community discussions")
        doc_lines.append("")

        doc_lines.append("### Medium Priority Features")
        tier_2_topics = [t for t in self.topic_analysis['topics'].values() if t['tier'].startswith('Tier 2')]
        for topic in sorted(tier_2_topics, key=lambda x: x['count'], reverse=True)[:5]:
            doc_lines.append(f"- **{topic['topic_label']}**: Based on {topic['count']} community discussions")
        doc_lines.append("")

        # Methodology
        doc_lines.append("## 🔬 Methodology")
        doc_lines.append("")
        doc_lines.append("### Data Collection")
        doc_lines.append("- Reddit posts and comments from mindfulness-related subreddits")
        doc_lines.append("- Content filtered for quality (minimum 5 words)")
        doc_lines.append("- ALL posts included + top comments by community engagement")
        doc_lines.append("- Preserved original context and community engagement metrics")
        doc_lines.append("")

        doc_lines.append("### Analysis Approach")
        doc_lines.append("- **Topic Modeling**: BERTopic with optimized hierarchical clustering")
        doc_lines.append("- **Semantic Analysis**: Sentence transformers for embedding generation")
        doc_lines.append("- **Tiered Classification**: Based on community engagement and volume")
        doc_lines.append("- **Application Focus**: Emphasis on user needs and pain points")
        doc_lines.append("- **Comprehensive Coverage**: All posts analyzed for complete insights")
        doc_lines.append("")

        # Processing Information
        processing_info = self.topic_analysis.get('processing_info', {})
        if processing_info:
            doc_lines.append("### Processing Details")
            doc_lines.append(f"- **Method**: {processing_info.get('method', 'Fast sampling')}")
            doc_lines.append(f"- **Generated**: {processing_info.get('timestamp', 'N/A')}")
            doc_lines.append("")

        # Save the document
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(doc_lines))

        logger.info(f"Comprehensive document saved as {output_filename}")

        return output_filename


def main():
    """Main function to generate comprehensive summary document."""

    # Configuration
    db_config = {
        'host': 'localhost',
        'port': 3306,
        'database': 'reddit_mindfulness',
        'user': 'root',
        'password': 'admin123',
        'charset': 'utf8mb4',
        'use_unicode': True
    }

    # Initialize generator
    generator = MindfulnessSummaryGenerator(db_config)

    try:
        # Auto-find the latest topic discovery files
        logger.info("Looking for latest topic discovery files...")
        files = generator.find_latest_files()

        if not files['analysis'] or not files['mappings']:
            print("❌ Could not find topic discovery files!")
            print("Please make sure you have run the topic discovery script first.")
            print("Expected files:")
            print("  - mindfulness_topics_fast_analysis_*.json")
            print("  - mindfulness_topics_fast_document_mappings_*.json")
            return

        print(f"✅ Found topic discovery files:")
        print(f"   📊 Analysis: {files['analysis']}")
        print(f"   🗂️  Mappings: {files['mappings']}")
        if files['model']:
            print(f"   🤖 Model: {files['model']}")

        # Load topic analysis results
        logger.info("Loading topic analysis results...")
        if not generator.load_topic_analysis(files['analysis'], files['mappings'], files['model']):
            print("❌ Failed to load topic analysis files")
            return

        # Generate comprehensive document
        logger.info("Generating comprehensive summary document...")
        output_file = generator.generate_comprehensive_document()

        print(f"\n✅ Comprehensive analysis document generated successfully!")
        print(f"📄 Document saved as: {output_file}")
        print(f"📊 File size: {os.path.getsize(output_file) / 1024:.1f} KB")

        # Display document info
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.count('\n')
            words = len(content.split())

        print(f"📋 Document statistics:")
        print(f"   • Lines: {lines:,}")
        print(f"   • Words: {words:,}")
        print(f"   • Characters: {len(content):,}")
        print(f"\n🎯 Your comprehensive mindfulness application analysis is ready!")

    except Exception as e:
        logger.error(f"Document generation failed: {e}")
        raise
    finally:
        generator.close_connection()


if __name__ == "__main__":
    main()