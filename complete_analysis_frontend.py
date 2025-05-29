import streamlit as st
import pandas as pd
import subprocess
import sys
import os
import json
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Reddit Analysis Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def run_script(script_name, args=None):
    """Execute a Python script and return the result"""
    try:
        cmd = [sys.executable, script_name]
        if args:
            cmd.extend(args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd()
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except Exception as e:
        return False, "", str(e)


def check_file_exists(filename):
    """Check if a file exists in the current directory"""
    return os.path.exists(filename)


def load_json_file(filename):
    """Load and return JSON file content"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None


def load_csv_file(filename):
    """Load and return CSV file content"""
    try:
        return pd.read_csv(filename)
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Reddit Analysis Pipeline</h1>', unsafe_allow_html=True)

    # Sidebar for inputs
    st.sidebar.header("📋 Configuration")

    # Input parameters
    subreddit_name = st.sidebar.text_input(
        "Subreddit Name",
        value="MachineLearning",
        help="Enter subreddit name without 'r/' prefix"
    )

    num_posts = st.sidebar.slider(
        "Number of Posts",
        min_value=5,
        max_value=100000,
        value=1000,
        help="Number of posts to extract from the subreddit"
    )

    num_comments = st.sidebar.slider(
        "Comments per Post",
        min_value=5,
        max_value=500,
        value=10,
        help="Number of comments to extract from each post"
    )

    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        post_type = st.selectbox(
            "Post Type",
            ["hot", "new", "top", "rising"],
            index=0,
            help="Type of posts to fetch"
        )

        time_filter = st.selectbox(
            "Time Filter (for 'top' posts)",
            ["day", "week", "month", "year", "all"],
            index=2
        )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🚀 Pipeline Overview")
        st.markdown("""
        This application will execute your Reddit analysis pipeline in the following order:
        1. **Data Collection** - Extract posts and comments from Reddit
        2. **Data Processing** - Clean and structure the data
        3. **Sentiment Analysis** - Analyze sentiment of posts and comments
        4. **Topic Modeling** - Identify key topics and themes
        5. **Clustering** - Group similar content together
        6. **Summary Generation** - Create comprehensive insights
        """)

    with col2:
        st.markdown("### 📊 Current Settings")
        st.info(f"""
        **Subreddit:** r/{subreddit_name}
        **Posts:** {num_posts}
        **Comments per post:** {num_comments}
        **Post type:** {post_type}
        **Time filter:** {time_filter}
        """)

    # Start analysis button
    if st.button("🔄 Start Reddit Analysis", type="primary", use_container_width=True):
        if not subreddit_name:
            st.error("Please enter a subreddit name!")
            return

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Results containers
        results_container = st.container()

        with results_container:
            st.markdown("---")
            st.markdown("## 📈 Analysis Results")

            # Step 1: Data Collection
            with st.expander("📥 Step 1: Data Collection", expanded=True):
                status_text.text("🔄 Collecting Reddit data...")
                progress_bar.progress(10)

                # Assuming your reddit scraper script exists
                success, stdout, stderr = run_script("reddit_scraper.py", [
                    "--subreddit", subreddit_name,
                    "--posts", str(num_posts),
                    "--comments", str(num_comments),
                    "--type", post_type
                ])

                if success:
                    st.markdown('<div class="success-box">✅ Data collection completed successfully!</div>',
                                unsafe_allow_html=True)
                    if check_file_exists("reddit_data.json"):
                        data = load_json_file("reddit_data.json")
                        if data:
                            st.write(f"📊 Collected {len(data.get('posts', []))} posts")
                            # Show a sample
                            if data.get('posts'):
                                st.write("Sample post:")
                                st.json(data['posts'][0])
                else:
                    st.markdown(f'<div class="error-box">❌ Data collection failed: {stderr}</div>',
                                unsafe_allow_html=True)
                    st.stop()

            # Step 2: Data Processing
            with st.expander("🔧 Step 2: Data Processing", expanded=True):
                status_text.text("🔄 Processing and cleaning data...")
                progress_bar.progress(25)

                success, stdout, stderr = run_script("data_processor.py")

                if success:
                    st.markdown('<div class="success-box">✅ Data processing completed!</div>', unsafe_allow_html=True)
                    if check_file_exists("processed_data.csv"):
                        df = load_csv_file("processed_data.csv")
                        if df is not None:
                            st.write(f"📊 Processed {len(df)} records")
                            st.dataframe(df.head())
                else:
                    st.markdown(f'<div class="error-box">❌ Data processing failed: {stderr}</div>',
                                unsafe_allow_html=True)
                    st.stop()

            # Step 3: Sentiment Analysis
            with st.expander("😊 Step 3: Sentiment Analysis", expanded=True):
                status_text.text("🔄 Analyzing sentiment...")
                progress_bar.progress(40)

                success, stdout, stderr = run_script("sentiment_analyzer.py")

                if success:
                    st.markdown('<div class="success-box">✅ Sentiment analysis completed!</div>',
                                unsafe_allow_html=True)
                    if check_file_exists("sentiment_results.csv"):
                        sentiment_df = load_csv_file("sentiment_results.csv")
                        if sentiment_df is not None:
                            # Create sentiment visualization
                            if 'sentiment' in sentiment_df.columns:
                                sentiment_counts = sentiment_df['sentiment'].value_counts()
                                fig = px.pie(
                                    values=sentiment_counts.values,
                                    names=sentiment_counts.index,
                                    title="Sentiment Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f'<div class="error-box">❌ Sentiment analysis failed: {stderr}</div>',
                                unsafe_allow_html=True)
                    st.stop()

            # Step 4: Topic Modeling
            with st.expander("🏷️ Step 4: Topic Modeling", expanded=True):
                status_text.text("🔄 Identifying topics...")
                progress_bar.progress(60)

                success, stdout, stderr = run_script("topic_modeler.py")

                if success:
                    st.markdown('<div class="success-box">✅ Topic modeling completed!</div>', unsafe_allow_html=True)
                    if check_file_exists("topics.json"):
                        topics_data = load_json_file("topics.json")
                        if topics_data:
                            st.write("📋 Identified Topics:")
                            for i, topic in enumerate(topics_data.get('topics', [])):
                                st.write(f"**Topic {i + 1}:** {', '.join(topic.get('words', []))}")
                else:
                    st.markdown(f'<div class="error-box">❌ Topic modeling failed: {stderr}</div>',
                                unsafe_allow_html=True)
                    st.stop()

            # Step 5: Clustering
            with st.expander("🎯 Step 5: Clustering Analysis", expanded=True):
                status_text.text("🔄 Performing clustering analysis...")
                progress_bar.progress(80)

                success, stdout, stderr = run_script("clustering_analyzer.py")

                if success:
                    st.markdown('<div class="success-box">✅ Clustering analysis completed!</div>',
                                unsafe_allow_html=True)
                    if check_file_exists("clusters.csv"):
                        cluster_df = load_csv_file("clusters.csv")
                        if cluster_df is not None and 'cluster' in cluster_df.columns:
                            cluster_counts = cluster_df['cluster'].value_counts()
                            fig = px.bar(
                                x=cluster_counts.index,
                                y=cluster_counts.values,
                                title="Cluster Distribution",
                                labels={'x': 'Cluster', 'y': 'Count'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f'<div class="error-box">❌ Clustering analysis failed: {stderr}</div>',
                                unsafe_allow_html=True)
                    st.stop()

            # Step 6: Summary Generation
            with st.expander("📝 Step 6: Summary Generation", expanded=True):
                status_text.text("🔄 Generating comprehensive summary...")
                progress_bar.progress(95)

                success, stdout, stderr = run_script("summary_generator.py")

                if success:
                    st.markdown('<div class="success-box">✅ Summary generation completed!</div>',
                                unsafe_allow_html=True)
                    if check_file_exists("final_summary.txt"):
                        with open("final_summary.txt", "r", encoding="utf-8") as f:
                            summary = f.read()
                        st.markdown("### 📊 Final Analysis Summary")
                        st.markdown(summary)

                    if check_file_exists("insights.json"):
                        insights = load_json_file("insights.json")
                        if insights:
                            st.markdown("### 🔍 Key Insights")
                            for insight in insights.get('key_insights', []):
                                st.write(f"• {insight}")
                else:
                    st.markdown(f'<div class="error-box">❌ Summary generation failed: {stderr}</div>',
                                unsafe_allow_html=True)

            # Completion
            progress_bar.progress(100)
            status_text.text("✅ Analysis pipeline completed successfully!")

            # Download section
            st.markdown("---")
            st.markdown("### 💾 Download Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                if check_file_exists("final_summary.txt"):
                    with open("final_summary.txt", "rb") as f:
                        st.download_button(
                            "📄 Download Summary",
                            f.read(),
                            "reddit_analysis_summary.txt",
                            "text/plain"
                        )

            with col2:
                if check_file_exists("processed_data.csv"):
                    df = pd.read_csv("processed_data.csv")
                    st.download_button(
                        "📊 Download Data (CSV)",
                        df.to_csv(index=False),
                        "reddit_processed_data.csv",
                        "text/csv"
                    )

            with col3:
                if check_file_exists("insights.json"):
                    with open("insights.json", "rb") as f:
                        st.download_button(
                            "🔍 Download Insights (JSON)",
                            f.read(),
                            "reddit_insights.json",
                            "application/json"
                        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🔍 Reddit Analysis Pipeline | Built with Streamlit</p>
        <p>Make sure all your analysis scripts are in the same directory as this Streamlit app.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()