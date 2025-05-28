import streamlit as st
import weaviate
import weaviate.classes.config as wvc
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
from typing import List, Dict, Any, Optional
import logging
from weaviate.classes.query import MetadataQuery


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Mindfulness Chat Assistant",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MindfulnessChatbot:
    def __init__(self):
        self.weaviate_client = None
        self.embedding_model = None
        self.collection_name = "MindfulnessContent"
        self.ollama_model = "gemma:2b"
        self.embedding_model_name = "all-MiniLM-L12-v2"
        
    @st.cache_resource
    def load_embedding_model(_self):
        """Load and cache the embedding model."""
        try:
            model = SentenceTransformer(_self.embedding_model_name)
            return model
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            return None
    
    def connect_to_weaviate(self):
        """Connect to Weaviate instance."""
        try:
            client = weaviate.connect_to_local(host="localhost", port=6060)
            if client.is_ready():
                return client
            else:
                st.error("Weaviate is not ready")
                return None
        except Exception as e:
            st.error(f"Failed to connect to Weaviate: {e}")
            return None
    
    def test_ollama_connection(self):
        """Test if Ollama is running and model is available."""
        try:
            # Test API connection
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                return False, "Ollama API not responding"
            
            # Check if our model is available
            models = response.json()
            available_models = [model['name'] for model in models['models']]
            
            if self.ollama_model not in available_models:
                return False, f"Model {self.ollama_model} not found. Available: {available_models}"
            
            return True, "Ollama connection successful"
            
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Ollama. Is it running?"
        except Exception as e:
            return False, f"Error checking Ollama: {e}"
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a text."""
        if not self.embedding_model:
            self.embedding_model = self.load_embedding_model()
            if not self.embedding_model:
                return None
        
        try:
            embedding = self.embedding_model.encode([text], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            st.error(f"Failed to generate embedding: {e}")
            return None
    
    def search_mindfulness_content(self, query: str, limit: int = 5, distance_threshold: float = 0.7) -> List[Dict]:
        """Search for relevant mindfulness content in Weaviate."""
        if not self.weaviate_client:
            self.weaviate_client = self.connect_to_weaviate()
            if not self.weaviate_client:
                return []
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            return []
        
        try:
            collection = self.weaviate_client.collections.get(self.collection_name)
            
            response = collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=limit,
                return_metadata=MetadataQuery(distance=True)
            )
            
            # Filter by distance threshold and format results
            results = []
            for obj in response.objects:
                distance = obj.metadata.distance if obj.metadata else 1.0
                
                # Only include results below distance threshold (more similar = lower distance)
                if distance <= distance_threshold:
                    result = {
                        'content': obj.properties.get('content', ''),
                        'chunk_id': obj.properties.get('chunk_id', ''),
                        'level': obj.properties.get('level', 0),
                        'content_type': obj.properties.get('content_type', ''),
                        'post_id': obj.properties.get('post_id', ''),
                        'author': obj.properties.get('author', ''),
                        'score': obj.properties.get('score', 0),
                        'title': obj.properties.get('title', ''),
                        'permalink': obj.properties.get('permalink', ''),
                        'distance': distance,
                        'relevance': 1 - distance  # Convert distance to relevance score
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"Failed to search Weaviate: {e}")
            return []
    
    def format_context(self, search_results: List[Dict]) -> str:
        """Format search results into context for the LLM."""
        if not search_results:
            return ""
        
        context_parts = []
        context_parts.append("Here is relevant mindfulness content from the Reddit community:\n")
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"--- Source {i} ---")
            context_parts.append(f"Type: {result['content_type']}")
            if result['title']:
                context_parts.append(f"Title: {result['title']}")
            context_parts.append(f"Author: {result['author']}")
            context_parts.append(f"Community Score: {result['score']}")
            context_parts.append(f"Relevance: {result['relevance']:.2f}")
            context_parts.append(f"Content: {result['content']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> tuple[str, bool]:
        """Generate response using Ollama with context."""
        # Check if query is mindfulness-related by checking if we found relevant context
        if not context.strip():
            return "I can only help with mindfulness and meditation topics. I don't have information about your question in my mindfulness knowledge base.", False
        
        # Create system prompt
        system_prompt = """You are a helpful mindfulness and meditation assistant. You ONLY answer questions about mindfulness, meditation, mental health, and related wellness topics.

STRICT RULES:
1. ONLY use the provided context from the Reddit mindfulness community to answer questions
2. If the question is not about mindfulness/meditation, respond: "I can only help with mindfulness and meditation topics."
3. If no relevant context is provided, respond: "I don't have information about that specific mindfulness topic."
4. Always be helpful, compassionate, and supportive
5. Cite your sources when providing advice
6. Do not make up information not in the context

Context from mindfulness community:
{context}

Question: {query}

Answer based only on the provided context:"""
        
        prompt = system_prompt.format(context=context, query=query)
        
        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            response = requests.post(url, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip(), True
            else:
                return "Sorry, I'm having trouble generating a response right now.", False
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I'm having trouble connecting to the AI model.", False
    
    def save_chat_as_markdown(self, messages: List[Dict], filename: str = None):
        """Save chat history as markdown file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mindfulness_chat_{timestamp}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Mindfulness Chat Session\n\n")
                f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                
                for msg in messages:
                    if msg['role'] == 'user':
                        f.write(f"## 🙋 **You**\n\n{msg['content']}\n\n")
                    else:
                        f.write(f"## 🧘 **Mindfulness Assistant**\n\n{msg['content']}\n\n")
                        
                        # Add sources if available
                        if 'sources' in msg:
                            f.write("### 📚 **Sources**\n\n")
                            for i, source in enumerate(msg['sources'], 1):
                                f.write(f"**Source {i}** (Relevance: {source['relevance']:.2f})\n")
                                f.write(f"- **Type**: {source['content_type']}\n")
                                if source['title']:
                                    f.write(f"- **Title**: {source['title']}\n")
                                f.write(f"- **Author**: {source['author']}\n")
                                f.write(f"- **Score**: {source['score']}\n")
                                f.write(f"- **Content**: {source['content'][:200]}...\n\n")
                        
                        f.write("---\n\n")
            
            return filename
        except Exception as e:
            st.error(f"Failed to save chat: {e}")
            return None

def main():
    st.title("🧘 Mindfulness Chat Assistant")
    st.markdown("*Ask me anything about mindfulness and meditation based on Reddit community wisdom*")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MindfulnessChatbot()
    
    chatbot = st.session_state.chatbot
    
    # Sidebar with system status
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Test Ollama connection
        ollama_status, ollama_msg = chatbot.test_ollama_connection()
        if ollama_status:
            st.success(f"✅ Ollama: {ollama_msg}")
        else:
            st.error(f"❌ Ollama: {ollama_msg}")
        
        # Test Weaviate connection
        weaviate_client = chatbot.connect_to_weaviate()
        if weaviate_client:
            st.success("✅ Weaviate: Connected")
            if chatbot.weaviate_client is None:
                chatbot.weaviate_client = weaviate_client
        else:
            st.error("❌ Weaviate: Connection failed")
        
        # Test embedding model
        embedding_model = chatbot.load_embedding_model()
        if embedding_model:
            st.success("✅ Embeddings: Model loaded")
            if chatbot.embedding_model is None:
                chatbot.embedding_model = embedding_model
        else:
            st.error("❌ Embeddings: Failed to load")
        
        st.markdown("---")
        
        # Settings
        st.header("⚙️ Settings")
        search_limit = st.slider("Max search results", 1, 10, 5)
        relevance_threshold = st.slider("Relevance threshold", 0.1, 1.0, 0.7, 0.1)
        
        st.markdown("---")
        
        # Save chat button
        if st.button("💾 Save Chat as Markdown"):
            if 'messages' in st.session_state and st.session_state.messages:
                filename = chatbot.save_chat_as_markdown(st.session_state.messages)
                if filename:
                    st.success(f"Chat saved as {filename}")
                    # Provide download button
                    with open(filename, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="📥 Download Chat",
                            data=f.read(),
                            file_name=filename,
                            mime="text/markdown"
                        )
            else:
                st.info("No chat history to save")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = """Hello! I'm your mindfulness assistant. I can help you with questions about meditation, mindfulness practices, dealing with stress and anxiety, and other wellness topics based on wisdom from the Reddit mindfulness community.

Ask me anything about:
- Meditation techniques
- Mindfulness practices  
- Dealing with anxiety or racing thoughts
- Breathing exercises
- Starting a meditation practice
- Mindfulness in daily life

What would you like to know about mindfulness today?"""
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": welcome_msg
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander(f"📚 View Sources ({len(message['sources'])})"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** - Relevance: {source['relevance']:.2f}")
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            st.markdown(f"**Type:** {source['content_type']}")
                            st.markdown(f"**Author:** {source['author']}")
                            st.markdown(f"**Score:** {source['score']}")
                            if source['title']:
                                st.markdown(f"**Title:** {source['title']}")
                        
                        with col2:
                            st.markdown("**Full Content:**")
                            st.text_area(
                                f"Content from source {i}", 
                                source['content'], 
                                height=150,
                                key=f"source_{i}_{hash(source['content'])}"
                            )
                        
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask about mindfulness and meditation..."):
        # Check system status before processing
        if not ollama_status:
            st.error("❌ Ollama is not available. Please check the sidebar for details.")
            return
        
        if not chatbot.weaviate_client:
            st.error("❌ Weaviate is not available. Please check the sidebar for details.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching mindfulness knowledge base..."):
                # Search for relevant content
                search_results = chatbot.search_mindfulness_content(
                    prompt, 
                    limit=search_limit, 
                    distance_threshold=relevance_threshold
                )
                
                # Format context
                context = chatbot.format_context(search_results)
                
                # Generate response
                response, success = chatbot.generate_response(prompt, context)
                
                # Display response
                st.markdown(response)
                
                # Store assistant message with sources
                assistant_msg = {
                    "role": "assistant", 
                    "content": response
                }
                
                if search_results:
                    assistant_msg["sources"] = search_results
                    
                    # Display sources
                    with st.expander(f"📚 View Sources ({len(search_results)})"):
                        for i, source in enumerate(search_results, 1):
                            st.markdown(f"**Source {i}** - Relevance: {source['relevance']:.2f}")
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                st.markdown(f"**Type:** {source['content_type']}")
                                st.markdown(f"**Author:** {source['author']}")
                                st.markdown(f"**Score:** {source['score']}")
                                if source['title']:
                                    st.markdown(f"**Title:** {source['title']}")
                            
                            with col2:
                                st.markdown("**Full Content:**")
                                st.text_area(
                                    f"Content from source {i}", 
                                    source['content'], 
                                    height=150,
                                    key=f"new_source_{i}_{hash(source['content'])}"
                                )
                            
                            st.markdown("---")
                
                st.session_state.messages.append(assistant_msg)

if __name__ == "__main__":
    main()