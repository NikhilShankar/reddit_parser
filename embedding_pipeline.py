import json
import logging
import numpy as np
import weaviate
import weaviate.classes.config as wvc
import weaviate.classes.data as wvd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from weaviate.classes.query import MetadataQuery


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        """
        Initialize embedding generator with specified model.
        
        Recommended models:
        - all-MiniLM-L12-v2: Good balance of speed/quality
        - all-mpnet-base-v2: Higher quality, slower
        - paraphrase-multilingual-MiniLM-L12-v2: Multilingual support
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
    def load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension
            test_embedding = self.model.encode("test")
            self.embedding_dim = len(test_embedding)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.model:
            self.load_model()
            
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Process in batches to manage memory
        all_embeddings = []
        
        if show_progress:
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            for batch in tqdm(batches, desc="Generating embeddings"):
                embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                all_embeddings.append(embeddings)
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                all_embeddings.append(embeddings)
        logger.info(f"Completed generating embeddings for query texts")
        return np.vstack(all_embeddings)

def test_weaviate_connection(host="localhost", port=6060):
    """Simple function to test Weaviate v4 connection."""
    print(f"Testing Weaviate v4 connection to {host}:{port}")
    
    # Test HTTP connectivity first
    try:
        import requests
        url = f"http://{host}:{port}/v1/meta"
        print(f"Testing HTTP request to: {url}")
        response = requests.get(url, timeout=10)
        print(f"HTTP Response Status: {response.status_code}")
        print(f"HTTP Response Content: {response.json()}")
    except Exception as e:
        print(f"HTTP request failed: {e}")
        return False
    
    # Test v4 client
    try:
        print("Testing Weaviate v4 client...")
        client = weaviate.connect_to_local(host=host, port=port)
        print("Client created successfully")
        
        ready = client.is_ready()
        print(f"is_ready() result: {ready}")
        
        if ready:
            print("✅ Weaviate v4 connection successful!")
            client.close()
            return True
        else:
            print("❌ Client reports not ready")
            client.close()
            return False
            
    except Exception as e:
        print(f"Weaviate v4 client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the embedding pipeline."""
    
    # Configuration
    CHUNKS_FILE = "hierarchical_chunks.json"  # or "hierarchical_chunks_full.json"
    WEAVIATE_HOST = "localhost"
    WEAVIATE_PORT = 6060
    EMBEDDING_MODEL = "all-MiniLM-L12-v2"
    COLLECTION_NAME = "MindfulnessContent"
    
    # First, test the connection
    print("=== Testing Weaviate v4 Connection ===")
    if not test_weaviate_connection(WEAVIATE_HOST, WEAVIATE_PORT):
        print("Connection test failed. Please check your Weaviate setup.")
        return
    
    # Check if chunks file exists
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        logger.info("Please run the data extraction script first")
        return
    
    # Load chunks
    logger.info(f"Loading chunks from {CHUNKS_FILE}")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL)
    embedding_generator.load_model()
    
    # Connect to Weaviate
    logger.info(f"Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}")
    try:
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
            grpc_port=50051
        )
        
        if not client.is_ready():
            # Fallback without gRPC port
            client = weaviate.connect_to_local(
                host=WEAVIATE_HOST,
                port=WEAVIATE_PORT
            )
            
        if not client.is_ready():
            raise Exception("Weaviate is not ready")
            
        logger.info("Successfully connected to Weaviate")
        
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        return
    
    try:
        # Delete existing collection if it exists
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        
        # Create collection
        logger.info(f"Creating collection: {COLLECTION_NAME}")
        collection = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=wvc.Configure.Vectorizer.none(),  # We provide our own vectors
            properties=[
                wvc.Property(
                    name="content",
                    data_type=wvc.DataType.TEXT,
                    description="The main text content"
                ),
                wvc.Property(
                    name="chunk_id", 
                    data_type=wvc.DataType.TEXT,
                    description="Unique identifier for the chunk"
                ),
                wvc.Property(
                    name="level",
                    data_type=wvc.DataType.INT,
                    description="Hierarchical level (1, 2, or 3)"
                ),
                wvc.Property(
                    name="content_type",
                    data_type=wvc.DataType.TEXT, 
                    description="Type of content (post_with_comments, comment_with_context, high_value_comment)"
                ),
                wvc.Property(
                    name="post_id",
                    data_type=wvc.DataType.TEXT,
                    description="Original Reddit post ID"
                ),
                wvc.Property(
                    name="author",
                    data_type=wvc.DataType.TEXT,
                    description="Author username"
                ),
                wvc.Property(
                    name="score",
                    data_type=wvc.DataType.INT,
                    description="Reddit score/upvotes"
                ),
                wvc.Property(
                    name="created_utc",
                    data_type=wvc.DataType.DATE,
                    description="Creation timestamp"
                ),
                wvc.Property(
                    name="title",
                    data_type=wvc.DataType.TEXT,
                    description="Post title (for post-level content)"
                ),
                wvc.Property(
                    name="num_comments",
                    data_type=wvc.DataType.INT,
                    description="Number of comments in post"
                ),
                wvc.Property(
                    name="permalink", 
                    data_type=wvc.DataType.TEXT,
                    description="Reddit permalink"
                ),
                wvc.Property(
                    name="embedding_model",
                    data_type=wvc.DataType.TEXT,
                    description="Model used for embedding generation"
                ),
                wvc.Property(
                    name="processed_at",
                    data_type=wvc.DataType.DATE, 
                    description="When this was processed into vector DB"
                )
            ]
        )
        logger.info(f"Collection created successfully")
        
        # Extract texts for embedding
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = embedding_generator.generate_embeddings(texts)
        
        # Prepare data objects
        logger.info("Preparing data objects...")
        data_objects = []
        
        for chunk, embedding in tqdm(zip(chunks, embeddings), total=len(chunks), desc="Preparing data objects"):
            # Prepare properties
            properties = {
                "content": chunk.get('content', ''),
                "chunk_id": chunk.get('id', ''),
                "level": chunk.get('level', 0),
                "content_type": chunk.get('metadata', {}).get('content_type', ''),
                "post_id": chunk.get('metadata', {}).get('post_id', ''),
                "author": chunk.get('metadata', {}).get('author', ''),
                "score": chunk.get('metadata', {}).get('score', 0),
                "title": chunk.get('metadata', {}).get('title', ''),
                "num_comments": chunk.get('metadata', {}).get('num_comments', 0),
                "permalink": chunk.get('metadata', {}).get('permalink', ''),
                "embedding_model": EMBEDDING_MODEL,
                "processed_at": datetime.now()
            }
            
            # Handle datetime conversion for created_utc
            created_utc = chunk.get('metadata', {}).get('created_utc')
            if created_utc:
                if isinstance(created_utc, str):
                    try:
                        properties["created_utc"] = datetime.fromisoformat(created_utc.replace('Z', '+00:00'))
                    except:
                        properties["created_utc"] = datetime.now()
                else:
                    properties["created_utc"] = created_utc if hasattr(created_utc, 'year') else datetime.now()
            else:
                properties["created_utc"] = datetime.now()
            
            # Create data object with vector
            data_object = wvd.DataObject(
                properties=properties,
                vector=embedding.tolist()
            )
            data_objects.append(data_object)
        
        # Insert all objects into Weaviate
        logger.info(f"Inserting {len(data_objects)} objects into Weaviate...")
        response = collection.data.insert_many(data_objects)
        
        # Check for errors
        failed_insertions = []
        if hasattr(response, 'errors') and response.errors:
            for error in response.errors:
                failed_insertions.append(str(error))
                logger.warning(f"Insertion error: {error}")
        
        total_inserted = len(chunks) - len(failed_insertions)
        
        print("\n=== Processing Results ===")
        print(f"Total chunks processed: {len(chunks)}")
        print(f"Successfully inserted: {total_inserted}")
        print(f"Failed insertions: {len(failed_insertions)}")
        
        # Get collection statistics
        try:
            total_response = collection.aggregate.over_all(total_count=True)
            total_objects = total_response.total_count
            
            # Get level distribution
            level_stats = {}
            for level in [1, 2, 3]:
                level_response = collection.aggregate.over_all(
                    total_count=True,
                    where=wvc.query.Filter.by_property("level").equal(level)
                )
                level_stats[level] = level_response.total_count
            
            print(f"\n=== Collection Statistics ===")
            print(f"Total objects in Weaviate: {total_objects}")
            print("Level distribution:")
            for level, count in level_stats.items():
                print(f"  Level {level}: {count} objects")
                
        except Exception as e:
            logger.warning(f"Could not get collection stats: {e}")
        
        # Test searches
        test_queries = [
            "How to deal with anxiety during meditation?",
            "Breathing techniques for mindfulness",
            "Racing thoughts while meditating"
        ]
        
        print("\n=== Testing Search Functionality ===")
        for query in test_queries:
            logger.info(f"Testing search with query: '{query}'")
            
            try:
                # Generate embedding for query
                query_embedding = embedding_generator.generate_embeddings([query], show_progress=False)[0]
                current_collection = client.collections.get(COLLECTION_NAME)
                # Perform search
                try:
                    search_response = current_collection.query.near_vector(
                        near_vector=query_embedding.tolist(),
                        limit=3,
                        return_metadata=MetadataQuery(distance=True)
                    )
                except Exception as e:
                    logger.error(f"Search failed Niks : ${collection} {e}")
                
                print(f"\n=== Search Results for: '{query}' === with {len(search_response.objects)} results")
                for i, obj in enumerate(search_response.objects, 1):
                    distance = obj.metadata.distance if obj.metadata else 'N/A'
                    print(f"\n{i}. [Level {obj.properties.get('level', 'N/A')}] Distance: {distance}")
                    print(f"   Type: {obj.properties.get('content_type', 'N/A')}")
                    print(f"   Score: {obj.properties.get('score', 'N/A')}")
                    if obj.properties.get('title'):
                        print(f"   Title: {obj.properties.get('title')}")
                    print(f"   Content: {obj.properties.get('content', '')[:200]}...")
                    print(f"   ID: {obj.properties.get('chunk_id', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Search test failed for query '{query}': {e}")
            
            print("-" * 60)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        # Close client connection
        client.close()
        logger.info("Weaviate client connection closed")

if __name__ == "__main__":
    main()