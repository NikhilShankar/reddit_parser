#!/usr/bin/env python3
"""
Simple diagnostic script to test Weaviate connection
"""

def test_basic_connection():
    """Test basic HTTP connectivity"""
    print("=== Basic HTTP Test ===")
    try:
        import requests
        url = "http://localhost:6060/v1/meta"
        print(f"Testing: {url}")
        
        response = requests.get(url, timeout=5)
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ HTTP test failed: {e}")
        return False

def test_weaviate_client():
    """Test Weaviate Python client"""
    print("\n=== Weaviate Client Test ===")
    try:
        import weaviate
        print("✅ Weaviate package imported successfully")
        
        # Test different client initialization methods
        configs_to_test = [
            {"url": "http://localhost:6060"},
            {"url": "localhost:6060"},
            {"url": "http://127.0.0.1:6060"},
        ]
        
        for i, config in enumerate(configs_to_test, 1):
            print(f"\nTest {i}: {config}")
            try:
                client = weaviate.Client(**config)
                print("  ✅ Client created")
                
                ready = client.is_ready()
                print(f"  ✅ is_ready(): {ready}")
                
                if ready:
                    meta = client.get_meta()
                    print(f"  ✅ Meta: {meta.get('version', 'unknown')}")
                    return client
                else:
                    print("  ❌ Client not ready")
                    
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                
        return None
        
    except ImportError:
        print("❌ Weaviate package not installed")
        print("Run: pip install weaviate-client")
        return None
    except Exception as e:
        print(f"❌ Weaviate client test failed: {e}")
        return None

def test_weaviate_operations(client):
    """Test basic Weaviate operations"""
    print("\n=== Weaviate Operations Test ===")
    try:
        # Test schema operations
        existing_classes = client.schema.get()
        print(f"✅ Existing classes: {len(existing_classes.get('classes', []))}")
        
        # Test simple query
        result = client.query.get("NonExistentClass").do()
        print("✅ Query executed (even for non-existent class)")
        
        return True
    except Exception as e:
        print(f"❌ Operations test failed: {e}")
        return False

def main():
    print("Weaviate Connection Diagnostic Tool")
    print("=" * 50)
    
    # Check if we can reach Weaviate via HTTP
    if not test_basic_connection():
        print("\n❌ Basic HTTP connection failed")
        print("Please check:")
        print("1. Is Weaviate running? (docker ps)")
        print("2. Is port 6060 accessible? (netstat -an | grep 6060)")
        print("3. Try: curl http://localhost:6060/v1/meta")
        return
    
    # Test Python client
    client = test_weaviate_client()
    if not client:
        print("\n❌ Weaviate client connection failed")
        print("This might be a client library issue.")
        print("Try: pip install --upgrade weaviate-client")
        return
    
    # Test operations
    if test_weaviate_operations(client):
        print("\n🎉 All tests passed! Weaviate is working correctly.")
    else:
        print("\n⚠️ Basic connection works but operations failed")

if __name__ == "__main__":
    main()