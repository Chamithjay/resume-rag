"""Direct test of Google embeddings API"""

import os
from dotenv import load_dotenv
import time

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print(f"API Key: {api_key[:20]}...\n")

# Test 1: Using google-generativeai directly
print("=" * 60)
print("TEST 1: Direct API Call")
print("=" * 60)

try:
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    print("Calling embed_content...")
    start = time.time()

    result = genai.embed_content(
        model="gemini-embedding-001",
        content="Hello world"
    )

    elapsed = time.time() - start

    print(f"✅ Success! Took {elapsed:.2f}s")
    print(f"   Embedding dimension: {len(result['embedding'])}")

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback

    traceback.print_exc()

# Test 2: Using LangChain wrapper
print("\n" + "=" * 60)
print("TEST 2: LangChain Wrapper")
print("=" * 60)

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    print("Calling embed_query...")
    start = time.time()

    result = embeddings.embed_query("Hello world")

    elapsed = time.time() - start

    print(f"✅ Success! Took {elapsed:.2f}s")
    print(f"   Embedding dimension: {len(result)}")

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback

    traceback.print_exc()

# Test 3: Batch embedding (like your app does)
print("\n" + "=" * 60)
print("TEST 3: Batch Embeddings (5 texts)")
print("=" * 60)

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]

    print(f"Embedding {len(texts)} texts...")
    start = time.time()

    results = embeddings.embed_documents(texts)

    elapsed = time.time() - start

    print(f"✅ Success! Took {elapsed:.2f}s")
    print(f"   Processed {len(results)} embeddings")
    print(f"   Average: {elapsed / len(results):.2f}s per text")

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback

    traceback.print_exc()

print("\n✅ All tests complete!")