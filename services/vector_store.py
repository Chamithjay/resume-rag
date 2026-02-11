from langchain_core.embeddings import Embeddings
from typing import List, Dict, Optional
import os
import time
import traceback
import uuid
import google.generativeai as genai
import chromadb
from chromadb.config import Settings


class GoogleEmbeddingsWrapper(Embeddings):
    """Custom wrapper for Google Gemini embeddings that works reliably"""

    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001"):
        self.model = model
        genai.configure(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        for i, text in enumerate(texts):
            print(f"      → Embedding text {i+1}/{len(texts)}...")
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
            # Small delay to avoid rate limiting
            if i < len(texts) - 1:  # Don't delay after last one
                time.sleep(0.1)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query"""
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']


class VectorStoreService:
    def __init__(self, persist_directory: str, api_key: str):
        self.persist_directory = persist_directory
        self.api_key = api_key
        os.makedirs(persist_directory, exist_ok=True)

        print("🔧 Initializing Google Embeddings...")

        # Use custom wrapper that works reliably with Google AI
        self.embeddings = GoogleEmbeddingsWrapper(api_key=api_key)

        print(f"🔧 Initializing ChromaDB at {persist_directory}...")

        # Use ChromaDB directly (more reliable than LangChain wrapper)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="resumes",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        print("✅ Vector store initialized successfully")

    def add_documents(self, chunks: List[Dict]) -> int:
        """Add document chunks to vector store using native ChromaDB API"""
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        print(f"📊 Adding {len(texts)} text chunks to vector store...")

        try:
            start_time = time.time()

            # Generate all embeddings first
            print(f"   Generating embeddings for all {len(texts)} chunks...")
            embeddings = self.embeddings.embed_documents(texts)

            # Generate IDs
            ids = [str(uuid.uuid4()) for _ in texts]

            # Add to ChromaDB in one call (native API is more reliable)
            print(f"   Storing in ChromaDB...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            elapsed = time.time() - start_time
            print(f"✅ Added {len(texts)} chunks in {elapsed:.2f}s")

            return len(texts)

        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Failed to add documents to vector store: {str(e)}")

    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents using native ChromaDB API"""
        print(f"🔍 Searching for: '{query}' (top {k} results)")

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search using native ChromaDB API
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict if filter_dict else None
            )

            print(f"✅ Found {len(results['documents'][0])} results")

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                    })

            return formatted_results

        except Exception as e:
            print(f"❌ Error searching: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Search failed: {str(e)}")

    def delete_by_filename(self, filename: str):
        """Delete all chunks for a specific file"""
        try:
            print(f"🗑️ Deleting chunks for: {filename}")

            # Query for all documents with this filename
            results = self.collection.get(where={"filename": filename})

            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"✅ Deleted {len(results['ids'])} chunks")
            else:
                print(f"⚠️ No chunks found for {filename}")
        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
            traceback.print_exc()
            raise Exception(f"Delete failed: {str(e)}")