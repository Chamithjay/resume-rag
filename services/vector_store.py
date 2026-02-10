from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Optional
import os
import time


class VectorStoreService:
    def __init__(self, persist_directory: str, api_key: str):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        print(f"🔧 Initializing Google Embeddings...")

        # Use Google's embedding model with timeout settings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
            request_options={
                "timeout": 60  # 60 seconds timeout
            }
        )

        print(f"🔧 Initializing ChromaDB at {persist_directory}...")

        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="resumes"
        )

        print(f"✅ Vector store initialized successfully")

    def add_documents(self, chunks: List[Dict]) -> int:
        """Add document chunks to vector store"""
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        print(f"📊 Adding {len(texts)} text chunks to vector store...")

        try:
            start_time = time.time()

            # Add texts in batches to avoid timeout
            batch_size = 5
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]

                print(f"   Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}...")

                self.vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metas
                )

            elapsed = time.time() - start_time
            print(f"✅ Added {len(texts)} chunks in {elapsed:.2f}s")

            return len(texts)

        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            raise Exception(f"Failed to add documents to vector store: {str(e)}")

    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents"""
        print(f"🔍 Searching for: '{query}' (top {k} results)")

        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )

            print(f"✅ Found {len(results)} results")

            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            print(f"❌ Error searching: {str(e)}")
            raise Exception(f"Search failed: {str(e)}")

    def delete_by_filename(self, filename: str):
        """Delete all chunks for a specific file"""
        try:
            print(f"🗑️ Deleting chunks for: {filename}")
            collection = self.vectorstore._collection
            results = collection.get(where={"filename": filename})

            if results and results['ids']:
                collection.delete(ids=results['ids'])
                print(f"✅ Deleted {len(results['ids'])} chunks")
            else:
                print(f"⚠️ No chunks found for {filename}")
        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
            raise Exception(f"Delete failed: {str(e)}")