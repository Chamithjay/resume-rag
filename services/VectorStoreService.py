import os
from pinecone import Pinecone, ServerlessSpec


class VectorStoreService:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = "resumes"

        if not api_key:
            raise ValueError("Missing PINECONE_API_KEY")

        self.pc = Pinecone(api_key=api_key)

        # Create index if not exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            print("Creating Pinecone index...")

            self.pc.create_index(
                name=index_name,
                dimension=768,  # must match Gemini embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"   # choose same as your Pinecone project
                )
            )
            print("Index created")

        # Connect to index
        self.index = self.pc.Index(index_name)

    # Store embeddings
    def store_embeddings(self, embeddings, chunks, filename, candidate_name):
        vectors = []

        for i, emb in enumerate(embeddings):
            vectors.append({
                "id": f"{filename}_{i}",
                "values": emb,   # Gemini embedding values
                "metadata": {
                    "text": chunks[i],
                    "filename": filename,
                    "candidate_name": candidate_name
                }
            })

        self.index.upsert(vectors=vectors)

        print(f"Stored {len(vectors)} vectors in Pinecone")

    # Count vectors (debug)
    def count(self):
        stats = self.index.describe_index_stats()
        return stats["total_vector_count"]
