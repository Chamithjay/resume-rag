from google import genai
from google.genai import types


class EmbeddingService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def embed_chunks(self, chunks: list[str], model="gemini-embedding-001", dim=768, batch_size=20):
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")

            result = self.client.models.embed_content(
                model=model,
                contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=dim)
            )

            for emb in result.embeddings:
                all_embeddings.append(emb.values)

        if all_embeddings:
            print("First chunk embedding (first 10 values):")
            print(all_embeddings[0][:10])

        return all_embeddings
