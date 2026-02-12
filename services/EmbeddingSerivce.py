from google import genai
from google.genai import types



class EmbeddingService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def embed_chunks(self,chunks: list[str], model="gemini-embedding-001", dim=768):
        embeddings = []

        for i, chunk in enumerate(chunks, 1):
            print(f"Embedding chunk {i}/{len(chunks)}...")
            result = self.client.models.embed_content(
                model=model,
                contents=[chunk],
                config=types.EmbedContentConfig(output_dimensionality=dim)
            )
            embeddings.append(result.embeddings[0].values)  # single chunk embedding
        if embeddings:
            print("First chunk embedding (first 10 values):")
            print(embeddings[0][:10])

        return embeddings
