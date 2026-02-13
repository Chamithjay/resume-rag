from typing import List

from typing import List


class ChunkService:
    def __init__(self, chunk_size: int = 30, overlap: int = 5):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))

            # Move start forward by chunk_size - overlap
            start += self.chunk_size - self.overlap

        print(f"\nSplit into {len(chunks)} chunks")
        if chunks:
            print(f"First chunk preview:\n{chunks[0][:200]}...\n")  # Preview first 200 chars

        return chunks
