from typing import List

class ChunkService:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def split_text(self, text: str) -> List[str]:

        words = text.split()
        chunks = []
        current_chunk = ""

        for word in words:
            # +1 for space
            if len(current_chunk) + len(word) + 1 > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = word + " "
            else:
                current_chunk += word + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        print(f"\nSplit into {len(chunks)} chunks")
        if chunks:
            print(f"First chunk preview:\n{chunks[0][:200]}...\n")  # Preview first 200 chars

        return chunks
