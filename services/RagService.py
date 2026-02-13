import os
from dotenv import load_dotenv

load_dotenv()


class RAGService:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY")
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.model = "gemini-2.5-flash-lite"
        except ImportError:
            raise ImportError("Install: pip install google-genai")

    def generate_structured_response(self, query: str, matched_chunks: list) -> dict:
        """
        Generate structured response with candidate information

        Args:
            query: User's question
            matched_chunks: List of relevant chunks with metadata

        Returns:
            Dict with answer and candidate details
        """

        if not matched_chunks:
            return {
                "answer": "No matching candidates found for your query.",
                "candidates": []
            }


        context = self._build_context(matched_chunks)

        prompt = self._create_prompt(query, context)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            answer = response.text
        except Exception as e:
            answer = f"Error generating response: {str(e)}"

        candidates = self._extract_candidates(matched_chunks)

        return {
            "answer": answer,
            "candidates": candidates
        }

    def _build_context(self, chunks: list) -> str:
        """Build context string from chunks"""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            candidate_name = chunk.get("candidate_name", "Unknown")
            filename = chunk.get("filename", "")
            text = chunk.get("text", "")

            context_parts.append(
                f"--- Resume {i}: {candidate_name} ---\n"
                f"File: {filename}\n"
                f"Content: {text}\n"
            )

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """Create structured prompt for LLM"""

        return f"""You are an expert technical recruiter analyzing resumes to find matching candidates.

AVAILABLE RESUME EXCERPTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Analyze each resume excerpt carefully
2. Identify candidates who match the query requirements
3. For each matching candidate, explain WHY they match with specific evidence
4. If no candidates match, state this clearly
5. Be factual and cite specific information from the resumes

FORMAT YOUR RESPONSE AS:

**MATCHING CANDIDATES:**

1. **[Candidate Name]**
   - Relevance: [Why they match the query]
   - Evidence: [Specific quotes or facts from their resume]

2. **[Candidate Name]**
   - Relevance: [Why they match]
   - Evidence: [Specific facts]

**SUMMARY:**
[Brief overview of findings - how many candidates match, key observations]

Now provide your analysis:"""

    def _extract_candidates(self, chunks: list) -> list:
        """Extract unique candidate information"""
        candidates_dict = {}

        for chunk in chunks:
            name = chunk.get("candidate_name", "Unknown")

            if name not in candidates_dict:
                candidates_dict[name] = {
                    "name": name,
                    "filename": chunk.get("filename", ""),
                    "relevance_score": round(chunk.get("score", 0.0), 2),
                    "matching_excerpts": []
                }

            text = chunk.get("text", "")
            excerpt = text[:200] + "..." if len(text) > 200 else text

            candidates_dict[name]["matching_excerpts"].append({
                "text": excerpt,
                "score": round(chunk.get("score", 0.0), 2)
            })

        candidates_list = list(candidates_dict.values())
        candidates_list.sort(key=lambda x: x["relevance_score"], reverse=True)

        return candidates_list
