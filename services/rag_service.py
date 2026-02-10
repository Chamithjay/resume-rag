from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import os


class RAGService:
    def __init__(self, vector_store_service, llm_model: str, api_key: str, temperature: float = 0.0,
                 max_tokens: int = 2048):
        self.vector_store = vector_store_service

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            convert_system_message_to_human=True  # Gemini doesn't support system messages directly
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("human", """You are an expert technical recruiter analyzing resumes. 

Your task is to:
1. Analyze the provided resume excerpts carefully
2. Match candidates against the given requirements
3. Provide specific evidence from their resumes
4. Be honest about limitations or gaps

Always cite specific information from the resumes.

Resume excerpts:
{context}

Question: {question}

Provide a detailed answer with:
- Recommended candidates (if any match)
- Specific evidence from their resumes (quote relevant parts)
- Reasoning for your recommendation
- Any concerns or gaps

Answer:""")
        ])

    def query(self, question: str, top_k: int = 5, filter_metadata: Dict = None) -> Dict:
        """Process a RAG query"""

        # 1. Retrieve relevant documents
        retrieved_docs = self.vector_store.search(
            query=question,
            k=top_k,
            filter_dict=filter_metadata
        )

        if not retrieved_docs:
            return {
                "answer": "No relevant resumes found in the database. Please upload resumes first.",
                "sources": []
            }

        # 2. Build context
        context = self._build_context(retrieved_docs)

        # 3. Generate response
        messages = self.prompt_template.format_messages(
            context=context,
            question=question
        )

        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            # Handle Gemini API errors
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": self._format_sources(retrieved_docs)
            }

        # 4. Format sources
        sources = self._format_sources(retrieved_docs)

        return {
            "answer": answer,
            "sources": sources
        }

    def _build_context(self, docs: List[Dict]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []

        for i, doc in enumerate(docs, 1):
            candidate = doc["metadata"].get("candidate_name", "Unknown")
            filename = doc["metadata"].get("filename", "Unknown")
            content = doc["content"]

            context_parts.append(
                f"--- Document {i} ---\n"
                f"Candidate: {candidate}\n"
                f"Source: {filename}\n"
                f"Content:\n{content}\n"
            )

        return "\n".join(context_parts)

    def _format_sources(self, docs: List[Dict]) -> List[Dict]:
        """Format source documents for response"""
        sources = []
        seen = set()

        for doc in docs:
            candidate = doc["metadata"].get("candidate_name", "Unknown")
            filename = doc["metadata"].get("filename", "Unknown")

            key = f"{candidate}_{filename}"
            if key not in seen:
                sources.append({
                    "content": doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"],
                    "candidate_name": candidate,
                    "filename": filename,
                    "page": doc["metadata"].get("page")
                })
                seen.add(key)

        return sources