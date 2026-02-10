from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import os


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            raise Exception(f"Error extracting DOCX: {str(e)}")

    def extract_text(self, file_path: str) -> str:
        """Extract text based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def extract_candidate_name(self, text: str) -> str:
        """Simple name extraction - first non-empty line"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return lines[0] if lines else "Unknown"

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into chunks with metadata"""
        chunks = self.text_splitter.split_text(text)

        return [
            {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i
                }
            }
            for i, chunk in enumerate(chunks)
        ]