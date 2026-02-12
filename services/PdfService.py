import PyPDF2
from pathlib import Path
from fastapi import HTTPException


class PDFService:

    def extract_text(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        text = ""
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

        print("\n📄 Extracted Text:\n", text)
        return text
