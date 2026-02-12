from pathlib import Path
from fastapi import UploadFile, HTTPException

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class FileService:
    @staticmethod
    async def save_file(file: UploadFile) -> str :
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        file_path = UPLOAD_DIR / file.filename

        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        return str(file_path)
