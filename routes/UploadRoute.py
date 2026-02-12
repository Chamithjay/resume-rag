from fastapi import UploadFile, APIRouter, Depends
from services.FileService import FileService
from services.PdfService import PDFService

router = APIRouter(prefix="", tags=["Upload"])


def get_file_service():
    return FileService()


def get_pdf_service():
    return PDFService()


@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile, file_service: FileService = Depends(get_file_service),
                             pdf_service: PDFService = Depends(get_pdf_service)):
    file_path = await file_service.save_file(file)

    text =  pdf_service.extract_text(file_path)
    return {
        "message": "File processed",
        "filename": file.filename, "char_count": len(text)
    }
