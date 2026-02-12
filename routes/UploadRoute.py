from fastapi import  UploadFile, APIRouter

router = APIRouter()

@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}