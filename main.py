from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import UploadRoute


app = FastAPI()

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(UploadRoute.router, prefix="", tags=["Upload"])
@app.get("/")
async def root():
    return {
        "message": "Resume RAG Analyzer API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080
    )