from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from pathlib import Path
import sys
from typing import Optional, List

sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from backend.services.document_service import DocumentService
from backend.services.rag_service import RAGService
from backend.models.schemas import ChatRequest, ChatResponse

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.logs_dir / 'backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

document_service = DocumentService()
rag_service = RAGService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Document Search API...")
    await rag_service.initialize()
    logger.info("API startup complete")
    yield
    logger.info("Shutting down Document Search API...")

app = FastAPI(
    title="Document Search API", 
    description="AI-powered document search and Q&A system", 
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Document Search API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "document-search-api"}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed: {', '.join(settings.allowed_extensions)}"
            )
        
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.max_file_size_mb}MB"
            )
        
        document_id = await document_service.process_document(file.filename, content)
        await rag_service.add_documents_to_index(document_id)
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "status": "processed",
            "message": "Document uploaded and indexed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    try:
        documents = await document_service.list_documents()
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = await rag_service.query(
            question=request.question,
            document_ids=request.document_ids,
            chat_history=request.chat_history,
            language=request.language
        )
        return response
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        await rag_service.remove_documents_from_index(document_id)
        await document_service.delete_document(document_id)
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-db")
async def reset_database():
    try:
        await rag_service.vector_store.reset_collection()
        return {"message": "Database reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-images")
async def search_images(
    query: str = Query(..., description="Search query for images"),
    document_ids: Optional[str] = Query(None, description="Comma-separated document IDs to search in"),
    limit: int = Query(5, ge=1, le=20, description="Number of images to return")
):
    try:
        doc_ids = None
        if document_ids:
            doc_ids = [doc_id.strip() for doc_id in document_ids.split(",") if doc_id.strip()]
        
        images = await rag_service.search_images(
            query=query,
            document_ids=doc_ids,
            k=limit
        )
        
        return {
            "query": query,
            "total_found": len(images),
            "images": images
        }
        
    except Exception as e:
        logger.error(f"Error searching images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{document_id}/{image_filename}")
async def get_image(document_id: str, image_filename: str):
    try:
        image_path = settings.documents_dir / "images" / document_id / image_filename
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(
            path=str(image_path),
            media_type="image/png",
            filename=image_filename
        )
        
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}/images")
async def get_document_images(document_id: str):
    try:
        images = await document_service.get_document_images(document_id)
        
        # Add URL paths for frontend access
        for image in images:
            filename = Path(image["image_path"]).name
            image["url"] = f"/images/{document_id}/{filename}"
        
        return {
            "document_id": document_id,
            "total_images": len(images),
            "images": images
        }
        
    except Exception as e:
        logger.error(f"Error getting document images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.backend_host, port=settings.backend_port, reload=True, log_level=settings.log_level.lower())