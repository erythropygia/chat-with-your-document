import asyncio
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import re
import chardet

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import settings
from backend.models.schemas import DocumentInfo, DocumentProcessingStatus

logger = logging.getLogger(__name__)

class DocumentService:
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    async def process_document(self, filename: str, content: bytes) -> str:
        try:
            document_id = str(uuid.uuid4())
            
            file_path = settings.documents_dir / f"{document_id}_{filename}"
            with open(file_path, "wb") as f:
                f.write(content)
            
            logger.info(f"Processing document: {filename} (ID: {document_id})")
            
            documents = await self._load_document(file_path)
            chunks = self.text_splitter.split_documents(documents)
            
            processed_chunks = []
            for i, chunk in enumerate(chunks):               
                chunk.metadata.update({
                    "document_id": document_id,
                    "document_name": filename,
                    "chunk_index": i,
                    "chunk_id": f"{document_id}_{i}",
                    "upload_date": datetime.now().isoformat()
                })
                processed_chunks.append(chunk)
            
            await self._save_chunks_metadata(document_id, processed_chunks)
            
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=filename,
                file_size=len(content),
                upload_date=datetime.now(),
                chunk_count=len(processed_chunks),
                status="completed"
            )
            await self._save_document_info(doc_info)
            
            logger.info(f"Document processed successfully: {filename} ({len(processed_chunks)} chunks)")
            return document_id
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            raise
    
    async def _load_document(self, file_path: Path) -> List[Document]:
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_extension == ".docx":
                loader = Docx2txtLoader(str(file_path))
            elif file_extension in [".txt", ".md"]:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                
                loader = TextLoader(str(file_path), encoding=encoding)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            
            if file_extension == ".pdf":
                for i, doc in enumerate(documents):
                    doc.metadata["page_number"] = i + 1
            else:
                for doc in documents:
                    doc.metadata["page_number"] = None
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    async def _save_chunks_metadata(self, document_id: str, chunks: List[Document]):
        chunks_file = settings.chunks_dir / f"{document_id}_chunks.json"
        
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "chunk_id": chunk.metadata["chunk_id"],
                "content": chunk.page_content,
                "metadata": chunk.metadata
            })
        
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2, default=str)
    
    async def _save_document_info(self, doc_info: DocumentInfo):
        doc_file = settings.chunks_dir / f"{doc_info.document_id}_info.json"
        
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(doc_info.dict(), f, ensure_ascii=False, indent=2, default=str)
    
    async def list_documents(self) -> List[DocumentInfo]:
        documents = []
        
        for info_file in settings.chunks_dir.glob("*_info.json"):
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)
                    documents.append(DocumentInfo(**doc_data))
            except Exception as e:
                logger.error(f"Error reading document info {info_file}: {str(e)}")
        
        return sorted(documents, key=lambda x: x.upload_date, reverse=True)
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        chunks_file = settings.chunks_dir / f"{document_id}_chunks.json"
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks not found for document {document_id}")
        
        with open(chunks_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    async def delete_document(self, document_id: str):
        try:
            doc_files = list(settings.documents_dir.glob(f"{document_id}_*"))
            for file_path in doc_files:
                file_path.unlink()
            
            chunks_file = settings.chunks_dir / f"{document_id}_chunks.json"
            info_file = settings.chunks_dir / f"{document_id}_info.json"
            
            if chunks_file.exists():
                chunks_file.unlink()
            if info_file.exists():
                info_file.unlink()
            
            logger.info(f"Document {document_id} deleted successfully")
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise