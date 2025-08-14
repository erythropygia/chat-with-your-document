import asyncio
import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import html

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import settings
from backend.models.schemas import DocumentChunk
from backend.services.document_service import DocumentService

logger = logging.getLogger(__name__)

class VectorStoreService:
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.document_service = DocumentService()
    
    def _clean_text(self, text: str) -> str:
        """Metin içeriğini temizler ve düzgün karakterlere dönüştürür."""
        if not text:
            return text

        try:
            # 1. Adım: Temel temizlik
            cleaned = text.strip()
            
            # 2. Adım: HTML entity'leri düzelt
            cleaned = html.unescape(cleaned)
            
            # 3. Adım: RTF formatlamalarını temizle
            rtf_patterns = [
                r'\\[a-zA-Z]+\d*',    # \f0, \b, \cf0 gibi
                r'\\[\{\}\\\']',      # \{, \}, \\, \'
                r'\\[0-9]+',          # \12345 gibi sayılar
                r'\\\(.*?\\\)',       # \(...\) yapıları
                r'\\[a-zA-Z]',        # Tek karakterli RTF komutları
            ]
            for pattern in rtf_patterns:
                cleaned = re.sub(pattern, '', cleaned)
            
            # 4. Adım: Karakter eşlemeleri
            char_mappings = {
                # Temel Türkçe karakterler
                "Ã§": "ç", "Ã¼": "ü", "Ä±": "ı", "ÅŸ": "ş",
                "Ã¶": "ö", "Ã‡": "Ç", "Ãœ": "Ü", "Ä°": "İ",
                "Åž": "Ş", "ÄŸ": "ğ", "Äž": "Ğ",
                
                # Çift kodlanmış karakterler
                "Ã¢ÂÂ": '"', "Ã¢Â€Â": '"', "Ã¢Â€Â˜": "'",
                
                # Yaygın hatalar
                "ÃÂ": "İ", "Ã": "ı", "Â": "",
                "Ä°": "İ", "Ä±": "ı", "Åž": "Ş",
                "ÅŸ": "ş", "Ã§": "ç", "ÄŸ": "ğ"
            }
            
            # 5. Adım: Karakter düzeltmelerini uygula (3 kez)
            for _ in range(3):
                for old, new in char_mappings.items():
                    cleaned = cleaned.replace(old, new)
            
            # 6. Adım: Unicode escape'leri çöz
            cleaned = re.sub(
                r'\\u([0-9a-fA-F]{4})',
                lambda m: chr(int(m.group(1), 16)),
                cleaned
            )
            
            # 7. Adım: Son temizlik
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = cleaned.strip()
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    async def reset_collection(self):
        """Koleksiyonu sıfırlar"""
        if self.collection:
            self.collection.delete(where={})
            logger.info("Collection reset completed")
    
    async def initialize(self):
        try:
            # Disable ChromaDB telemetry completely
            import logging
            import os
            
            # Disable telemetry via environment variables
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.environ["CHROMA_TELEMETRY"] = "False"
            
            # Disable all ChromaDB telemetry loggers
            logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
            logging.getLogger("chromadb.telemetry.product").setLevel(logging.CRITICAL)
            logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
            logging.getLogger("chromadb.telemetry.events").setLevel(logging.CRITICAL)
            
            self.client = chromadb.PersistentClient(
                path=str(settings.vector_db_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    async def add_document(self, document_id: str):
        try:
            chunks_data = await self.document_service.get_document_chunks(document_id)
            
            if not chunks_data:
                logger.warning(f"No chunks found for document {document_id}")
                return
            
            documents = []
            metadatas = []
            ids = []
            
            for chunk_data in chunks_data:
                # İçeriği temizle ve öyle kaydet
                cleaned_content = self._clean_text(chunk_data["content"])
                documents.append(cleaned_content)
                
                metadata = {
                    "document_id": document_id,
                    "document_name": chunk_data["metadata"]["document_name"],
                    "chunk_index": chunk_data["metadata"]["chunk_index"],
                    "upload_date": chunk_data["metadata"]["upload_date"]
                }
                
                page_number = chunk_data["metadata"].get("page_number")
                if page_number is not None:
                    metadata["page_number"] = page_number
                
                metadatas.append(metadata)
                ids.append(chunk_data["chunk_id"])
            
            logger.info(f"Generating embeddings for {len(documents)} chunks...")
            embeddings = self.embedding_model.encode(documents, convert_to_tensor=False)
            embeddings_list = embeddings.tolist()
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings_list
            )
            
            logger.info(f"Added {len(documents)} chunks to vector store for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise
    
    async def similarity_search(self, query: str, document_ids: Optional[List[str]] = None, k: int = 5) -> List[DocumentChunk]:
        try:
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)[0]
            
            where_filter = None
            if document_ids:
                where_filter = {"document_id": {"$in": document_ids}}
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            chunks = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    similarity_score = 1.0 - distance
                    
                    # İçeriği temizle ve öyle döndür
                    cleaned_content = self._clean_text(doc)
                    chunk = DocumentChunk(
                        chunk_id=results["ids"][0][i],
                        content=cleaned_content,
                        document_id=metadata["document_id"],
                        document_name=metadata["document_name"],
                        page_number=metadata.get("page_number"),
                        chunk_index=int(metadata["chunk_index"]),
                        similarity_score=float(similarity_score)
                    )
                    chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} relevant chunks for query")
            return chunks
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    async def remove_document(self, document_id: str):
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Removed {len(results['ids'])} chunks for document {document_id}")
            else:
                logger.warning(f"No chunks found in vector store for document {document_id}")
                
        except Exception as e:
            logger.error(f"Error removing document from vector store: {str(e)}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"total_chunks": 0, "collection_name": "unknown"}