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
import logging
import os

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
    

    
    async def reset_collection(self):
        if self.collection:
            self.collection.delete(where={})
            logger.info("Collection reset completed")
    
    async def initialize(self):
        try:     
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.environ["CHROMA_TELEMETRY"] = "False"
            
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
                documents.append(chunk_data["content"])
                
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
                    
                    chunk = DocumentChunk(
                        chunk_id=results["ids"][0][i],
                        content=doc,
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