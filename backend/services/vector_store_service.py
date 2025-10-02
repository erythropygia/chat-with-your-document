import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import uuid
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import settings
from backend.models.schemas import DocumentChunk
from backend.services.document_service import DocumentService

logger = logging.getLogger(__name__)

class VectorStoreService:
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.image_collection = None
        self.embedding_model = None
        self.image_embedding_model = None
        self.image_tokenizer = None
        self.device = None
        self.image_transform = None
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
            
            self.image_collection = self.client.get_or_create_collection(
                name="images",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            
            self.image_embedding_model = AutoModel.from_pretrained("erythropygia/turkish-clip-vit-bert", trust_remote_code=True)
            self.image_tokenizer = AutoTokenizer.from_pretrained("erythropygia/turkish-clip-vit-bert")
  
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
            else:
                self.device = "cpu"

            self.image_embedding_model.to(self.device)
            self.image_embedding_model.eval()
            
            self.image_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

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
            # Remove text chunks
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Removed {len(results['ids'])} chunks for document {document_id}")
            else:
                logger.warning(f"No chunks found in vector store for document {document_id}")
            
            # Remove images
            image_results = self.image_collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if image_results["ids"]:
                self.image_collection.delete(ids=image_results["ids"])
                logger.info(f"Removed {len(image_results['ids'])} images for document {document_id}")
            else:
                logger.warning(f"No images found in vector store for document {document_id}")
                
        except Exception as e:
            logger.error(f"Error removing document from vector store: {str(e)}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        try:
            text_count = self.collection.count()
            image_count = self.image_collection.count()
            return {
                "total_chunks": text_count,
                "total_images": image_count,
                "collection_name": self.collection.name,
                "image_collection_name": self.image_collection.name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"total_chunks": 0, "total_images": 0, "collection_name": "unknown", "image_collection_name": "unknown"}

    async def add_image(self, image_path: str, document_id: str, document_name: str, page_number: Optional[int] = None):
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_embeds = self.image_embedding_model.encode_image(pixel_values)
                image_embedding = image_embeds.cpu().numpy().flatten().tolist()
            
            image_id = str(uuid.uuid4())
            
            metadata = {
                "document_id": document_id,
                "document_name": document_name,
                "image_path": str(image_path),
                "upload_date": datetime.now().isoformat(),
                "type": "image"
            }
            
            if page_number is not None:
                metadata["page_number"] = page_number
            
            self.image_collection.add(
                documents=[f"Image from {document_name}"],
                metadatas=[metadata],
                ids=[image_id],
                embeddings=[image_embedding]
            )
            
            logger.info(f"Added image {image_path} to vector store with ID {image_id}")
            return image_id
            
        except Exception as e:
            logger.error(f"Error adding image to vector store: {str(e)}")
            raise
    
    async def search_images_by_text(self, query: str, document_ids: Optional[List[str]] = None, k: int = 5) -> List[Dict[str, Any]]:
        try:
            inputs = self.image_tokenizer(query, return_tensors="pt", padding=True,
                                        truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                text_embeds = self.image_embedding_model.encode_text(inputs["input_ids"], inputs["attention_mask"])
                text_embedding = text_embeds.cpu().numpy().flatten().tolist()
            
            where_filter = None
            if document_ids:
                where_filter = {"document_id": {"$in": document_ids}}
            
            results = self.image_collection.query(
                query_embeddings=[text_embedding],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            images = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    similarity_score = 1.0 - distance
                    
                    image_result = {
                        "image_id": results["ids"][0][i],
                        "document_id": metadata["document_id"],
                        "document_name": metadata["document_name"],
                        "image_path": metadata["image_path"],
                        "page_number": metadata.get("page_number"),
                        "similarity_score": float(similarity_score),
                        "upload_date": metadata["upload_date"]
                    }
                    images.append(image_result)
            
            logger.info(f"Found {len(images)} relevant images for query: {query}")
            return images
            
        except Exception as e:
            logger.error(f"Error searching images: {str(e)}")
            raise