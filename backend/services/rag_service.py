import asyncio
import time
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import settings
from backend.models.schemas import ChatResponse, DocumentChunk, ChatMessage
from backend.services.vector_store_service import VectorStoreService
from backend.services.llm_service import LLMService
from backend.services.chat_logger import ChatLogger

logger = logging.getLogger(__name__)

class RAGService:
    
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService()
        self.chat_logger = ChatLogger()
        self.initialized = False
    
    async def initialize(self):
        try:
            await self.vector_store.initialize()
            await self.llm_service.initialize()
            self.initialized = True
        except Exception as e:
            logger.error(f"Error initializing RAG service: {str(e)}")
            raise
    
    async def query(self, question: str, document_ids: Optional[List[str]] = None, chat_history: Optional[List[ChatMessage]] = None, max_results: int = 5, language: str = "en") -> ChatResponse:
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            relevant_chunks = await self.vector_store.similarity_search(
                query=question,
                document_ids=document_ids,
                k=max_results
            )
            
            context = self._build_context(relevant_chunks)
            formatted_history = self._format_chat_history(chat_history or [])
            
            response = await self.llm_service.generate_response(
                question=question,
                context=context,
                chat_history=formatted_history,
                language=language
            )
            
            processing_time = time.time() - start_time
            
            self.chat_logger.log_chat_interaction(
                question=question,
                response=response,
                language=language,
                context=context,
                chat_history=[msg.dict() if hasattr(msg, 'dict') else msg for msg in (chat_history or [])],
                sources=[chunk.dict() if hasattr(chunk, 'dict') else chunk for chunk in relevant_chunks],
                processing_time=processing_time
            )
            
            return ChatResponse(
                answer=response,
                sources=relevant_chunks,
                total_chunks_found=len(relevant_chunks),
                processing_time=processing_time,
                llm_model=settings.ollama_model
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def _build_context(self, chunks: List[DocumentChunk]) -> str:
        if not chunks:
            return "No relevant documents found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f" (Page {chunk.page_number})" if chunk.page_number else ""
            context_parts.append(
                f"[Source {i} - {chunk.document_name}{page_info}]\n"
                f"{chunk.content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _format_chat_history(self, chat_history: List[ChatMessage]) -> str:
        if not chat_history:
            return ""
        
        formatted_messages = []
        for msg in chat_history[-6:]:
            role = "User" if msg.role == "user" else "Assistant"
            formatted_messages.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted_messages)
    
    async def add_documents_to_index(self, document_id: str):
        try:
            await self.vector_store.add_document(document_id)
        except Exception as e:
            logger.error(f"Error adding document to index: {str(e)}")
            raise
    
    async def remove_documents_from_index(self, document_id: str):
        try:
            await self.vector_store.remove_document(document_id)
        except Exception as e:
            logger.error(f"Error removing document from index: {str(e)}")
            raise