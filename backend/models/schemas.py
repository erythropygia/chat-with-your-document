from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_size: int
    upload_date: datetime
    chunk_count: int
    status: str

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    question: str = Field(..., description="User question")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to search in")
    chat_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous chat messages")
    max_results: Optional[int] = Field(5, description="Maximum number of results to retrieve")
    language: Optional[str] = Field("en", description="Response language (en/tr)")

class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_index: int
    similarity_score: float

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[DocumentChunk] = Field(default=[], description="Source chunks used")
    total_chunks_found: int = Field(..., description="Total number of relevant chunks found")
    processing_time: float = Field(..., description="Processing time in seconds")
    llm_model: str = Field(..., description="LLM model used for generation")

class DocumentProcessingStatus(BaseModel):
    document_id: str
    status: str
    progress: float
    message: str
    chunk_count: Optional[int] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime