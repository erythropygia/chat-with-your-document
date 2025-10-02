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
import fitz  # PyMuPDF
import io
import zipfile
import xml.etree.ElementTree as ET
from docx import Document as DocxDocument

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PIL import Image

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
            
            extracted_images = []
            file_extension = file_path.suffix.lower()
            if file_extension == ".pdf":
                extracted_images = await self._extract_images_from_pdf(file_path, document_id)
            elif file_extension == ".docx":
                extracted_images = await self._extract_images_from_docx(file_path, document_id)
            
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=filename,
                file_size=len(content),
                upload_date=datetime.now(),
                chunk_count=len(processed_chunks),
                status="completed"
            )
            await self._save_document_info(doc_info)
            
            if extracted_images:
                await self._save_images_metadata(document_id, extracted_images)
            
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
            
            images_dir = settings.documents_dir / "images" / document_id
            if images_dir.exists():
                import shutil
                shutil.rmtree(images_dir)
            
            images_file = settings.chunks_dir / f"{document_id}_images.json"
            if images_file.exists():
                images_file.unlink()
            
            logger.info(f"Document {document_id} deleted successfully")
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise
    
    async def _extract_images_from_pdf(self, pdf_path: Path, document_id: str) -> List[Dict[str, Any]]:
        try:
            images_dir = settings.documents_dir / "images" / document_id
            images_dir.mkdir(parents=True, exist_ok=True)
            
            extracted_images = []
            
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.width < 50 or pix.height < 50:
                            pix = None
                            continue
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("ppm")
                            pil_image = Image.open(io.BytesIO(img_data))
                        else:  # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("ppm")
                            pil_image = Image.open(io.BytesIO(img_data))
                            pix1 = None
                        
                        image_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                        image_path = images_dir / image_filename
                        pil_image.save(image_path, "PNG")
                        
                        image_info = {
                            "image_id": f"{document_id}_p{page_num + 1}_i{img_index + 1}",
                            "image_path": str(image_path),
                            "page_number": page_num + 1,
                            "image_index": img_index + 1,
                            "width": pix.width,
                            "height": pix.height,
                            "extracted_date": datetime.now().isoformat()
                        }
                        extracted_images.append(image_info)
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Error extracting image {img_index} from page {page_num + 1}: {str(e)}")
                        continue
            
            pdf_document.close()
            logger.info(f"Extracted {len(extracted_images)} images from PDF {pdf_path}")
            return extracted_images
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF {pdf_path}: {str(e)}")
            return []
    
    async def _save_images_metadata(self, document_id: str, images: List[Dict[str, Any]]):
        images_file = settings.chunks_dir / f"{document_id}_images.json"
        
        with open(images_file, "w", encoding="utf-8") as f:
            json.dump(images, f, ensure_ascii=False, indent=2, default=str)
    
    async def get_document_images(self, document_id: str) -> List[Dict[str, Any]]:
        images_file = settings.chunks_dir / f"{document_id}_images.json"
        
        if not images_file.exists():
            return []
        
        with open(images_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    async def _extract_images_from_docx(self, docx_path: Path, document_id: str) -> List[Dict[str, Any]]:
        """Extract images from DOCX and save them"""
        try:
            # Create images directory for this document
            images_dir = settings.documents_dir / "images" / document_id
            images_dir.mkdir(parents=True, exist_ok=True)
            
            extracted_images = []
            
            # Open DOCX as zip file to extract images
            with zipfile.ZipFile(docx_path, 'r') as docx_zip:
                # Find all image files in the media folder
                image_files = [f for f in docx_zip.namelist() if f.startswith('word/media/') and 
                             any(f.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp'])]
                
                for idx, image_file in enumerate(image_files):
                    try:
                        # Extract image data
                        image_data = docx_zip.read(image_file)
                        
                        # Create PIL Image to check size and convert
                        pil_image = Image.open(io.BytesIO(image_data))
                        
                        # Skip if image is too small (likely decorative)
                        if pil_image.width < 50 or pil_image.height < 50:
                            continue
                        
                        # Convert to RGB if necessary and save as PNG
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        # Save image
                        image_filename = f"docx_img_{idx + 1}.png"
                        image_path = images_dir / image_filename
                        pil_image.save(image_path, "PNG")
                        
                        # Store image metadata
                        image_info = {
                            "image_id": f"{document_id}_docx_i{idx + 1}",
                            "image_path": str(image_path),
                            "page_number": None,  # DOCX doesn't have clear page numbers
                            "image_index": idx + 1,
                            "width": pil_image.width,
                            "height": pil_image.height,
                            "extracted_date": datetime.now().isoformat(),
                            "original_name": Path(image_file).name
                        }
                        extracted_images.append(image_info)
                        
                    except Exception as e:
                        logger.warning(f"Error extracting image {image_file}: {str(e)}")
                        continue
            
            logger.info(f"Extracted {len(extracted_images)} images from DOCX {docx_path}")
            return extracted_images
            
        except Exception as e:
            logger.error(f"Error extracting images from DOCX {docx_path}: {str(e)}")
            return []