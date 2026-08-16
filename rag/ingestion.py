"""Document ingestion and processing."""

import os
import shutil
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import logging
from datetime import datetime

from config.settings import settings
from rag.chunking import DocumentChunker
from utils.error_handler import DocumentProcessingError

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Handle document upload, extraction, and processing."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt'}
    MAX_FILE_SIZE = settings.max_upload_size_mb * 1024 * 1024
    
    def __init__(self):
        self.uploads_dir = settings.uploads_dir
        self.chunker = DocumentChunker()
        os.makedirs(self.uploads_dir, exist_ok=True)
    
    def validate_file(self, filename: str, file_size: int) -> None:
        """Validate uploaded file."""
        if file_size > self.MAX_FILE_SIZE:
            raise DocumentProcessingError(
                f"File size exceeds maximum of {settings.max_upload_size_mb}MB",
                filename=filename
            )
        
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise DocumentProcessingError(
                f"Unsupported file type: {ext}. Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}",
                filename=filename
            )
        
        if file_size == 0:
            raise DocumentProcessingError("File is empty", filename=filename)
    
    def save_upload(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(self.uploads_dir, safe_filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"Saved upload: {filename} -> {safe_filename}")
        return file_path
    
    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from document."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self._extract_pdf(file_path)
        elif ext == '.txt':
            return self._extract_txt(file_path)
        else:
            raise DocumentProcessingError(
                f"Unsupported file type: {ext}",
                filename=os.path.basename(file_path)
            )
    
    def _extract_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF."""
        try:
            doc = fitz.open(file_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    pages.append({
                        "text": text,
                        "page": page_num + 1
                    })
            
            doc.close()
            logger.info(f"Extracted {len(pages)} pages from {os.path.basename(file_path)}")
            return pages
        
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to extract text from PDF: {str(e)}",
                filename=os.path.basename(file_path)
            )
    
    def _extract_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                raise DocumentProcessingError(
                    "TXT file is empty",
                    filename=os.path.basename(file_path)
                )
            
            return [{"text": text, "page": 1}]
        
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                return [{"text": text, "page": 1}]
            except Exception as e:
                raise DocumentProcessingError(
                    f"Failed to read TXT file: {str(e)}",
                    filename=os.path.basename(file_path)
                )
        
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to extract text from TXT: {str(e)}",
                filename=os.path.basename(file_path)
            )
    
    def process_document(
        self,
        file_content: bytes,
        filename: str
    ) -> List[Dict[str, Any]]:
        """Process an uploaded document."""
        # Reset chunker for new document
        self.chunker.reset()
        
        self.validate_file(filename, len(file_content))
        file_path = self.save_upload(file_content, filename)
        
        try:
            pages = self.extract_text(file_path)
            all_chunks = []
            
            for page_data in pages:
                chunks = self.chunker.chunk_document(
                    text=page_data["text"],
                    filename=filename,
                    file_type=os.path.splitext(filename)[1][1:],
                    page=page_data.get("page")
                )
                all_chunks.extend(chunks)
            
            logger.info(f"Processed {filename}: {len(all_chunks)} chunks from {len(pages)} pages")
            return all_chunks
        
        except Exception as e:
            try:
                os.remove(file_path)
            except:
                pass
            raise
    
    def cleanup_upload(self, file_path: str) -> None:
        """Clean up uploaded file after processing."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")