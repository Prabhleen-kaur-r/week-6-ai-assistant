"""Document chunking utilities."""

import re
from typing import List, Dict, Any
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunk documents into overlapping segments with metadata."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        # Track processed chunks to avoid duplicates
        self._processed_chunks = set()
    
    def chunk_text(
        self,
        text: str,
        source: str,
        document_type: str = "txt",
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text into overlapping segments."""
        if not text or not text.strip():
            logger.warning(f"Empty text for source: {source}")
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # Handle text shorter than chunk size
        if len(text) <= self.chunk_size:
            chunk_metadata = {
                "source": source,
                "chunk_id": f"{source}_0",
                "chunk_index": 0,
                "document_type": document_type,
                "char_start": 0,
                "char_end": len(text),
                "chunk_size": len(text)
            }
            if metadata:
                chunk_metadata.update(metadata)
            
            chunk_key = f"{source}_0"
            if chunk_key not in self._processed_chunks:
                self._processed_chunks.add(chunk_key)
                return [{
                    "text": text.strip(),
                    "metadata": chunk_metadata
                }]
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        max_chunks = 50
        
        while start < len(text) and chunk_id < max_chunks:
            end = min(start + self.chunk_size, len(text))
            
            if end < len(text):
                boundary_chars = ['.', '!', '?']
                for char in boundary_chars:
                    last_pos = text.rfind(char, start, end)
                    if last_pos != -1 and last_pos > start + self.chunk_size // 2:
                        end = last_pos + 1
                        break
                
                if end < len(text):
                    last_space = text.rfind(' ', start, end)
                    if last_space != -1 and last_space > start + self.chunk_size // 2:
                        end = last_space + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_key = f"{source}_{chunk_id}"
                if chunk_key not in self._processed_chunks:
                    self._processed_chunks.add(chunk_key)
                    chunk_metadata = {
                        "source": source,
                        "chunk_id": chunk_key,
                        "chunk_index": chunk_id,
                        "document_type": document_type,
                        "char_start": start,
                        "char_end": end,
                        "chunk_size": len(chunk_text)
                    }
                    if metadata:
                        chunk_metadata.update(metadata)
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": chunk_metadata
                    })
                    chunk_id += 1
            
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start
            
            if start >= len(text):
                break
        
        if chunk_id >= max_chunks:
            logger.warning(f"Document {source} truncated to {max_chunks} chunks to avoid quota issues")
        
        logger.info(f"Chunked {source} into {len(chunks)} chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = text.replace('"', '"').replace('"', '"')
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()
    
    def chunk_document(
        self,
        text: str,
        filename: str,
        file_type: str,
        page: int = None
    ) -> List[Dict[str, Any]]:
        """Chunk a document with appropriate metadata."""
        metadata = {
            "filename": filename,
            "file_type": file_type,
        }
        if page is not None:
            metadata["page"] = page
        
        return self.chunk_text(
            text=text,
            source=filename,
            document_type=file_type,
            metadata=metadata
        )
    
    def reset(self):
        """Reset processed chunks tracker."""
        self._processed_chunks.clear()