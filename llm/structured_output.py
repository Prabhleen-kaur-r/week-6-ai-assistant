"""Structured output parsing for Gemini responses."""

import json
import re
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError
from enum import Enum

logger = logging.getLogger(__name__)


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResearchAnswer(BaseModel):
    answer: str
    key_points: List[str]
    confidence: Confidence
    sources: List[str]


class StructuredOutputParser:
    """Parse structured output from LLM responses."""
    
    @staticmethod
    def parse(response_text: str) -> Optional[ResearchAnswer]:
        if not response_text or not response_text.strip():
            return None
        
        parsers = [
            StructuredOutputParser._parse_json,
            StructuredOutputParser._parse_markdown_json,
            StructuredOutputParser._parse_extract,
            StructuredOutputParser._parse_fallback
        ]
        
        for parser in parsers:
            try:
                result = parser(response_text)
                if result:
                    return result
            except Exception as e:
                logger.debug(f"Parser {parser.__name__} failed: {str(e)}")
                continue
        
        return ResearchAnswer(
            answer=response_text.strip()[:500],
            key_points=[],
            confidence=Confidence.MEDIUM,
            sources=[]
        )
    
    @staticmethod
    def _parse_json(text: str) -> Optional[ResearchAnswer]:
        try:
            json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ResearchAnswer(**data)
            return None
        except (json.JSONDecodeError, ValidationError):
            return None
    
    @staticmethod
    def _parse_markdown_json(text: str) -> Optional[ResearchAnswer]:
        json_pattern = r'```(?:json)?\s*({.*?})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                return ResearchAnswer(**data)
            except (json.JSONDecodeError, ValidationError):
                continue
        return None
    
    @staticmethod
    def _parse_extract(text: str) -> Optional[ResearchAnswer]:
        try:
            clean_text = text.strip()
            
            # Extract answer
            answer_patterns = [
                r'(?:Answer|Response|Summary)[:\s]+([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n(?:Key Points|Sources|Confidence|Based on|According to)|$)',
                r'^([^\n]{50,}?)(?:\n\n|\n(?:Based on|According to|Source|Key Point))',
                r'^([^\n]+)',
            ]
            
            answer = None
            for pattern in answer_patterns:
                match = re.search(pattern, clean_text, re.IGNORECASE | re.DOTALL)
                if match:
                    candidate = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                    candidate = re.sub(r'\*\*([^*]+)\*\*', r'\1', candidate)
                    candidate = re.sub(r'#{1,6}\s*', '', candidate)
                    if len(candidate) > 20:
                        answer = candidate
                        break
            
            if not answer:
                sentences = re.split(r'[.!?]\s+', clean_text)
                for sent in sentences:
                    sent = sent.strip()
                    sent = re.sub(r'\*\*([^*]+)\*\*', r'\1', sent)
                    sent = re.sub(r'#{1,6}\s*', '', sent)
                    if len(sent) > 30:
                        answer = sent
                        break
            
            if not answer:
                answer = clean_text[:300]
            
            # Extract key points
            key_points = []
            bullet_pattern = r'(?:^|\n)\s*[•\-*]\s*([^\n]+)'
            bullet_matches = re.findall(bullet_pattern, clean_text, re.MULTILINE)
            
            for match in bullet_matches:
                cleaned = match.strip()
                cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
                cleaned = re.sub(r'#{1,6}\s*', '', cleaned)
                if cleaned and len(cleaned) > 5:
                    key_points.append(cleaned)
            
            if not key_points:
                numbered_pattern = r'(?:^|\n)\s*(\d+)[.)]\s*([^\n]+)'
                numbered_matches = re.findall(numbered_pattern, clean_text, re.MULTILINE)
                for _, text_part in numbered_matches:
                    cleaned = text_part.strip()
                    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
                    if cleaned and len(cleaned) > 5:
                        key_points.append(cleaned)
            
            if not key_points:
                key_section = re.search(r'(?:Key Points|Points|Highlights)[:\s]+([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n(?:Source|Confidence|Based on)|$)', 
                                       clean_text, re.IGNORECASE | re.DOTALL)
                if key_section:
                    section_text = key_section.group(1)
                    lines = section_text.split('\n')
                    for line in lines[:5]:
                        cleaned = line.strip()
                        cleaned = re.sub(r'^[•\-*\d+.)]\s*', '', cleaned)
                        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
                        if cleaned and len(cleaned) > 5:
                            key_points.append(cleaned)
            
            key_points = key_points[:5]
            
            # Extract confidence
            confidence = Confidence.MEDIUM
            lower_text = clean_text.lower()
            if 'high confidence' in lower_text or 'definitely' in lower_text or 'clearly' in lower_text or 'directly stated' in lower_text:
                confidence = Confidence.HIGH
            elif 'low confidence' in lower_text or 'uncertain' in lower_text or 'not sure' in lower_text:
                confidence = Confidence.LOW
            
            # Extract sources – only capture strings that look like filenames with extensions
            sources = []
            source_patterns = [
                r'(?:Sources?|From|Based on)[:\s]+([^\n]+)',
                r'`([^`]+\.(?:pdf|txt|docx))`',
                r'([^\s,;]+\.(?:pdf|txt|docx))',
                r'According to ([^\s,;.]+\.(?:pdf|txt|docx))',
            ]
            
            for pattern in source_patterns:
                matches = re.findall(pattern, clean_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    # Clean up
                    cleaned = re.sub(r'^[•\-*\d+.)\s]+', '', match.strip())
                    cleaned = re.sub(r'^```', '', cleaned)
                    cleaned = re.sub(r'```$', '', cleaned)
                    # Ensure it looks like a filename with extension
                    if cleaned and '.' in cleaned and len(cleaned) > 3:
                        # Extract just the filename part (remove extra text)
                        filename_match = re.search(r'([^/\s,;]+\.(?:pdf|txt|docx))', cleaned, re.IGNORECASE)
                        if filename_match:
                            cleaned = filename_match.group(1)
                        # Avoid capturing page numbers or other stray text
                        if not cleaned.startswith('Page') and 'page' not in cleaned.lower():
                            if cleaned not in sources:
                                sources.append(cleaned)
            
            # If still no sources, try to find any filename in the entire text
            if not sources:
                file_matches = re.findall(r'([^\s,;]+\.(?:pdf|txt|docx))', clean_text, re.IGNORECASE)
                for match in file_matches:
                    if match not in sources:
                        sources.append(match)
            
            sources = [s for s in sources if s and '.' in s and len(s) > 3 and not s.startswith('Page')]
            sources = list(dict.fromkeys(sources))[:5]
            
            return ResearchAnswer(
                answer=answer,
                key_points=key_points,
                confidence=confidence,
                sources=sources
            )
        
        except Exception as e:
            logger.debug(f"Extract parsing failed: {str(e)}")
            return None
    
    @staticmethod
    def _parse_fallback(text: str) -> Optional[ResearchAnswer]:
        try:
            clean_text = text.strip()
            sentences = re.split(r'[.!?]\s+', clean_text)
            
            answer = ""
            for sent in sentences:
                sent = sent.strip()
                sent = re.sub(r'\*\*([^*]+)\*\*', r'\1', sent)
                sent = re.sub(r'#{1,6}\s*', '', sent)
                if len(sent) > 30:
                    answer = sent
                    break
            
            if not answer:
                answer = clean_text[:300]
            
            key_points = []
            bullet_pattern = r'(?:^|\n)\s*[•\-*]\s*([^\n]+)'
            bullet_matches = re.findall(bullet_pattern, clean_text, re.MULTILINE)
            for match in bullet_matches[:5]:
                cleaned = match.strip()
                cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
                cleaned = re.sub(r'#{1,6}\s*', '', cleaned)
                if cleaned and len(cleaned) > 5:
                    key_points.append(cleaned)
            
            sources = []
            file_matches = re.findall(r'([^\s,;]+\.(?:pdf|txt|docx))', clean_text, re.IGNORECASE)
            for match in file_matches:
                if match not in sources:
                    sources.append(match)
            
            confidence = Confidence.MEDIUM
            lower_text = clean_text.lower()
            if 'high' in lower_text or 'definitely' in lower_text or 'clearly' in lower_text:
                confidence = Confidence.HIGH
            elif 'low' in lower_text or 'uncertain' in lower_text or 'not sure' in lower_text:
                confidence = Confidence.LOW
            
            sources = [s for s in sources if s and '.' in s and len(s) > 3 and not s.startswith('Page')]
            sources = list(dict.fromkeys(sources))[:5]
            
            return ResearchAnswer(
                answer=answer,
                key_points=key_points,
                confidence=confidence,
                sources=sources
            )
        
        except Exception as e:
            logger.debug(f"Fallback parsing failed: {str(e)}")
            return None