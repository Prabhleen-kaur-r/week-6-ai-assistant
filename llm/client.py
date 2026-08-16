"""Gemini LLM client using current google-genai SDK."""

import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Generator
from google import genai
from google.genai.types import GenerateContentConfig

from config.settings import settings
from utils.error_handler import LLMError

logger = logging.getLogger(__name__)


class GeminiClient:
    """Gemini LLM client with structured output support."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.llm_model
        self.temperature = settings.temperature
        self.max_output_tokens = settings.max_output_tokens
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response from Gemini.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Temperature for generation
            max_tokens: Maximum output tokens
            response_format: Expected response format (JSON schema)
            
        Returns:
            Generated text
            
        Raises:
            LLMError: If generation fails
        """
        try:
            # Set up config
            config = {
                "temperature": temperature or self.temperature,
                "max_output_tokens": max_tokens or self.max_output_tokens,
            }
            
            # Add response format if specified
            if response_format:
                config["response_mime_type"] = "application/json"
                config["response_schema"] = response_format
            
            # Generate response with system prompt if provided
            if system_prompt:
                # Use system instruction via the config
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=GenerateContentConfig(
                        temperature=config["temperature"],
                        max_output_tokens=config["max_output_tokens"],
                        system_instruction=system_prompt,
                        **({"response_mime_type": "application/json", "response_schema": response_format} if response_format else {})
                    )
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                )
            
            if hasattr(response, 'text'):
                result = response.text
            else:
                result = str(response)
            
            logger.debug(f"Gemini generation successful (tokens: {len(result.split())})")
            return result
        
        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            raise LLMError(f"Failed to generate response: {str(e)}")
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response from Gemini.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Temperature for generation
            max_tokens: Maximum output tokens
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            LLMError: If generation fails
        """
        try:
            # Set up config
            config = {
                "temperature": temperature or self.temperature,
                "max_output_tokens": max_tokens or self.max_output_tokens,
            }
            
            # Generate streaming response with system prompt if provided
            if system_prompt:
                response = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=prompt,
                    config=GenerateContentConfig(
                        temperature=config["temperature"],
                        max_output_tokens=config["max_output_tokens"],
                        system_instruction=system_prompt
                    )
                )
            else:
                response = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=prompt,
                    config=config
                )
            
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
        
        except Exception as e:
            logger.error(f"Gemini streaming generation failed: {str(e)}")
            raise LLMError(f"Failed to generate streaming response: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            response = self.client.models.count_tokens(
                model=self.model,
                contents=text
            )
            return response.total_tokens
        except Exception as e:
            logger.warning(f"Token counting failed: {str(e)}")
            return len(text.split())  # Fallback estimate