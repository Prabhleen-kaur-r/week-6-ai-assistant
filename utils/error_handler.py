"""Error handling utilities for the application."""

from typing import Optional, Any, Dict
from fastapi import HTTPException, status
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base application error class."""
    
    def __init__(
        self,
        message: str,
        error_type: str = "application_error",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    ):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)


class DocumentProcessingError(AppError):
    """Error during document processing."""
    
    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_type="document_processing_error",
            details={"filename": filename, **(details or {})},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


class RetrievalError(AppError):
    """Error during retrieval operations."""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_type="retrieval_error",
            details={"query": query, **(details or {})},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class LLMError(AppError):
    """Error during LLM operations."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_type="llm_error",
            details=details or {},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class AuthenticationError(AppError):
    """Error during authentication."""
    
    def __init__(
        self,
        message: str = "Invalid or missing API key",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_type="authentication_error",
            details=details or {},
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class ValidationError(AppError):
    """Error during validation."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_type="validation_error",
            details={"field": field, **(details or {})},
            status_code=status.HTTP_400_BAD_REQUEST
        )


def handle_app_error(error: AppError) -> Dict[str, Any]:
    """
    Convert an AppError to a standardized error response.
    
    Args:
        error: The application error
    
    Returns:
        Dict with error details
    """
    logger.error(
        f"Application error: {error.error_type} - {error.message}",
        extra={"error": error.details}
    )
    
    return {
        "error": {
            "type": error.error_type,
            "message": error.message,
            "details": error.details
        }
    }


def handle_pydantic_error(error: ValidationError) -> Dict[str, Any]:
    """
    Convert a Pydantic validation error to a standardized response.
    
    Args:
        error: The Pydantic validation error
    
    Returns:
        Dict with error details
    """
    errors = []
    for err in error.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in err["loc"]),
            "message": err["msg"],
            "type": err["type"]
        })
    
    return {
        "error": {
            "type": "validation_error",
            "message": "Invalid request data",
            "details": {"errors": errors}
        }
    }


class ErrorHandler:
    """Helper class for handling errors in a consistent way."""
    
    @staticmethod
    def safe_execute(func, fallback_value: Any = None, **kwargs) -> Any:
        """
        Safely execute a function with error handling.
        
        Args:
            func: The function to execute
            fallback_value: Value to return on error
            **kwargs: Arguments to pass to the function
        
        Returns:
            The function result or fallback value
        """
        try:
            return func(**kwargs)
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
            return fallback_value
    
    @staticmethod
    def http_exception(
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        """
        Create a FastAPI HTTP exception.
        
        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details
        
        Returns:
            HTTPException
        """
        return HTTPException(
            status_code=status_code,
            detail={
                "message": message,
                "details": details or {}
            }
        )