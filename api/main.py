"""FastAPI application main entry point."""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from config.settings import settings
from utils.logging_config import setup_logging, get_logger
from utils.error_handler import (
    AppError,
    AuthenticationError,
    handle_app_error,
    handle_pydantic_error
)
from pydantic import ValidationError

# Set up logging
setup_logging(settings.log_level)
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI Knowledge Base Assistant with RAG capabilities",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle application-specific errors."""
    logger.error(f"AppError: {exc.message}", extra={"error": exc.details})
    return JSONResponse(
        status_code=exc.status_code,
        content=handle_app_error(exc)
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.error(f"ValidationError: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=handle_pydantic_error(exc)
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "internal_server_error",
                "message": "An unexpected error occurred",
                "details": {"error": str(exc) if settings.log_level == "DEBUG" else None}
            }
        }
    )


# Import routes after app is created to avoid circular imports
from api.routes_health import router as health_router
from api.routes_chat import router as chat_router
from api.routes_documents import router as documents_router
from api.routes_tools import router as tools_router

# Register routes
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(tools_router)


@app.on_event("startup")
async def startup_event():
    """Perform startup tasks."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Using LLM: {settings.llm_model}")
    logger.info(f"Using Embedding Model: {settings.embedding_model}")


@app.on_event("shutdown")
async def shutdown_event():
    """Perform shutdown tasks."""
    logger.info("Shutting down application")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )