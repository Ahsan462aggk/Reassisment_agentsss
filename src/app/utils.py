import os
import sys
import uuid
import tempfile
import io
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

# Handle magic import for Windows vs other OS
if sys.platform == 'win32':
    import magic  # This will use python-magic-bin on Windows
else:
    import magic  # This will use python-magic on other platforms
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logger = logging.getLogger(__name__)

# Supported file types and their MIME types
SUPPORTED_FILE_TYPES = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'application/msword': '.doc',
    'application/vnd.ms-powerpoint': '.ppt',
    'application/vnd.ms-excel': '.xls',
    'text/plain': '.txt'
}

def get_file_extension(file_path: str) -> Optional[str]:
    """Get file extension based on MIME type."""
    mime = magic.Magic(mime=True)
    file_mime = mime.from_file(file_path)
    return SUPPORTED_FILE_TYPES.get(file_mime)

def save_upload_file(upload_file, suffix: str = "") -> str:
    """Save uploaded file to a temporary file and return its path."""
    try:
        suffix = "." + suffix.lstrip(".") if suffix else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload_file.file.read())
            return tmp.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

def load_document(file_path: str) -> List[Document]:
    """Load document using appropriate loader based on file type."""
    file_ext = get_file_extension(file_path)
    
    if not file_ext:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    try:
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext in ['.docx', '.doc']:
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_ext in ['.pptx', '.ppt']:
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"No loader available for file type: {file_ext}")
        
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}")
        raise

def chunk_documents(documents: List[Document], 
                   chunk_size: int = 1000, 
                   chunk_overlap: int = 200) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def process_uploaded_file(upload_file, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Process an uploaded file: save, load, and chunk it."""
    temp_file_path = None
    try:
        # Save the uploaded file temporarily
        temp_file_path = save_upload_file(upload_file)
        
        # Load the document
        documents = load_document(temp_file_path)
        if not documents:
            logger.warning(f"No content could be extracted from {upload_file.filename}")
            return []
        
        # Chunk the document
        chunks = chunk_documents(documents, chunk_size, chunk_overlap)
        if not chunks:
            logger.warning(f"No chunks could be created from {upload_file.filename}")
            return []
            
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing file {upload_file.filename}: {str(e)}")
        return []
        
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error deleting temporary file {temp_file_path}: {str(e)}")

def generate_slide_metadata(slide: Optional[dict] = None) -> dict:
    """Generate metadata dictionary for a slide."""
    now = datetime.utcnow().isoformat()
    if slide is None:
        return {
            "slide_name": "",
            "course_name": "",
            "subject_name": "",
            "description": "",
            "created_at": now,
            "updated_at": now
        }
    return {
        "slide_name": getattr(slide, 'slide_name', ''),
        "course_name": getattr(slide, 'course_name', ''),
        "subject_name": getattr(slide, 'subject_name', ''),
        "description": getattr(slide, 'description', '') or "",
        "created_at": now,
        "updated_at": now
    }

def generate_slide_id() -> str:
    """Generate a unique ID for a slide."""
    return str(uuid.uuid4())

def extract_text_from_pdf(file_content: bytes) -> str:
    """Legacy function for backward compatibility."""
    from PyPDF2 import PdfReader
    import io
    
    pdf_reader = PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text