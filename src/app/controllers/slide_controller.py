from fastapi import HTTPException, UploadFile, status
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ..database import get_pinecone_index
from .. import utils
from ..schemas.slide import Slide, SlideCreate, SlideInDB
import os 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
pinecone_index = get_pinecone_index()

# Maximum file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024

async def process_file_chunks(file: UploadFile) -> List[Dict[str, Any]]:
    """Process file and return chunks with metadata."""
    try:
        if not file or not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided or invalid file"
            )

        # Check file size
        file.file.seek(0, 2)  # Go to end of file
        file_size = file.file.tell()
        file.file.seek(0)  # Reset file pointer
        
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
            
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size ({file_size/1024/1024:.2f}MB) exceeds maximum allowed size of {MAX_FILE_SIZE/1024/1024}MB"
            )
        
        # Process the file using our utility functions
        chunks = utils.process_uploaded_file(file)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid content could be extracted from the file. Please check if the file format is supported."
            )
            
        # Filter out any None or invalid chunks
        valid_chunks = [chunk for chunk in chunks if chunk and hasattr(chunk, 'page_content') and chunk.page_content.strip()]
        
        if not valid_chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid text content could be extracted from the file."
            )
            
        return valid_chunks
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename if file and hasattr(file, 'filename') else 'unknown'}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the file: {str(e)}"
        )

async def upload_slide(slide_data: SlideCreate, file: UploadFile) -> List[SlideInDB]:
    """Upload a new slide with metadata and process its content."""
    try:
        logger.info(f"Starting slide upload for file: {file.filename if file else 'unknown'}")
        
        if not slide_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No slide data provided"
            )
            
        if not file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
            
        if not slide_data.slide_name or not slide_data.course_name or not slide_data.subject_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Slide name, course name, and subject name are required"
            )
        
        # Process the file and get chunks
        try:
            chunks = await process_file_chunks(file)
            logger.info(f"Processed {len(chunks)} chunks from file")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing file chunks: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process file: {str(e)}"
            )
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid content could be extracted from the file"
            )
        
        # Initialize embeddings with explicit model and dimension
        try:
            doc_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                task_type="retrieval_document",  # Use retrieval_document for document embeddings
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Verify the embedding dimension
            test_embedding = doc_embeddings.embed_documents(["test"])[0]
            embedding_dimension = len(test_embedding)
            
            if embedding_dimension != 768:  # Expected dimension for text-embedding-004
                error_msg = f"Unexpected embedding dimension: {embedding_dimension}. Expected 768."
                logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize text embeddings: {str(e)}"
            )
        
        # Process each chunk
        results = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            try:
                if not chunk or not hasattr(chunk, 'page_content') or not chunk.page_content.strip():
                    logger.warning(f"Skipping empty or invalid chunk {i}")
                    continue
                    
                # Generate a unique ID for this chunk
                chunk_id = f"{utils.generate_slide_id()}_{i}"
                
                # Get embedding for the chunk
                try:
                    embedding = doc_embeddings.embed_documents([chunk.page_content])[0]
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk {i}: {str(e)}")
                    continue
                
                # Generate timestamps
                current_time = datetime.utcnow().isoformat()
                
                # Generate metadata (excluding 'id' as it will be passed separately)
                metadata = {
                    "slide_name": f"{slide_data.slide_name} (Part {i+1})",
                    "course_name": slide_data.course_name,
                    "subject_name": slide_data.subject_name,
                    "description": f"{slide_data.description or ''} [Part {i+1} of {total_chunks}]",
                    "created_at": current_time,
                    "updated_at": current_time,
                    "file_name": file.filename,
                    "content_type": file.content_type or "application/octet-stream",
                    "chunk_index": i,
                    "total_chunks": total_chunks
                }
                
                # Store in Pinecone
                try:
                    pinecone_index.upsert(
                        vectors=[(
                            chunk_id,
                            embedding,
                            metadata
                        )]
                    )
                    
                    # Create SlideInDB object
                    slide = SlideInDB(
                        id=chunk_id,
                        **metadata,
                        vector_id=chunk_id,
                        embedding=embedding
                    )
                    results.append(slide)
                    logger.info(f"Successfully processed chunk {i+1}/{total_chunks}")
                    
                except Exception as e:
                    logger.error(f"Failed to store chunk {i} in Pinecone: {str(e)}")
                    continue
                
            except Exception as e:
                logger.error(f"Unexpected error processing chunk {i}: {str(e)}", exc_info=True)
                continue
                
        if not results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process any valid chunks from the file. Please check if the file contains extractable text content."
            )
            
        logger.info(f"Successfully processed {len(results)}/{total_chunks} chunks from file")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_slide: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

def get_slides(
    course_name: Optional[str] = None,
    subject_name: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
) -> List[Slide]:
    """Get all slides with optional filtering by course and subject."""
    try:
        # In a real implementation, you would query your database here
        # This is a simplified example
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_slide(slide_id: str) -> SlideInDB:
    """Get a specific slide by ID."""
    try:
        # In a real implementation, you would query your database here
        # This is a simplified example
        raise HTTPException(status_code=404, detail="Slide not found")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

def search_slides(query: str, limit: int = 5) -> List[SlideInDB]:
    """Search slides by semantic similarity to the query."""
    try:
        try:
            # Initialize query embeddings with the appropriate task type
            query_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                task_type="retrieval_query"  # Use retrieval_query for search queries
            )
            
            # Get embedding for the query
            query_embedding = query_embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process search query: {str(e)}"
            )
        
        # Query Pinecone
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True
        )
        
        # Convert results to SlideInDB objects
        slides = []
        for match in results.matches:
            metadata = match.metadata
            slides.append(SlideInDB(
                id=match.id,
                vector_id=match.id,
                embedding=match.values,
                **metadata
            ))
            
        return slides
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
