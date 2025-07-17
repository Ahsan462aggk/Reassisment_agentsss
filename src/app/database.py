import os
import logging
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variable to store the Pinecone client and index
_pinecone_client = None
_pinecone_index = None

def init_db() -> bool:
    """
    Initialize the database connection.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global _pinecone_client, _pinecone_index
    
    try:
        # Get required environment variables
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            logger.error("Missing PINECONE_API_KEY environment variable")
            return False
            
        # Initialize Pinecone client
        logger.info("Initializing Pinecone connection...")
        _pinecone_client = Pinecone(api_key=api_key)
        
        # Verify connection by listing indexes
        try:
            _pinecone_client.list_indexes()
            logger.info("Successfully connected to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

# Dimension for the embedding model (Google's text-embedding-004 uses 768 dimensions)
EMBEDDING_DIMENSION = 768

def get_pinecone_index():
    """
    Get or create a Pinecone index with the correct dimension for the embedding model.
    
    Returns:
        pinecone.Index: The Pinecone index object
        
    Raises:
        RuntimeError: If there's an error initializing the database or getting the index
    """
    global _pinecone_client, _pinecone_index
    
    if _pinecone_index is not None:
        return _pinecone_index
        
    # Initialize Pinecone if not already done
    if _pinecone_client is None and not init_db():
        raise RuntimeError("Failed to initialize database connection")
    
    index_name = os.getenv("PINECONE_INDEX", "reassesment")
    
    try:
        existing_indexes = _pinecone_client.list_indexes()
        index_exists = any(index.name == index_name for index in existing_indexes)
        
        if index_exists:
            # Get the existing index
            _pinecone_index = _pinecone_client.Index(index_name)
            logger.info(f"Using existing Pinecone index: {index_name}")
            
            # Verify the index has the correct dimension
            try:
                index_stats = _pinecone_index.describe_index_stats()
                index_dimension = index_stats.dimension
                
                if index_dimension != EMBEDDING_DIMENSION:
                    logger.warning(f"Deleting existing index with dimension {index_dimension} to create new one with dimension {EMBEDDING_DIMENSION}")
                    _pinecone_client.delete_index(index_name)
                    index_exists = False
                    logger.info(f"Deleted index {index_name}")
                    
            except Exception as e:
                logger.warning(f"Could not verify index dimension: {str(e)}")
                logger.warning("Will attempt to create a new index")
                
        else:
            # Create a new index with the correct dimension
            logger.info(f"Creating new Pinecone index: {index_name} with dimension {EMBEDDING_DIMENSION}")
            try:
                # Try with AWS us-east-1 first (free tier supported)
                try:
                    _pinecone_client.create_index(
                        name=index_name,
                        dimension=EMBEDDING_DIMENSION,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'  # Free tier supported region
                        )
                    )
                    logger.info("Successfully created index in AWS us-east-1")
                except Exception as aws_error:
                    logger.warning(f"Failed to create index in AWS us-east-1: {str(aws_error)}")
                    logger.info("Trying GCP us-central1...")
                    # Fall back to GCP if AWS fails
                    _pinecone_client.create_index(
                        name=index_name,
                        dimension=EMBEDDING_DIMENSION,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud='gcp',
                            region='us-central1'  # Common GCP region
                        )
                    )
                    logger.info("Successfully created index in GCP us-central1")
                
                _pinecone_index = _pinecone_client.Index(index_name)
                logger.info(f"Successfully created Pinecone index: {index_name}")
                
            except Exception as create_error:
                error_msg = f"Failed to create Pinecone index: {str(create_error)}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from create_error
        
        return _pinecone_index
        
    except Exception as e:
        error_msg = f"Failed to get Pinecone index: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
