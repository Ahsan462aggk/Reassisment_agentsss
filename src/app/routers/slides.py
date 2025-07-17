from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import List, Optional
from ..controllers import slide_controller
from ..schemas.slide import Slide, SlideCreate, SlideInDB

router = APIRouter(
    prefix="/slides",
    tags=["slides"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=List[SlideInDB])
async def upload_slide(
    file: UploadFile = File(...),
    slide_name: str = Form(...),
    course_name: str = Form(...),
    subject_name: str = Form(...),
    description: Optional[str] = Form(None)
):
    """
    Upload a new slide with metadata.
    
    Returns a list of SlideInDB objects, one for each chunk of the processed file.
    """
    try:
        # Read file content and create slide data
        file_content = await file.read()
        
        slide_data = SlideCreate(
            slide_name=slide_name,
            course_name=course_name,
            subject_name=subject_name,
            description=description,
            file_content=file_content,
            file_name=file.filename,
            content_type=file.content_type
        )
        
        # Reset file pointer for the controller to read it again
        await file.seek(0)
        
        # Process the file and return the list of slides
        slides = await slide_controller.upload_slide(slide_data, file)
        
        # Convert Pydantic models to dict to ensure proper serialization
        return [slide.dict() for slide in slides]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )

@router.get("/", response_model=List[Slide])
async def get_slides(
    course_name: Optional[str] = None,
    subject_name: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
):
    """Get all slides with optional filtering by course and subject."""
    return slide_controller.get_slides(course_name, subject_name, limit, offset)

@router.get("/{slide_id}", response_model=SlideInDB)
async def get_slide(slide_id: str):
    """Get a specific slide by ID."""
    return slide_controller.get_slide(slide_id)

@router.get("/search/", response_model=List[SlideInDB])
async def search_slides(query: str, limit: int = 5):
    """Search slides by semantic similarity to the query."""
    return slide_controller.search_slides(query, limit)
