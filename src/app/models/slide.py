from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class SlideBase(BaseModel):
    slide_name: str
    course_name: str
    subject_name: str
    description: Optional[str] = None

class SlideCreate(SlideBase):
    file_content: bytes

class Slide(SlideBase):
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class SlideInDB(Slide):
    vector_id: str
