from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import router
from .routers import slides as slides_router

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include router
app.include_router(slides_router.router)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": " Teacher Dasboard API is running"}
