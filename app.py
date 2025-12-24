from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers.analyze import router as analyze_router

app = FastAPI(title="Shopinion AI API")

#Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],  # Includes OPTIONS, GET, POST, etc.
    allow_headers=["*"],
)

#Include your analysis route
app.include_router(analyze_router)