import asyncio
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from simpletuner.simpletuner_sdk.configuration import Configuration

# Import the WebInterface class
from simpletuner.simpletuner_sdk.interface import WebInterface
from simpletuner.simpletuner_sdk.training_host import TrainingHost


# Pydantic models for request/response
class TrainerConfig(BaseModel):
    trainer_config: Dict[str, Any]
    dataloader_config: List[Dict[str, Any]]
    webhooks_config: Dict[str, Any]
    job_id: str


class CancelRequest(BaseModel):
    job_id: str


# Create FastAPI app
app = FastAPI(title="SimpleTuner Training API")

# Configure CORS
origins = [
    "http://localhost:8000",
    "http://localhost:8001",
    "http://localhost:8002",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize and include web interface
web_interface = WebInterface()
app.include_router(web_interface.router)

# Store active training jobs
active_jobs: Dict[str, Dict[str, Any]] = {}


# Root redirect
@app.get("/")
async def root():
    """Redirect to web interface"""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/web/trainer")


config_controller = Configuration()
training_host = TrainingHost()

#####################################################
#   configuration controller for argument handling  #
#####################################################
app.include_router(config_controller.router)

#####################################################
#   traininghost controller for training job mgmt   #
#####################################################
app.include_router(training_host.router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def main():
    """Main entry point for the server worker."""
    import uvicorn

    # Create necessary directories
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True, log_level="info")


# Main entry point
if __name__ == "__main__":
    main()
