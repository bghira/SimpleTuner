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

# Import route modules
from simpletuner.simpletuner_sdk.server.routes.checkpoints import router as checkpoints_router
from simpletuner.simpletuner_sdk.server.routes.datasets import router as dataset_router
from simpletuner.simpletuner_sdk.server.routes.publishing import router as publishing_router
from simpletuner.simpletuner_sdk.server.routes.web import router as web_router
from simpletuner.simpletuner_sdk.training_host import TrainingHost


# Pydantic models for request/response
class TrainerConfig(BaseModel):
    trainer_config: Dict[str, Any]
    dataloader_config: List[Dict[str, Any]]
    webhook_config: Dict[str, Any]
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

# Include web interface router (uses TabService with all tabs)
app.include_router(web_router)

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

# Dataset blueprint + plan API
app.include_router(dataset_router)

# Publishing API for HuggingFace Hub operations
app.include_router(publishing_router)

# Checkpoints API for checkpoint management operations
app.include_router(checkpoints_router)


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

    # Check for SSL configuration
    ssl_enabled = os.environ.get("SIMPLETUNER_SSL_ENABLED", "false").lower() == "true"
    ssl_keyfile = os.environ.get("SIMPLETUNER_SSL_KEYFILE")
    ssl_certfile = os.environ.get("SIMPLETUNER_SSL_CERTFILE")

    # Configure uvicorn
    uvicorn_config = {"app": app, "host": "0.0.0.0", "port": 8001, "reload": True, "log_level": "info"}

    if ssl_enabled and ssl_keyfile and ssl_certfile:
        uvicorn_config.update({"ssl_keyfile": ssl_keyfile, "ssl_certfile": ssl_certfile})
        print("SSL enabled for service worker")
    else:
        print("SSL disabled for service worker")

    # Run the server
    uvicorn.run(**uvicorn_config)


# Main entry point
if __name__ == "__main__":
    main()
