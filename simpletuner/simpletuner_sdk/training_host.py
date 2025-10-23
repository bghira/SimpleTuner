from fastapi import APIRouter
from pydantic import BaseModel

from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.thread_keeper import list_threads, terminate_thread


class TrainingHost:
    def __init__(self):
        self.router = APIRouter(prefix="/training")
        self.router.add_api_route("/", self.get_job, methods=["GET"])
        self.router.add_api_route("/", self.get_job, methods=["GET"])
        self.router.add_api_route("/state", self.get_host_state, methods=["GET"])
        self.router.add_api_route("/cancel", self.cancel_job, methods=["POST"])
        self.router.add_api_route("/status/{job_id}", self.get_job_status, methods=["GET"])
        self.router.add_api_route("/jobs", self.list_jobs, methods=["GET"])

    def get_host_state(self):
        """
        Get the current host status using APIState
        """
        return {"result": APIState.get_state(), "job_list": list_threads()}

    def get_job(self):
        """
        Returns just the currently active job from APIState
        """
        return {"result": APIState.get_job()}

    def cancel_job(self):
        """
        Cancel the currently active job
        """
        trainer = APIState.get_trainer()
        if not trainer:
            return {"status": False, "result": "No job to cancel"}
        trainer.abort()
        is_terminated = terminate_thread(job_id=APIState.get_state("current_job_id"))
        APIState.set_trainer(None)
        APIState.cancel_job()

        return {"result": "Job marked for cancellation."}

    def get_job_status(self, job_id: str):
        """
        Get status of a training job
        """
        active_jobs = APIState.get_active_jobs()
        if job_id not in active_jobs:
            return {"detail": f"Job ID '{job_id}' not found"}, 404

        job = active_jobs[job_id]
        return {
            "job_id": job_id,
            "status": job["status"],
            "start_time": job["start_time"],
            "end_time": job.get("end_time"),
        }

    def list_jobs(self):
        """
        List all active training jobs
        """
        active_jobs = APIState.get_active_jobs()
        return {
            "jobs": [
                {
                    "job_id": job_id,
                    "status": job["status"],
                    "start_time": job["start_time"],
                }
                for job_id, job in active_jobs.items()
            ]
        }
