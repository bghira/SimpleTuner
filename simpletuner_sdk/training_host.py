from fastapi import APIRouter
from pydantic import BaseModel
from simpletuner_sdk.api_state import APIState
from simpletuner_sdk.thread_keeper import list_threads, terminate_thread


class TrainingHost:
    def __init__(self):
        self.router = APIRouter(prefix="/training")
        self.router.add_api_route("/", self.get_job, methods=["GET"])
        self.router.add_api_route("/", self.get_job, methods=["GET"])
        self.router.add_api_route("/state", self.get_host_state, methods=["GET"])
        self.router.add_api_route("/cancel", self.cancel_job, methods=["POST"])

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
