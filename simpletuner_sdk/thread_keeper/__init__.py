from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict
import threading

# Global executor for managing threads
executor = ThreadPoolExecutor(max_workers=1)  # Adjust max_workers based on your needs
thread_registry: Dict[str, Future] = {}  # A dictionary to store threads with their IDs
lock = threading.Lock()  # For thread-safe operations on the registry


def submit_job(job_id: str, func, *args, **kwargs):
    with lock:
        if job_id in thread_registry:
            raise Exception(f"Job with ID {job_id} is already running.")
        future = executor.submit(func, *args, **kwargs)
        thread_registry[job_id] = future


def get_thread_status(job_id: str) -> str:
    with lock:
        future = thread_registry.get(job_id)
        if not future:
            return "No such job."
        if future.running():
            return "Running"
        elif future.done():
            if future.exception():
                return f"Failed: {future.exception()}"
            return "Completed"
        return "Pending"


def terminate_thread(job_id: str) -> bool:
    with lock:
        executor.shutdown(wait=False)
        future = thread_registry.get(job_id)
        if not future:
            print(f"thread {job_id} not found")
            return None
        cancelled = future.cancel()
        if cancelled:
            del thread_registry[job_id]
        return cancelled


def list_threads():
    output_data = {}
    for item in thread_registry:
        output_data[item] = get_thread_status(item)
    return output_data
