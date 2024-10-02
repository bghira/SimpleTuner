from helpers.webhooks.handler import WebhookHandler
from helpers.training.state_tracker import StateTracker
from helpers.training.multi_process import _get_rank as get_rank

current_rank = get_rank()


class WebhookMixin:
    webhook_handler: WebhookHandler = None

    def set_webhook_handler(self, webhook_handler: WebhookHandler):
        self.webhook_handler = webhook_handler

    def send_progress_update(self, type: str, progress: int, total: int, current: int):
        if total == 1:
            return
        if int(current_rank) != 0:
            return
        progress = {
            "message_type": "progress_update",
            "message": {
                "progress_type": type,
                "progress": progress,
                "total_elements": total,
                "current_estimated_index": current,
            },
        }

        self.webhook_handler.send_raw(
            progress, "progress_update", job_id=StateTracker.get_job_id()
        )
