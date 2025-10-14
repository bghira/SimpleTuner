from simpletuner.helpers.training.multi_process import _get_rank as get_rank
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.webhooks.events import attach_timestamp, lifecycle_stage_event
from simpletuner.helpers.webhooks.handler import WebhookHandler

current_rank = get_rank()


class WebhookMixin:
    webhook_handler: WebhookHandler = None

    def set_webhook_handler(self, webhook_handler: WebhookHandler):
        self.webhook_handler = webhook_handler

    def send_progress_update(self, type: str, progress: int, total: int, current: int, readable_type: str = None):
        if total == 1:
            return
        if int(current_rank) != 0:
            return
        if not self.webhook_handler:
            return

        event = lifecycle_stage_event(
            key=type,
            label=readable_type,
            percent=progress,
            current=current,
            total=total,
            job_id=StateTracker.get_job_id(),
        )
        attach_timestamp(event)
        self.webhook_handler.send_raw(event, message_level="info", job_id=StateTracker.get_job_id())
