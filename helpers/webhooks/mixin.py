from helpers.webhooks.handler import WebhookHandler

class WebhookMixin:
    ebhook_handler: WebhookHandler = None

    def set_webhook_handler(self, webhook_handler: WebhookHandler):
        self.webhook_handler = webhook_handler

    def send_progress_update(self, type:str, progress: int, total:int, current:int):
        progress = {
            "message_type": "progress_update",
            "message": {
                "progress_type": type,
                "progress": progress,
                "total_elements": total,
                "current_estimated_index": current,
            },
        }

        self.webhook_handler.send_raw(progress, "progress_update")
