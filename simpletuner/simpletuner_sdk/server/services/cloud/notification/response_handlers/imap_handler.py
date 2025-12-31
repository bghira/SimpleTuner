"""IMAP IDLE handler for email reply processing."""

from __future__ import annotations

import asyncio
import email
import logging
from email.message import Message
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..models import ChannelConfig, ResponseAction
from .parser import extract_response_token, parse_approval_response

if TYPE_CHECKING:
    from ..notification_service import NotificationService

logger = logging.getLogger(__name__)


class IMAPResponseHandler:
    """Listen for email replies to approval notifications using IMAP IDLE.

    Features:
    - IMAP IDLE (PUSH) for real-time delivery
    - Parse affirmative/rejection responses
    - Handle unknown responses with helpful reply
    - Ignore unrecognized senders (avoid bounce spam)
    - Automatic reconnection on connection loss
    """

    def __init__(
        self,
        config: ChannelConfig,
        notification_service: "NotificationService",
    ):
        """Initialize the IMAP handler.

        Args:
            config: Email channel configuration with IMAP settings
            notification_service: NotificationService for callbacks
        """
        self._config = config
        self._service = notification_service
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._client: Any = None
        self._reconnect_delay = 30  # seconds

    @property
    def is_running(self) -> bool:
        """Check if handler is running."""
        return self._running and self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start IMAP IDLE loop in background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Started IMAP handler for channel %d (%s)",
            self._config.id,
            self._config.name,
        )

    async def stop(self) -> None:
        """Stop IMAP IDLE loop."""
        self._running = False

        if self._client:
            try:
                await self._client.logout()
            except Exception:
                pass
            self._client = None

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Stopped IMAP handler for channel %d", self._config.id)

    async def _run_loop(self) -> None:
        """Main IMAP IDLE loop with reconnection."""
        while self._running:
            try:
                await self._connect_and_idle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "IMAP connection error for channel %d: %s, reconnecting in %ds",
                    self._config.id,
                    exc,
                    self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)

    async def _connect_and_idle(self) -> None:
        """Connect to IMAP server and enter IDLE mode."""
        try:
            import aioimaplib
        except ImportError:
            logger.error("aioimaplib not installed. Install with: pip install aioimaplib")
            self._running = False
            return

        # Get password from secrets
        password = await self._get_imap_password()
        if not password:
            logger.error("No IMAP password found for channel %d", self._config.id)
            self._running = False
            return

        # Connect
        if self._config.imap_use_ssl:
            self._client = aioimaplib.IMAP4_SSL(
                host=self._config.imap_host,
                port=self._config.imap_port,
            )
        else:
            self._client = aioimaplib.IMAP4(
                host=self._config.imap_host,
                port=self._config.imap_port,
            )

        await self._client.wait_hello_from_server()
        await self._client.login(self._config.imap_username, password)
        await self._client.select(self._config.imap_folder or "INBOX")

        logger.info(
            "Connected to IMAP server %s for channel %d",
            self._config.imap_host,
            self._config.id,
        )

        # Enter IDLE loop
        while self._running:
            try:
                # Start IDLE with 5 minute timeout (re-issue to keep alive)
                idle_task = await self._client.idle_start(timeout=300)

                # Wait for new messages or timeout
                msg = await self._client.wait_server_push()

                # Stop IDLE to process
                self._client.idle_done()
                await asyncio.wait_for(idle_task, timeout=5)

                # Check for new messages
                if msg and "EXISTS" in str(msg):
                    await self._process_new_messages()

            except asyncio.TimeoutError:
                # IDLE timeout, just restart
                continue
            except Exception as exc:
                logger.warning("IDLE error: %s", exc)
                raise

    async def _process_new_messages(self) -> None:
        """Process new messages in inbox."""
        try:
            # Search for unseen messages
            _, data = await self._client.search("UNSEEN")
            if not data or not data[0]:
                return

            message_nums = data[0].split()
            logger.debug("Found %d new messages", len(message_nums))

            for num in message_nums:
                await self._process_message(num)

        except Exception as exc:
            logger.error("Error processing messages: %s", exc)

    async def _process_message(self, message_num: bytes) -> None:
        """Process a single message.

        Args:
            message_num: IMAP message number
        """
        try:
            # Fetch the message
            _, data = await self._client.fetch(message_num, "(RFC822)")
            if not data or not data[1]:
                return

            # Parse email
            raw_email = data[1]
            msg = email.message_from_bytes(raw_email)

            # Process the response
            action = await self.process_response(msg, {})

            if action:
                # Mark as processed
                await self._client.store(message_num, "+FLAGS", "\\Seen")

                # Handle the action
                if not action.is_unknown():
                    await self._service.handle_response(action)
                else:
                    # Send unknown response help
                    await self._service._send_unknown_response_reply(action)

        except Exception as exc:
            logger.error("Error processing message %s: %s", message_num, exc)

    async def process_response(
        self,
        raw_response: Any,
        context: Dict[str, Any],
    ) -> Optional[ResponseAction]:
        """Parse email reply and determine approval action.

        Args:
            raw_response: email.message.Message object
            context: Additional context

        Returns:
            ResponseAction or None if not a valid response
        """
        if not isinstance(raw_response, Message):
            return None

        # Extract sender
        sender = raw_response.get("From", "")
        if "<" in sender:
            # Extract email from "Name <email>" format
            import re

            match = re.search(r"<([^>]+)>", sender)
            if match:
                sender = match.group(1)

        # Extract subject and token
        subject = raw_response.get("Subject", "")
        in_reply_to = raw_response.get("In-Reply-To", "")
        token = extract_response_token(subject, in_reply_to)

        if not token:
            logger.debug("No response token found in email from %s", sender)
            return None

        # Verify this is a response to our notification
        from ..notification_store import NotificationStore

        store = NotificationStore()
        pending = await store.get_pending_by_token(token)

        if not pending:
            logger.debug("No pending response found for token %s", token[:8])
            return None

        # Check if sender is authorized
        sender_lower = sender.lower()
        authorized = any(auth.lower() == sender_lower for auth in pending.authorized_senders)

        if not authorized:
            logger.debug("Sender %s not authorized for token %s", sender, token[:8])
            return None

        # Extract body
        body = self._extract_body(raw_response)

        # Parse response
        action = parse_approval_response(body)

        return ResponseAction(
            action=action,
            approval_request_id=pending.approval_request_id,
            sender_email=sender,
            raw_body=body,
        )

    def _extract_body(self, msg: Message) -> str:
        """Extract plain text body from email message.

        Args:
            msg: Email message

        Returns:
            Plain text body
        """
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                # Prefer plain text
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body = payload.decode(charset, errors="replace")
                        break
                elif content_type == "text/html" and not body:
                    # Fall back to HTML if no plain text
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        html = payload.decode(charset, errors="replace")
                        # Simple HTML stripping
                        import re

                        body = re.sub(r"<[^>]+>", " ", html)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body = payload.decode(charset, errors="replace")

        return body.strip()

    async def _get_imap_password(self) -> Optional[str]:
        """Get IMAP password from secrets manager.

        Returns:
            Password or None
        """
        if not self._config.imap_password_key:
            return None

        try:
            from ...secrets import get_secrets_manager

            secrets = get_secrets_manager()
            return secrets.get(self._config.imap_password_key)
        except Exception as exc:
            logger.debug("Could not get IMAP password: %s", exc)
            return None
