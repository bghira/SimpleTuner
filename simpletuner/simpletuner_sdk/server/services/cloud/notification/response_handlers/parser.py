"""Parse approval responses from email body."""

from __future__ import annotations

import re
from typing import Optional

# Affirmative patterns - multilingual support
AFFIRMATIVE_PATTERNS = [
    # English
    r"\b(?:ok|okay|yes|yep|yeah|yea|sure|approved?|lgtm|go\s*ahead|proceed|accept(?:ed)?)\b",
    # French
    r"\b(?:d'accord|d accord|oui|ouais|bien\s+s[uû]r)\b",
    # Spanish
    r"\b(?:si|s[íi]|vale|bueno|de\s*acuerdo|aprobado)\b",
    # German
    r"\b(?:ja|jawohl|klar|einverstanden|genehmigt)\b",
    # Italian
    r"\b(?:si|s[ìi]|va\s*bene|certo)\b",
    # Portuguese
    r"\b(?:sim|claro|aprovado)\b",
    # Russian (transliterated)
    r"\b(?:da|khorosho|odobren[ao]?)\b",
    # Japanese (transliterated)
    r"\b(?:hai|un|ok[ae]y?|shouchi)\b",
    # Chinese (transliterated)
    r"\b(?:hao|shi|keyi|tongyi)\b",
    # Korean (transliterated)
    r"\b(?:ne|ye|joayo)\b",
    # Common abbreviations
    r"\b(?:y|yy|ys|ya|yup)\b",
    # Thumbs up emoji
    r"(?:\U0001F44D|\U0001F44C|\u2705|\u2714)",
]

# Rejection patterns - multilingual support
REJECTION_PATTERNS = [
    # English
    r"\b(?:no|nope|nah|reject(?:ed)?|denied?|decline[sd]?|stop|cancel(?:led)?|refuse[sd]?)\b",
    # French
    r"\b(?:non|pas\s+d'accord|rejet[eé]?|refus[eé]?)\b",
    # Spanish
    r"\b(?:no|nada|rechazado|denegado)\b",
    # German
    r"\b(?:nein|abgelehnt|verweigert)\b",
    # Italian
    r"\b(?:no|rifiutato|negato)\b",
    # Portuguese
    r"\b(?:n[aã]o|rejeitado|negado)\b",
    # Russian (transliterated)
    r"\b(?:net|nyet|niet|otklonen[ao]?)\b",
    # Japanese (transliterated)
    r"\b(?:iie|dame|kyohi)\b",
    # Korean (transliterated)
    r"\b(?:aniyo|ani)\b",
    # Common abbreviations
    r"\b(?:n|nn|na)\b",
    # Thumbs down emoji
    r"(?:\U0001F44E|\u274C|\u2716)",
]


def parse_approval_response(body: str) -> str:
    """Parse email body for approval/rejection response.

    The parser looks for clear affirmative or rejection patterns
    in the first few words of the response. It prioritizes:
    1. Single-word responses (most common for mobile)
    2. First line of the email
    3. Full body as fallback

    Args:
        body: Email body text

    Returns:
        "approve", "reject", or "unknown"
    """
    # Clean and normalize the body
    cleaned = _clean_email_body(body)

    if not cleaned:
        return "unknown"

    # Try to match against the first line only (most reliable)
    first_line = cleaned.split("\n")[0].strip()

    # Single word responses are most reliable
    if len(first_line.split()) <= 3:
        # Check affirmative patterns first
        for pattern in AFFIRMATIVE_PATTERNS:
            if re.search(pattern, first_line, re.IGNORECASE | re.UNICODE):
                return "approve"

        # Then check rejection patterns
        for pattern in REJECTION_PATTERNS:
            if re.search(pattern, first_line, re.IGNORECASE | re.UNICODE):
                return "reject"

    # If first line didn't match, check full body
    # (less reliable as it may contain quoted text)
    for pattern in AFFIRMATIVE_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE | re.UNICODE):
            return "approve"

    for pattern in REJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE | re.UNICODE):
            return "reject"

    return "unknown"


def _clean_email_body(body: str) -> str:
    """Clean email body for parsing.

    Removes:
    - Quoted text (lines starting with >)
    - Email signatures (lines after --)
    - HTML tags
    - Excessive whitespace

    Args:
        body: Raw email body

    Returns:
        Cleaned body text
    """
    if not body:
        return ""

    lines = []
    in_signature = False

    for line in body.split("\n"):
        # Strip whitespace
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Stop at signature delimiter
        if line == "--" or line == "-- ":
            in_signature = True
            continue

        if in_signature:
            continue

        # Skip quoted lines
        if line.startswith(">"):
            continue

        # Skip common email artifacts
        if line.startswith("On ") and ("wrote:" in line or "said:" in line):
            # Start of quoted section
            break

        if line.startswith("From:") or line.startswith("Sent:"):
            # Forwarded email header
            break

        lines.append(line)

    result = "\n".join(lines)

    # Remove HTML tags if any
    result = re.sub(r"<[^>]+>", " ", result)

    # Normalize whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result


def get_unknown_response_message() -> str:
    """Get helpful reply message for unrecognized responses.

    Returns:
        Help message explaining valid responses
    """
    return """I couldn't understand your response.

To take action on this approval request, please reply with one of:

**To APPROVE:**
- "ok", "yes", "approved", "lgtm", "proceed"
- Or in your language: "oui", "si", "ja", "da", "hai"

**To REJECT:**
- "no", "reject", "denied", "decline", "cancel"
- Or in your language: "non", "nein", "niet", "iie"

You can also log in to the SimpleTuner admin panel to review and take action on pending approvals.

---
This is an automated message from SimpleTuner Cloud Training.
"""


def extract_response_token(subject: str, in_reply_to: Optional[str] = None) -> Optional[str]:
    """Extract response token from email subject or headers.

    The token is embedded in the subject line as [REF:xxxxxxxx].

    Args:
        subject: Email subject line
        in_reply_to: In-Reply-To header value

    Returns:
        Response token or None if not found
    """
    # Try to find token in subject
    match = re.search(r"\[REF:([a-zA-Z0-9_-]+)\]", subject)
    if match:
        return match.group(1)

    # Could also check Message-ID patterns in in_reply_to
    # if we embedded tokens there

    return None
