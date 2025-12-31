"""
Cloud API utilities for CLI commands.

Provides HTTP request helpers and response formatting for cloud API calls.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def get_cloud_server_url() -> str:
    """Get the cloud server URL from environment or default."""
    host = os.environ.get("SIMPLETUNER_HOST", "localhost")
    port = os.environ.get("SIMPLETUNER_PORT", "8001")
    scheme = "https" if os.environ.get("SIMPLETUNER_SSL_ENABLED") == "true" else "http"
    return f"{scheme}://{host}:{port}"


def cloud_api_request(
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Make a request to the cloud API.

    Args:
        method: HTTP method (GET, POST, PUT, PATCH, DELETE)
        endpoint: API endpoint (e.g., "/api/cloud/jobs")
        data: Optional JSON body for POST/PUT/PATCH requests
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response

    Raises:
        SystemExit: On connection or API errors
    """
    base_url = get_cloud_server_url()
    url = f"{base_url}{endpoint}"

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Add API key authentication if available
    api_key = os.environ.get("SIMPLETUNER_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key

    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    # Handle SSL verification
    ssl_context = None
    if url.startswith("https"):
        import ssl

        if os.environ.get("SIMPLETUNER_SSL_NO_VERIFY") == "true":
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode("utf-8"))
            error_msg = error_body.get("detail", str(e))
        except Exception:
            error_msg = str(e)

        # Provide helpful guidance for auth errors
        if e.code == 401 or "authentication required" in error_msg.lower():
            print("Error: Authentication required", file=sys.stderr)
            print("", file=sys.stderr)
            print("This command requires authentication.", file=sys.stderr)
            print("", file=sys.stderr)
            print("To authenticate:", file=sys.stderr)
            print("  1. Open the web UI at the server URL", file=sys.stderr)
            print("  2. Log in with your username and password", file=sys.stderr)
            print("  3. Click your avatar (top-right) > API Keys", file=sys.stderr)
            print("  4. Create a new API key and copy it", file=sys.stderr)
            print("  5. Set the environment variable:", file=sys.stderr)
            print("     export SIMPLETUNER_API_KEY=<your-api-key>", file=sys.stderr)
            sys.exit(1)

        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Error: Could not connect to server at {base_url}", file=sys.stderr)
        print(f"  Reason: {e.reason}", file=sys.stderr)
        print("  Make sure the SimpleTuner server is running: simpletuner server", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# --- Status Formatters ---


def format_job_status(status: str) -> str:
    """Format job status with emoji indicators."""
    status_icons = {
        "pending": "...",
        "queued": "[Q]",
        "uploading": "[^]",
        "running": "[>]",
        "completed": "[+]",
        "failed": "[X]",
        "cancelled": "[-]",
    }
    return f"{status_icons.get(status, '[?]')} {status}"


def format_bool(value: bool) -> str:
    """Format boolean for display."""
    return "yes" if value else "no"
