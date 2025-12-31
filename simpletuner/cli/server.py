"""
Server command for SimpleTuner CLI.

Handles starting the web server with various modes and SSL configuration.
"""

import datetime
import os
from pathlib import Path
from typing import Optional

from .common import _find_webui_state_file, _get_webui_configs_dir, _validate_environment_config


def _setup_ssl_config(ssl_key: Optional[str] = None, ssl_cert: Optional[str] = None) -> Optional[dict]:
    """Set up SSL configuration, generating certificates if needed."""

    if ssl_key and ssl_cert:
        key_path = Path(ssl_key).expanduser()
        cert_path = Path(ssl_cert).expanduser()

        if not key_path.exists():
            print(f"Error: SSL key file not found: {key_path}")
            return None
        if not cert_path.exists():
            print(f"Error: SSL certificate file not found: {cert_path}")
            return None

        print(f"Using provided SSL certificate: {cert_path}")
        print(f"Using provided SSL key: {key_path}")
        return {"keyfile": str(key_path), "certfile": str(cert_path)}

    ssl_dir = Path.home() / ".simpletuner" / "ssl"
    ssl_dir.mkdir(parents=True, exist_ok=True)

    key_path = ssl_dir / "server.key"
    cert_path = ssl_dir / "server.crt"

    if key_path.exists() and cert_path.exists():
        print(f"Using existing SSL certificate: {cert_path}")
        print(f"Using existing SSL key: {key_path}")
        return {"keyfile": str(key_path), "certfile": str(cert_path)}

    print("Generating self-signed SSL certificate...")

    try:
        import ipaddress

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SimpleTuner"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("localhost"),
                        x509.DNSName("*.localhost"),
                        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                        x509.IPAddress(ipaddress.IPv4Address("0.0.0.0")),
                    ]
                ),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )

        with open(key_path, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        key_path.chmod(0o600)
        cert_path.chmod(0o644)

        print(f"Generated SSL certificate: {cert_path}")
        print(f"Generated SSL key: {key_path}")
        print("Note: This is a self-signed certificate. Browsers will show security warnings.")

        return {"keyfile": str(key_path), "certfile": str(cert_path)}

    except ImportError:
        print("Error: cryptography package required for SSL certificate generation.")
        print("Install it with: pip install cryptography")
        return None
    except Exception as e:
        print(f"Error generating SSL certificate: {e}")
        return None


def cmd_server(args) -> int:
    """Handle server command."""
    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", None)
    reload = getattr(args, "reload", False)
    mode = getattr(args, "mode", "unified")
    ssl = getattr(args, "ssl", False)
    ssl_key = getattr(args, "ssl_key", None)
    ssl_cert = getattr(args, "ssl_cert", None)
    ssl_no_verify = getattr(args, "ssl_no_verify", False)
    env = getattr(args, "env", None)

    if not os.environ.get("SIMPLETUNER_CONFIG_DIR"):
        webui_configs_dir = _get_webui_configs_dir()
        if webui_configs_dir:
            os.environ["SIMPLETUNER_CONFIG_DIR"] = str(webui_configs_dir)

    if port is None:
        if mode == "trainer":
            port = 8001
        elif mode == "callback":
            port = 8002
        else:
            port = 8001

    if env:
        os.environ["ENV"] = env

        from simpletuner.helpers.configuration.loader import load_env_variables

        load_env_variables()

        config_backend_env = os.environ.get(
            "SIMPLETUNER_CONFIG_BACKEND",
            os.environ.get("CONFIG_BACKEND", os.environ.get("CONFIG_TYPE")),
        )
        config_path_env = os.environ.get("CONFIG_PATH")

        try:
            _validate_environment_config(env, config_backend_env, config_path_env)
        except FileNotFoundError as validation_error:
            print(f"Error: {validation_error}")
            return 1

        os.environ["SIMPLETUNER_SERVER_AUTOSTART_ENV"] = env

    ssl_config = None
    if ssl:
        ssl_config = _setup_ssl_config(ssl_key, ssl_cert)
        if not ssl_config:
            return 1

    protocol = "https" if ssl_config else "http"
    print(f"Starting SimpleTuner {mode} server:")
    if mode in {"trainer", "unified"}:
        print(f"> API: {protocol}://{host}:{port}/api")
        print(f"> Web: {protocol}://{host}:{port}/web")
    if env:
        print(f"> Environment: {env}")
        print("> Training will start automatically once server is ready.")

    os.environ["SIMPLETUNER_SSL_ENABLED"] = "true" if ssl_config else "false"
    os.environ["SIMPLETUNER_SSL_NO_VERIFY"] = "true" if ssl_no_verify else "false"
    os.environ["SIMPLETUNER_WEBHOOK_HOST"] = host
    os.environ["SIMPLETUNER_WEBHOOK_PORT"] = str(port)
    os.environ["SIMPLETUNER_WEB_MODE"] = "1"

    if ssl_config:
        os.environ["SIMPLETUNER_SSL_KEYFILE"] = ssl_config["keyfile"]
        os.environ["SIMPLETUNER_SSL_CERTFILE"] = ssl_config["certfile"]

    from simpletuner.simpletuner_sdk.server.utils.paths import get_config_directory, get_template_directory

    os.environ.setdefault("TEMPLATE_DIR", str(get_template_directory()))

    config_dir = get_config_directory()
    os.environ.setdefault("SIMPLETUNER_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("SIMPLETUNER_SERVER_ROOT_PID", str(os.getpid()))
    os.environ.setdefault("SIMPLETUNER_STRICT_MODEL_IMPORTS", "1")

    try:
        import uvicorn

        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        server_mode = {"trainer": ServerMode.TRAINER, "callback": ServerMode.CALLBACK, "unified": ServerMode.UNIFIED}.get(
            mode, ServerMode.UNIFIED
        )

        app = create_app(mode=server_mode, ssl_no_verify=ssl_no_verify)

        uvicorn_config = {
            "app": app,
            "host": host,
            "port": port,
            "reload": reload,
            "log_level": "warning",
            "timeout_graceful_shutdown": 5,  # Don't wait forever for SSE connections
        }

        if ssl_config:
            uvicorn_config.update({"ssl_keyfile": ssl_config["keyfile"], "ssl_certfile": ssl_config["certfile"]})

        uvicorn.run(**uvicorn_config)

        # Give background threads (e.g., aiosqlite workers) time to exit gracefully
        import threading
        import time

        deadline = time.time() + 2.0
        while time.time() < deadline:
            non_daemon_threads = [
                t for t in threading.enumerate() if not t.daemon and t.is_alive() and t.name != "MainThread"
            ]
            if not non_daemon_threads:
                break
            time.sleep(0.1)

        # Force exit if background threads are still hanging
        remaining = [t for t in threading.enumerate() if not t.daemon and t.is_alive() and t.name != "MainThread"]
        if remaining:
            os._exit(0)

        return 0
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        return 130
    except ImportError as e:
        print(f"Error importing server dependencies: {e}")
        print("Make sure FastAPI and uvicorn are installed.")
        return 1
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback

        traceback.print_exc()
        return 1
