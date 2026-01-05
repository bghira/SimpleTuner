"""
Worker command for SimpleTuner CLI.

Connects to an orchestrator panel and executes training jobs.
"""

import argparse


def cmd_worker(args: argparse.Namespace) -> int:
    """Handle worker command - connect to orchestrator as a worker agent."""
    orchestrator_url = getattr(args, "orchestrator_url", None)
    worker_token = getattr(args, "worker_token", None)
    name = getattr(args, "name", None)
    persistent = getattr(args, "persistent", False)
    verbose = getattr(args, "verbose", False)

    # Build namespace for worker agent
    import asyncio
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        from simpletuner.worker_agent import WorkerAgent, WorkerConfig

        # Get config from args or environment
        if orchestrator_url and worker_token:
            config = WorkerConfig(
                orchestrator_url=orchestrator_url.rstrip("/"),
                worker_token=worker_token,
                name=name or __import__("socket").gethostname(),
                persistent=persistent,
            )
        else:
            config = WorkerConfig.from_env()

        agent = WorkerAgent(config)
        asyncio.run(agent.run())
        return 0
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nWorker stopped by user.")
        return 130
    except Exception as e:
        print(f"Error running worker: {e}")
        import traceback

        traceback.print_exc()
        return 1
