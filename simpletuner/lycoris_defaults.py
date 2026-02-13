"""Shared LyCORIS default configurations used by both training and the WebUI."""

from __future__ import annotations

lycoris_defaults = {
    "lora": {
        "algo": "lora",
        "multiplier": 1.0,
        "linear_dim": 64,
        "linear_alpha": 32,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
    "loha": {
        "algo": "loha",
        "multiplier": 1.0,
        "linear_dim": 32,
        "linear_alpha": 16,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
    "lokr": {
        "algo": "lokr",
        "multiplier": 1.0,
        "linear_dim": 10000,  # Full dimension
        "linear_alpha": 1,  # Ignored in full dimension
        "factor": 16,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
    "full": {
        "algo": "full",
        "multiplier": 1.0,
        "linear_dim": 1024,  # Example full matrix size
        "linear_alpha": 512,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
        },
    },
    "ia3": {
        "algo": "ia3",
        "multiplier": 1.0,
        "linear_dim": None,  # No network arguments
        "linear_alpha": None,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
        },
    },
    "dylora": {
        "algo": "dylora",
        "multiplier": 1.0,
        "linear_dim": 128,
        "linear_alpha": 64,
        "block_size": 1,  # Update one row/col per step
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
    "diag-oft": {
        "algo": "diag-oft",
        "multiplier": 1.0,
        "linear_dim": 64,  # Block size
        "constraint": False,
        "rescaled": False,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
    "boft": {
        "algo": "boft",
        "multiplier": 1.0,
        "linear_dim": 64,  # Block size
        "constraint": False,
        "rescaled": False,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
    "tlora": {
        "algo": "tlora",
        "multiplier": 1.0,
        "linear_dim": 64,
        "linear_alpha": 32,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
        },
    },
}
