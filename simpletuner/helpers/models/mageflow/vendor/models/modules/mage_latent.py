from __future__ import annotations

import hashlib
import os
from math import erfc, sqrt

import numpy as np
import torch

DEFAULT_GS_PAYLOAD = "MageFlow"


_ENV_KEY = "MAGEFLOW_GS_KEY"
_ENV_KEYFILE = "MAGEFLOW_GS_KEY_FILE"
_DEFAULT_KEYFILE = os.path.expanduser("~/.mageflow/gs_key")
DEFAULT_GS_KEY = 20260720

# Payload length in bits. Short + heavily replicated across the latent.
_MSG_BITS = 256


def _key_to_int(value) -> int:
    """Normalize a key (int / digit-string / passphrase) to a non-negative int.

    A pure integer or all-digits string is used directly; anything else is
    treated as a passphrase and hashed (SHA-256) into a 256-bit integer.
    """
    if isinstance(value, int):
        return abs(value)
    s = str(value).strip()
    if not s:
        raise ValueError("empty Gaussian-Shading key")
    if s.lstrip("-").isdigit():
        return abs(int(s))
    return int.from_bytes(hashlib.sha256(s.encode()).digest(), "big")


def resolve_gs_key(explicit=None):
    if explicit is not None:
        return _key_to_int(explicit)
    env = os.environ.get(_ENV_KEY)
    if env and env.strip():
        return _key_to_int(env)
    keyfile = os.environ.get(_ENV_KEYFILE) or _DEFAULT_KEYFILE
    try:
        with open(keyfile) as fh:
            content = fh.read().strip()
        if content:
            return _key_to_int(content)
    except OSError:
        pass
    return _key_to_int(DEFAULT_GS_KEY)


def _payload_to_bits(payload: str, n_bits: int = _MSG_BITS) -> np.ndarray:
    """Deterministically expand an arbitrary string into an ``n_bits`` bit vector."""
    out: list[int] = []
    counter = 0
    while len(out) < n_bits:
        digest = hashlib.sha256(f"{payload}:{counter}".encode()).digest()
        for byte in digest:
            for k in range(8):
                out.append((byte >> k) & 1)
        counter += 1
    return np.asarray(out[:n_bits], dtype=np.int64)


def _pad_and_pos(n: int, key, n_bits: int = _MSG_BITS):
    """Key-seeded per-entry XOR pad and message-index map (length ``n``)."""
    rng = np.random.default_rng(_key_to_int(key))
    pad = rng.integers(0, 2, size=n).astype(np.int64)  # XOR mask
    pos = rng.integers(0, n_bits, size=n).astype(np.int64)  # msg index per entry
    return pad, pos


def encode_noise(shape, *, key, seed: int = 0, device=None, dtype=torch.bfloat16) -> torch.Tensor:
    C, H, W = shape
    n = C * H * W
    msg = _payload_to_bits(DEFAULT_GS_PAYLOAD)
    pad, pos = _pad_and_pos(n, key)
    target_half = (msg[pos] ^ pad).astype(np.float64)  # {0,1} per entry

    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    u = torch.rand(n, generator=gen, dtype=torch.float64)  # U(0,1) magnitudes
    half = torch.from_numpy(target_half)
    arg = ((half + u) / 2.0).clamp(1e-6, 1.0 - 1e-6)
    z = torch.special.ndtri(arg)  # inverse normal CDF
    z = z.reshape(1, C, H, W)
    return z.to(device=device, dtype=dtype)


def decode_bits(noise: torch.Tensor, *, key) -> dict:
    z = noise.detach().float().reshape(-1).cpu()
    n = int(z.numel())
    msg = _payload_to_bits(DEFAULT_GS_PAYLOAD)
    pad, pos = _pad_and_pos(n, key)

    observed_half = (z > 0).numpy().astype(np.int64)  # sign -> half
    expected_half = msg[pos] ^ pad
    matches = int((observed_half == expected_half).sum())
    raw_acc = matches / n

    # Recover payload by majority vote of each entry's implied message bit.
    implied = observed_half ^ pad  # estimate of m[pos]
    votes = np.zeros((_MSG_BITS, 2), dtype=np.int64)
    np.add.at(votes, (pos, implied), 1)
    msg_hat = votes.argmax(axis=1)
    msg_acc = float((msg_hat == msg).mean())

    # One-sided significance under Binomial(n, 0.5) via normal approximation.
    z_score = (matches - 0.5 * n) / (0.5 * sqrt(n))
    pvalue = 0.5 * erfc(z_score / sqrt(2))

    return {
        "raw_acc": raw_acc,
        "msg_acc": msg_acc,
        "matches": matches,
        "n": n,
        "z_score": z_score,
        "pvalue": pvalue,
        "present": pvalue < 1e-6,
        "msg_hat": msg_hat,
        "msg": msg,
    }
