# Vendored KohakuFA; SimpleTuner namespace changes. See NOTICE.
"""The hardware a kernel configuration is planned for.

Fields come in two groups: queried from the driver (always available) and measured
with a microbenchmark on the card (``benchmarks/device.py``). ``KNOWN`` holds measured
descriptions; an unknown card falls back to the closest architecture's numbers, which
only shifts the planner's estimates, never the legality of a configuration.

``ARCHS`` maps a compute capability to the module that plans and runs it, so another
architecture (sm_120 consumer Blackwell, Hopper) plugs in its own ``plan`` without
touching the sm_100 kernels.
"""

import dataclasses
import json

import torch


@dataclasses.dataclass(frozen=True)
class Device:
    """Everything a planner needs to know about a card.

    Rates are per SM and per clock unless noted: ``mma_macs`` 16-bit multiply-adds of
    the tensor cores, ``ex2`` exponentials (MUFU), ``fma`` fp32 FMA lanes, ``smem_bw``
    shared-memory bytes; ``l2_bw`` / ``dram_bw`` are whole-chip GB/s."""

    name: str
    arch: str  # "sm100", "sm103", ... (the key into ARCHS)
    sms: int
    clock_ghz: float
    smem_per_cta: int
    tmem_columns: int  # 0 when the architecture has no tensor memory
    l2_bytes: int

    mma_macs: float
    ex2: float
    fma: float
    smem_bw: float
    l2_bw: float
    dram_bw: float

    @classmethod
    def query(cls, index: int = 0, **measured) -> "Device":
        """Driver-reported fields; measured ones from ``measured`` or, for a card in
        ``KNOWN``, from its stored description."""
        p = torch.cuda.get_device_properties(index)
        arch = f"sm{p.major}{p.minor}"
        base = dataclasses.asdict(_closest(p.name, arch))
        base.update(
            name=p.name,
            arch=arch,
            sms=p.multi_processor_count,
            smem_per_cta=p.shared_memory_per_block_optin,
            l2_bytes=p.L2_cache_size,
        )
        base.update(measured)
        return cls(**base)

    @classmethod
    def from_json(cls, path: str) -> "Device":
        with open(path) as fh:
            return cls(**json.load(fh))

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(dataclasses.asdict(self), fh, indent=2)

    def cycles(self, seconds: float) -> float:
        return seconds * self.clock_ghz * 1e9


# Measured with benchmarks/device.py (ex2 / fma: dependent-chain microbenchmarks with
# 16 warps per SM; l2_bw: all SMs re-reading an L2-resident buffer).
B300 = Device(
    name="NVIDIA B300", arch="sm103", sms=148, clock_ghz=1.9, smem_per_cta=232448,
    tmem_columns=512, l2_bytes=126 * 2**20,
    mma_macs=4096, ex2=32, fma=128, smem_bw=128, l2_bw=15000, dram_bw=7500,
)  # fmt: skip
# B200: half the exponential rate of B300 (sm_103 doubled the MUFU ex2 throughput)
B200 = dataclasses.replace(B300, name="NVIDIA B200", arch="sm100", ex2=16, l2_bytes=126 * 2**20)

KNOWN = {"B300": B300, "B200": B200}


def _closest(name: str, arch: str) -> Device:
    for key, dev in KNOWN.items():
        if key in name:
            return dev
    return B300 if arch == "sm103" else B200


# Compute capability -> the module with ``attention_forward`` / ``attention_backward``
# and a ``plan`` submodule for it. sm_120 (consumer Blackwell: no tensor memory, no
# tcgen05) needs its own kernels: register its module here.
ARCHS = {"sm100": "simpletuner.helpers.training.kohakufa.sm100", "sm103": "simpletuner.helpers.training.kohakufa.sm100"}
