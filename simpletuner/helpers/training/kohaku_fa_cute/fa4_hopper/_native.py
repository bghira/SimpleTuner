"""Flat fourteen-tensor Hopper host entries for an explicit native runner.

Buffers stay BHSD/BHS at the boundary. Only metadata is transposed inside the
compiled host program, and every invocation receives fresh tensor descriptors.
"""

from types import SimpleNamespace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from flash_attn.cute.config import Sm90Registers
from flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_attn.cute.utils import AuxData

from ._backward import FlashAttentionBackwardSm90
from ._forward import FlashAttentionForwardSm90
from ._persistent import PersistentFlashAttentionForwardSm90
from ._preprocess import FlashAttentionBackwardPreprocess


def configuration_for(dimension):
    """Return the constructor policy qualified for D64/D128 on full H200."""
    if dimension not in (64, 128):
        raise ValueError("The qualified flat Hopper family supports D64/D128.")
    return dict(
        forward_m=128,
        forward_n=128,
        forward_stages=2,
        query_in_registers=False,
        forward_threads=384,
        backward_m=128 if dimension == 64 else 64,
        backward_n=128,
        query_stages=2,
        dout_stages=2,
        threads=384,
        score_m_warps=1,
        kv_m_warps=2,
        dq_m_warps=2 if dimension == 64 else 1,
        v_in_regs=False,
    )


@cute.jit
def _bshd(tensor):
    return cute.make_tensor(tensor.iterator, cute.select(tensor.layout, mode=[0, 2, 1, 3]))


class FlatForward:
    def __init__(self, dtype, dimension, causal, sequence, config):
        persistent = dimension == 128 and not causal and sequence == 4096
        forward = PersistentFlashAttentionForwardSm90 if persistent else FlashAttentionForwardSm90
        self.forward = forward(
            dtype,
            dimension,
            dimension,
            1,
            is_causal=causal,
            is_local=False,
            pack_gqa=False,
            tile_m=config.forward_m,
            tile_n=config.forward_n,
            num_stages=config.forward_stages,
            num_threads=config.forward_threads,
            Q_in_regs=config.query_in_registers,
            has_aux_tensors=True,
            registers=Sm90Registers(mma=240, producer=24),
        )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        unused_mask: cute.Tensor,
        out: cute.Tensor,
        maximum: cute.Tensor,
        logsum: cute.Tensor,
        unused_partial: cute.Tensor,
        unused_split: cute.Tensor,
        unused_dq: cute.Tensor,
        unused_dk: cute.Tensor,
        unused_dv: cute.Tensor,
        unused_blocks: cute.Tensor,
        unused_base: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.forward(
            _bshd(q),
            _bshd(k),
            _bshd(v),
            _bshd(out),
            logsum,
            scale,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            AuxData([maximum], None),
            None,
            None,
            None,
            stream,
        )


class FlatBackward:
    def __init__(self, dtype, dimension, causal, config):
        self.prepare = FlashAttentionBackwardPreprocess(dtype, dimension, dimension, config.backward_m)
        self.main = FlashAttentionBackwardSm90(
            dtype,
            dimension,
            dimension,
            1,
            is_causal=causal,
            is_local=False,
            deterministic=False,
            tile_m=config.backward_m,
            tile_n=config.backward_n,
            Q_stage=config.query_stages,
            dO_stage=config.dout_stages,
            PdS_stage=2,
            SdP_swapAB=True,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=config.score_m_warps,
            AtomLayoutNdKV=config.kv_m_warps,
            AtomLayoutMdQ=config.dq_m_warps,
            num_threads=config.threads,
            V_in_regs=config.v_in_regs,
            has_aux_tensors=True,
            dQ_single_wg=False,
        )
        self.convert = FlashAttentionBackwardPostprocess(
            dtype,
            dimension,
            90,
            config.backward_m,
            256,
            config.dq_m_warps,
            False,
        )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        unused_mask: cute.Tensor,
        out: cute.Tensor,
        maximum: cute.Tensor,
        logsum: cute.Tensor,
        dout: cute.Tensor,
        delta: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        scratch: cute.Tensor,
        lognorm: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.prepare(
            _bshd(out),
            _bshd(dout),
            delta,
            logsum,
            lognorm,
            scratch,
            None,
            None,
            None,
            None,
            None,
            cutlass.Float32(1.0),
            None,
            None,
            stream,
        )
        self.main(
            _bshd(q),
            _bshd(k),
            _bshd(v),
            _bshd(dout),
            lognorm,
            delta,
            scratch,
            _bshd(dk),
            _bshd(dv),
            scale,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            AuxData([maximum], None),
            None,
            None,
            None,
            stream,
        )
        self.convert(scratch, _bshd(dq), scale, None, None, None, None, stream)


def compile_kernel(mode, tensors, causal, configuration=None):
    """Compile the native ABI without specializing tensor dimensions to constants."""
    from . import available

    if mode not in ("forward", "backward") or not available() or len(tensors) != 14:
        raise ValueError("The flat Hopper launcher requires its pinned runtime and fourteen tensors.")
    q = tensors[0]
    properties = torch.cuda.get_device_properties(q.device)
    if (
        torch.cuda.get_device_capability(q.device) != (9, 0)
        or properties.name not in ("NVIDIA H200", "NVIDIA H200 NVL")
        or properties.multi_processor_count != 132
        or q.ndim != 4
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.shape != tensors[1].shape
        or q.shape != tensors[2].shape
        or not all(tensor.is_contiguous() for tensor in tensors[:3])
        or q.shape[-1] not in (64, 128)
        or q.shape[-2] % 128
    ):
        raise ValueError("This flat Hopper qualification requires full H200 and 128-aligned square D64/D128 inputs.")
    expected = configuration_for(q.shape[-1])
    configuration = dict(expected if configuration is None else configuration)
    maximum_multiple = configuration.pop("maximum_multiple", 1)
    accumulator_head_dim = configuration.pop("accumulator_head_dim", 0)
    persistent_override = configuration.pop("persistent", 0)
    if maximum_multiple not in (1, 128) or accumulator_head_dim != 0 or persistent_override != 0:
        raise ValueError("The flat Hopper allocation or persistence override is not qualified.")
    if configuration != expected:
        raise ValueError("The flat Hopper constructor policy differs from its qualified configuration.")
    config = SimpleNamespace(**configuration)
    dtypes = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}
    arguments = [
        make_fake_tensor(
            dtypes[tensor.dtype],
            tuple(cute.sym_int() for _ in tensor.shape),
            stride=tuple(
                stride if stride <= 1 else cute.sym_int64(divisibility=min(16 // tensor.element_size(), stride & -stride))
                for stride in tensor.stride()
            ),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        for tensor in tensors
    ]
    program = (
        FlatBackward(dtypes[q.dtype], q.shape[-1], causal, config)
        if mode == "backward"
        else FlatForward(dtypes[q.dtype], q.shape[-1], causal, q.shape[-2], config)
    )
    with torch.cuda.device(q.device):
        return cute.compile(
            program,
            *arguments,
            cutlass.Float32(1.0),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
