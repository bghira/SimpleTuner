from __future__ import annotations

import importlib
import logging
from importlib import metadata
from inspect import signature
from typing import Any

import torch

_PATCH_MARKER = "_simpletuner_fd6d7e0_checkpoint_patch"


def _sdnq_version() -> str | None:
    try:
        return metadata.version("sdnq")
    except metadata.PackageNotFoundError:
        return None


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split("."):
        number = ""
        for char in part:
            if not char.isdigit():
                break
            number += char
        if not number:
            break
        parts.append(int(number))
    return tuple(parts)


def _has_upstream_checkpoint_fix() -> bool:
    module = importlib.import_module("sdnq.training.layers.linear.linear_int8.linear_int8_ckpt")
    return "input_shape" in signature(module.int8_matmul_backward_ckpt).parameters


def _output_shape(grad_output: torch.Tensor, input: torch.Tensor | None, input_shape: torch.Size | None) -> list[int]:
    output_shape = list(grad_output.shape)
    output_shape[-1] = input_shape[-1] if input_shape is not None else input.shape[-1]
    return output_shape


def _patch_int8_static(module: Any) -> None:
    def int8_matmul_backward_ckpt(
        grad_output,
        input,
        weight,
        input_scale,
        scale,
        bias=None,
        svd_up=None,
        svd_down=None,
        zero_point=None,
        hadamard=None,
        input_shape=None,
        do_grad_input=True,
        do_grad_weight=True,
        do_grad_bias=True,
    ):
        grad_input = grad_weight = grad_bias = None
        output_shape = _output_shape(grad_output, input, input_shape)
        grad_output = grad_output.flatten(0, -2)
        if do_grad_input:
            dequantized_weight = (
                module.dequantize_symmetric_compiled(weight, scale)
                if zero_point is None
                else module.dequantize_asymmetric_compiled(weight, scale, zero_point)
            )
            grad_input = module.int8_matmul_dynamic(
                grad_output,
                dequantized_weight,
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
                output_shape=output_shape,
                do_input_reshape=False,
            )
        if do_grad_weight:
            grad_weight = module.int8_matmul(
                grad_output.t(),
                input,
                input_scale,
                hadamard=hadamard,
                output_shape=None,
                do_input_reshape=False,
                do_transpose=False,
            )
        if do_grad_bias and bias is not None:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias

    class INT8MatmulBackwardCKPT(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            if weight.sdnq_dequantizer.use_hadamard:
                hadamard = module.get_hadamard(
                    weight.sdnq_dequantizer.hadamard_group_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                hadamard = None

            result = module.int8_matmul(
                input,
                weight.weight,
                weight.scale,
                bias=bias,
                svd_up=weight.svd_up,
                svd_down=weight.svd_down,
                zero_point=weight.zero_point,
                hadamard=hadamard,
                do_transpose=True,
            )
            if ctx.needs_input_grad[1]:
                new_input, input_scale = module.get_int8_matmul_backward_inputs(input, hadamard)
            else:
                new_input = input_scale = None
            ctx.save_for_backward(new_input, weight, input_scale, bias)
            ctx.input_shape = input.shape
            return result

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, input_scale, bias = ctx.saved_tensors
            if weight.sdnq_dequantizer.use_hadamard:
                hadamard = module.get_hadamard(
                    weight.sdnq_dequantizer.hadamard_group_size,
                    dtype=grad_output.dtype,
                    device=grad_output.device,
                )
            else:
                hadamard = None
            return module.int8_matmul_backward_ckpt(
                grad_output,
                input,
                weight.weight,
                input_scale,
                weight.scale,
                bias=bias,
                svd_up=weight.svd_up,
                svd_down=weight.svd_down,
                zero_point=weight.zero_point,
                hadamard=hadamard,
                input_shape=ctx.input_shape,
                do_grad_input=ctx.needs_input_grad[0],
                do_grad_weight=ctx.needs_input_grad[1],
                do_grad_bias=ctx.needs_input_grad[2],
            )

    module.int8_matmul_backward_ckpt = int8_matmul_backward_ckpt
    module.INT8MatmulBackwardCKPT = INT8MatmulBackwardCKPT
    module.int8_matmul_with_backward_ckpt = INT8MatmulBackwardCKPT.apply
    module.int8_matmul_backward_ckpt.__dict__[_PATCH_MARKER] = True


def _patch_int8_dynamic(module: Any) -> None:
    def get_int8_matmul_dynamic_backward_inputs(input, weight, hadamard, do_grad_weight=True):
        weight, scale = module.quantize_int_mm(weight.to(dtype=torch.float32), dim=0)
        if do_grad_weight:
            input, input_scale = module.quantize_int_mm(
                input.flatten(0, -2).to(dtype=torch.float32),
                dim=0,
                hadamard=hadamard,
            )
            return input, weight, input_scale, scale
        return None, weight, None, scale

    def int8_matmul_dynamic_backward_ckpt(
        grad_output,
        input,
        weight,
        input_scale,
        weight_scale,
        bias=None,
        svd_up=None,
        svd_down=None,
        hadamard=None,
        input_shape=None,
        do_grad_input=True,
        do_grad_weight=True,
        do_grad_bias=True,
    ):
        grad_input = grad_weight = grad_bias = None
        output_shape = _output_shape(grad_output, input, input_shape)
        grad_output = grad_output.flatten(0, -2)
        if do_grad_input:
            grad_input = module.int8_matmul(
                grad_output,
                weight,
                weight_scale,
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
                output_shape=output_shape,
                do_input_reshape=False,
                do_transpose=False,
            )
        if do_grad_weight:
            grad_weight = module.int8_matmul(
                grad_output.t(),
                input,
                input_scale,
                hadamard=hadamard,
                output_shape=None,
                do_input_reshape=False,
                do_transpose=False,
            )
        if do_grad_bias and bias is not None:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias

    class INT8MatmulDynamicBackwardCKPT(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            if isinstance(weight, module.SDNQTensor):
                svd_up, svd_down = weight.svd_up, weight.svd_down
                ctx.use_hadamard = weight.sdnq_dequantizer.use_hadamard
                ctx.hadamard_group_size = weight.sdnq_dequantizer.hadamard_group_size
                weight = weight.dequantize(non_svd=True, non_hadamard=True)
            else:
                svd_up, svd_down = None, None
                ctx.use_hadamard = False
                ctx.hadamard_group_size = 256
            hadamard = (
                module.get_hadamard(ctx.hadamard_group_size, dtype=input.dtype, device=input.device)
                if ctx.use_hadamard
                else None
            )
            result = module.int8_matmul_dynamic(
                input,
                weight,
                bias=bias,
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
            )
            new_input, new_weight, input_scale, weight_scale = module.get_int8_matmul_dynamic_backward_inputs(
                input,
                weight,
                hadamard,
                do_grad_weight=ctx.needs_input_grad[1],
            )
            ctx.save_for_backward(new_input, new_weight, input_scale, weight_scale, bias, svd_up, svd_down)
            ctx.input_shape = input.shape
            return result

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, input_scale, weight_scale, bias, svd_up, svd_down = ctx.saved_tensors
            hadamard = (
                module.get_hadamard(ctx.hadamard_group_size, dtype=grad_output.dtype, device=grad_output.device)
                if ctx.use_hadamard
                else None
            )
            return module.int8_matmul_dynamic_backward_ckpt(
                grad_output,
                input,
                weight,
                input_scale,
                weight_scale,
                bias=bias,
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
                input_shape=ctx.input_shape,
                do_grad_input=ctx.needs_input_grad[0],
                do_grad_weight=ctx.needs_input_grad[1],
                do_grad_bias=ctx.needs_input_grad[2],
            )

    module.get_int8_matmul_dynamic_backward_inputs = module.compile_func(get_int8_matmul_dynamic_backward_inputs)
    module.int8_matmul_dynamic_backward_ckpt = int8_matmul_dynamic_backward_ckpt
    module.INT8MatmulDynamicBackwardCKPT = INT8MatmulDynamicBackwardCKPT
    module.int8_matmul_dynamic_with_backward_ckpt = INT8MatmulDynamicBackwardCKPT.apply
    module.int8_matmul_dynamic_backward_ckpt.__dict__[_PATCH_MARKER] = True


def _patch_fp_static(module: Any, *, dtype_name: str, matmul_dtype: str) -> None:
    matmul = getattr(module, f"{dtype_name}_matmul")
    matmul_dynamic = getattr(module, f"{dtype_name}_matmul_dynamic")

    def backward_ckpt(
        grad_output,
        input,
        weight,
        input_scale,
        scale,
        bias=None,
        svd_up=None,
        svd_down=None,
        hadamard=None,
        input_shape=None,
        do_grad_input=True,
        do_grad_weight=True,
        do_grad_bias=True,
    ):
        grad_input = grad_weight = grad_bias = None
        output_shape = _output_shape(grad_output, input, input_shape)
        grad_output = grad_output.flatten(0, -2)
        if do_grad_input:
            grad_input = matmul_dynamic(
                grad_output,
                module.dequantize_symmetric_compiled(weight, scale),
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
                output_shape=output_shape,
                do_input_reshape=False,
            )
        if do_grad_weight:
            grad_weight = matmul(
                grad_output.t(),
                input,
                input_scale,
                hadamard=hadamard,
                output_shape=None,
                do_input_reshape=False,
                do_transpose=False,
            )
        if do_grad_bias and bias is not None:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias

    class FPMatmulBackwardCKPT(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            hadamard = (
                module.get_hadamard(
                    weight.sdnq_dequantizer.hadamard_group_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                if weight.sdnq_dequantizer.use_hadamard
                else None
            )
            result = matmul(
                input,
                weight.weight,
                weight.scale,
                bias=bias,
                svd_up=weight.svd_up,
                svd_down=weight.svd_down,
                hadamard=hadamard,
                do_transpose=True,
            )
            if ctx.needs_input_grad[1]:
                new_input, input_scale = module.quantize_fp_mm(
                    input.flatten(0, -2).to(dtype=torch.float32),
                    dim=0,
                    hadamard=hadamard,
                    matmul_dtype=matmul_dtype,
                )
            else:
                new_input = input_scale = None
            ctx.save_for_backward(new_input, weight, input_scale, bias)
            ctx.input_shape = input.shape
            return result

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, input_scale, bias = ctx.saved_tensors
            hadamard = (
                module.get_hadamard(
                    weight.sdnq_dequantizer.hadamard_group_size,
                    dtype=grad_output.dtype,
                    device=grad_output.device,
                )
                if weight.sdnq_dequantizer.use_hadamard
                else None
            )
            return getattr(module, f"{dtype_name}_matmul_backward_ckpt")(
                grad_output,
                input,
                weight.weight,
                input_scale,
                weight.scale,
                bias=bias,
                svd_up=weight.svd_up,
                svd_down=weight.svd_down,
                hadamard=hadamard,
                input_shape=ctx.input_shape,
                do_grad_input=ctx.needs_input_grad[0],
                do_grad_weight=ctx.needs_input_grad[1],
                do_grad_bias=ctx.needs_input_grad[2],
            )

    setattr(module, f"{dtype_name}_matmul_backward_ckpt", backward_ckpt)
    setattr(module, f"{dtype_name.upper()}MatmulBackwardCKPT", FPMatmulBackwardCKPT)
    setattr(module, f"{dtype_name}_matmul_with_backward_ckpt", FPMatmulBackwardCKPT.apply)
    getattr(module, f"{dtype_name}_matmul_backward_ckpt").__dict__[_PATCH_MARKER] = True


def _patch_fp_dynamic(module: Any, *, dtype_name: str, matmul_dtype: str) -> None:
    matmul = getattr(module, f"{dtype_name}_matmul")
    matmul_dynamic = getattr(module, f"{dtype_name}_matmul_dynamic")

    def dynamic_backward_inputs(input, weight, hadamard, do_grad_weight=True):
        new_weight, weight_scale = module.quantize_fp_mm(
            weight.to(dtype=torch.float32),
            dim=0,
            matmul_dtype=matmul_dtype,
        )
        if do_grad_weight:
            new_input, input_scale = module.quantize_fp_mm(
                input.flatten(0, -2).to(dtype=torch.float32),
                dim=0,
                hadamard=hadamard,
                matmul_dtype=matmul_dtype,
            )
            return new_input, new_weight, input_scale, weight_scale
        return None, new_weight, None, weight_scale

    def dynamic_backward_ckpt(
        grad_output,
        input,
        weight,
        input_scale,
        weight_scale,
        bias=None,
        svd_up=None,
        svd_down=None,
        hadamard=None,
        input_shape=None,
        do_grad_input=True,
        do_grad_weight=True,
        do_grad_bias=True,
    ):
        grad_input = grad_weight = grad_bias = None
        output_shape = _output_shape(grad_output, input, input_shape)
        grad_output = grad_output.flatten(0, -2)
        if do_grad_input:
            grad_input = matmul(
                grad_output,
                weight,
                weight_scale,
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
                output_shape=output_shape,
                do_input_reshape=False,
                do_transpose=False,
            )
        if do_grad_weight:
            grad_weight = matmul(
                grad_output.t(),
                input,
                input_scale,
                hadamard=hadamard,
                output_shape=None,
                do_input_reshape=False,
                do_transpose=False,
            )
        if do_grad_bias and bias is not None:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias

    class FPMatmulDynamicBackwardCKPT(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            if isinstance(weight, module.SDNQTensor):
                svd_up, svd_down = weight.svd_up, weight.svd_down
                ctx.use_hadamard = weight.sdnq_dequantizer.use_hadamard
                ctx.hadamard_group_size = weight.sdnq_dequantizer.hadamard_group_size
                weight = weight.dequantize(non_svd=True, non_hadamard=True)
            else:
                svd_up, svd_down = None, None
                ctx.use_hadamard = False
                ctx.hadamard_group_size = 256
            hadamard = (
                module.get_hadamard(ctx.hadamard_group_size, dtype=input.dtype, device=input.device)
                if ctx.use_hadamard
                else None
            )
            result = matmul_dynamic(
                input,
                weight,
                bias=bias,
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
            )
            new_input, new_weight, input_scale, weight_scale = getattr(
                module,
                f"get_{dtype_name}_matmul_dynamic_backward_inputs",
            )(
                input,
                weight,
                hadamard,
                do_grad_weight=ctx.needs_input_grad[1],
            )
            ctx.save_for_backward(new_input, new_weight, input_scale, weight_scale, bias, svd_up, svd_down)
            ctx.input_shape = input.shape
            return result

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, input_scale, weight_scale, bias, svd_up, svd_down = ctx.saved_tensors
            hadamard = (
                module.get_hadamard(ctx.hadamard_group_size, dtype=grad_output.dtype, device=grad_output.device)
                if ctx.use_hadamard
                else None
            )
            return getattr(module, f"{dtype_name}_matmul_dynamic_backward_ckpt")(
                grad_output,
                input,
                weight,
                input_scale,
                weight_scale,
                bias=bias,
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
                input_shape=ctx.input_shape,
                do_grad_input=ctx.needs_input_grad[0],
                do_grad_weight=ctx.needs_input_grad[1],
                do_grad_bias=ctx.needs_input_grad[2],
            )

    setattr(module, f"get_{dtype_name}_matmul_dynamic_backward_inputs", module.compile_func(dynamic_backward_inputs))
    setattr(module, f"{dtype_name}_matmul_dynamic_backward_ckpt", dynamic_backward_ckpt)
    setattr(module, f"{dtype_name.upper()}MatmulDynamicBackwardCKPT", FPMatmulDynamicBackwardCKPT)
    setattr(module, f"{dtype_name}_matmul_dynamic_with_backward_ckpt", FPMatmulDynamicBackwardCKPT.apply)
    getattr(module, f"{dtype_name}_matmul_dynamic_backward_ckpt").__dict__[_PATCH_MARKER] = True


def _patch_uint8_static(module: Any) -> None:
    def uint8_matmul_backward_ckpt(
        grad_output,
        input,
        weight,
        input_scale,
        scale,
        input_zero_point,
        zero_point,
        bias=None,
        svd_up=None,
        svd_down=None,
        hadamard=None,
        input_shape=None,
        do_grad_input=True,
        do_grad_weight=True,
        do_grad_bias=True,
    ):
        grad_input = grad_weight = grad_bias = None
        output_shape = _output_shape(grad_output, input, input_shape)
        grad_output = grad_output.flatten(0, -2)
        if do_grad_input:
            dequantized_weight = (
                module.dequantize_symmetric_compiled(weight, scale)
                if zero_point is None
                else module.dequantize_asymmetric_compiled(weight, scale, zero_point)
            )
            grad_input = module.uint8_matmul_dynamic(
                grad_output,
                dequantized_weight,
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
                output_shape=output_shape,
                do_input_reshape=False,
            )
        if do_grad_weight:
            grad_weight = module.uint8_matmul(
                grad_output.t(),
                input,
                input_scale,
                input_zero_point,
                hadamard=hadamard,
                output_shape=None,
                do_input_reshape=False,
                do_transpose=False,
            )
        if do_grad_bias and bias is not None:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias

    class UINT8MatmulBackwardCKPT(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            hadamard = (
                module.get_hadamard(
                    weight.sdnq_dequantizer.hadamard_group_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                if weight.sdnq_dequantizer.use_hadamard
                else None
            )
            result = module.uint8_matmul(
                input,
                weight.weight,
                weight.scale,
                weight.zero_point,
                bias=bias,
                svd_up=weight.svd_up,
                svd_down=weight.svd_down,
                hadamard=hadamard,
                do_transpose=True,
            )
            if ctx.needs_input_grad[1]:
                new_input, input_scale, input_zero_point = module.get_uint8_matmul_backward_inputs(input, hadamard)
            else:
                new_input = input_scale = input_zero_point = None
            ctx.save_for_backward(new_input, weight, input_scale, input_zero_point, bias)
            ctx.input_shape = input.shape
            return result

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, input_scale, input_zero_point, bias = ctx.saved_tensors
            hadamard = (
                module.get_hadamard(
                    weight.sdnq_dequantizer.hadamard_group_size,
                    dtype=grad_output.dtype,
                    device=grad_output.device,
                )
                if weight.sdnq_dequantizer.use_hadamard
                else None
            )
            return module.uint8_matmul_backward_ckpt(
                grad_output,
                input,
                weight.weight,
                input_scale,
                weight.scale,
                input_zero_point,
                weight.zero_point,
                bias=bias,
                svd_up=weight.svd_up,
                svd_down=weight.svd_down,
                hadamard=hadamard,
                input_shape=ctx.input_shape,
                do_grad_input=ctx.needs_input_grad[0],
                do_grad_weight=ctx.needs_input_grad[1],
                do_grad_bias=ctx.needs_input_grad[2],
            )

    module.uint8_matmul_backward_ckpt = uint8_matmul_backward_ckpt
    module.UINT8MatmulBackwardCKPT = UINT8MatmulBackwardCKPT
    module.uint8_matmul_with_backward_ckpt = UINT8MatmulBackwardCKPT.apply
    module.uint8_matmul_backward_ckpt.__dict__[_PATCH_MARKER] = True


def _patch_uint8_dynamic(module: Any) -> None:
    def get_uint8_matmul_dynamic_backward_inputs(input, weight, hadamard, do_grad_weight=True):
        weight, scale, zero_point = module.quantize_uint_mm(weight.to(dtype=torch.float32), dim=0)
        if do_grad_weight:
            input, input_scale, input_zero_point = module.quantize_uint_mm(
                input.flatten(0, -2).to(dtype=torch.float32),
                dim=0,
                hadamard=hadamard,
            )
            return input, weight, input_scale, scale, input_zero_point, zero_point
        return None, weight, None, scale, None, zero_point

    def uint8_matmul_dynamic_backward_ckpt(
        grad_output,
        input,
        weight,
        input_scale,
        weight_scale,
        input_zero_point,
        weight_zero_point,
        bias=None,
        svd_up=None,
        svd_down=None,
        hadamard=None,
        input_shape=None,
        do_grad_input=True,
        do_grad_weight=True,
        do_grad_bias=True,
    ):
        grad_input = grad_weight = grad_bias = None
        output_shape = _output_shape(grad_output, input, input_shape)
        grad_output = grad_output.flatten(0, -2)
        if do_grad_input:
            grad_input = module.uint8_matmul(
                grad_output,
                weight,
                weight_scale,
                weight_zero_point,
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
                output_shape=output_shape,
                do_input_reshape=False,
                do_transpose=False,
            )
        if do_grad_weight:
            grad_weight = module.uint8_matmul(
                grad_output.t(),
                input,
                input_scale,
                input_zero_point,
                hadamard=hadamard,
                output_shape=None,
                do_input_reshape=False,
                do_transpose=False,
            )
        if do_grad_bias and bias is not None:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias

    class UINT8MatmulDynamicBackwardCKPT(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            if isinstance(weight, module.SDNQTensor):
                svd_up, svd_down = weight.svd_up, weight.svd_down
                ctx.use_hadamard = weight.sdnq_dequantizer.use_hadamard
                ctx.hadamard_group_size = weight.sdnq_dequantizer.hadamard_group_size
                weight = weight.dequantize(non_svd=True, non_hadamard=True)
            else:
                svd_up, svd_down = None, None
                ctx.use_hadamard = False
                ctx.hadamard_group_size = 256
            hadamard = (
                module.get_hadamard(ctx.hadamard_group_size, dtype=input.dtype, device=input.device)
                if ctx.use_hadamard
                else None
            )
            result = module.uint8_matmul_dynamic(
                input,
                weight,
                bias=bias,
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
            )
            new_input, new_weight, input_scale, weight_scale, input_zero_point, weight_zero_point = (
                module.get_uint8_matmul_dynamic_backward_inputs(
                    input,
                    weight,
                    hadamard,
                    do_grad_weight=ctx.needs_input_grad[1],
                )
            )
            ctx.save_for_backward(
                new_input,
                new_weight,
                input_scale,
                weight_scale,
                input_zero_point,
                weight_zero_point,
                bias,
                svd_up,
                svd_down,
            )
            ctx.input_shape = input.shape
            return result

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, input_scale, weight_scale, input_zero_point, weight_zero_point, bias, svd_up, svd_down = (
                ctx.saved_tensors
            )
            hadamard = (
                module.get_hadamard(ctx.hadamard_group_size, dtype=grad_output.dtype, device=grad_output.device)
                if ctx.use_hadamard
                else None
            )
            return module.uint8_matmul_dynamic_backward_ckpt(
                grad_output,
                input,
                weight,
                input_scale,
                weight_scale,
                input_zero_point,
                weight_zero_point,
                bias=bias,
                svd_up=svd_up,
                svd_down=svd_down,
                hadamard=hadamard,
                input_shape=ctx.input_shape,
                do_grad_input=ctx.needs_input_grad[0],
                do_grad_weight=ctx.needs_input_grad[1],
                do_grad_bias=ctx.needs_input_grad[2],
            )

    module.get_uint8_matmul_dynamic_backward_inputs = module.compile_func(get_uint8_matmul_dynamic_backward_inputs)
    module.uint8_matmul_dynamic_backward_ckpt = uint8_matmul_dynamic_backward_ckpt
    module.UINT8MatmulDynamicBackwardCKPT = UINT8MatmulDynamicBackwardCKPT
    module.uint8_matmul_dynamic_with_backward_ckpt = UINT8MatmulDynamicBackwardCKPT.apply
    module.uint8_matmul_dynamic_backward_ckpt.__dict__[_PATCH_MARKER] = True


def apply_sdnq_checkpointed_backward_fix(logger: logging.Logger | None = None) -> bool:
    """Apply Disty0/sdnq fd6d7e0 when the installed SDNQ wheel predates it."""

    version = _sdnq_version()
    if version is None or _version_tuple(version) < (0, 2, 2):
        return False

    try:
        int8_static = importlib.import_module("sdnq.training.layers.linear.linear_int8.linear_int8_ckpt")
        if getattr(int8_static.int8_matmul_backward_ckpt, _PATCH_MARKER, False) or _has_upstream_checkpoint_fix():
            return False

        _patch_int8_static(int8_static)
        _patch_int8_dynamic(importlib.import_module("sdnq.training.layers.linear.linear_int8.linear_int8_dynamic_ckpt"))
        _patch_uint8_static(importlib.import_module("sdnq.training.layers.linear.linear_uint8.linear_uint8_ckpt"))
        _patch_uint8_dynamic(importlib.import_module("sdnq.training.layers.linear.linear_uint8.linear_uint8_dynamic_ckpt"))
        _patch_fp_static(
            importlib.import_module("sdnq.training.layers.linear.linear_fp8.linear_fp8_ckpt"),
            dtype_name="fp8",
            matmul_dtype="float8_e4m3fn",
        )
        _patch_fp_dynamic(
            importlib.import_module("sdnq.training.layers.linear.linear_fp8.linear_fp8_dynamic_ckpt"),
            dtype_name="fp8",
            matmul_dtype="float8_e4m3fn",
        )
        _patch_fp_static(
            importlib.import_module("sdnq.training.layers.linear.linear_fp16.linear_fp16_ckpt"),
            dtype_name="fp16",
            matmul_dtype="float16",
        )
        _patch_fp_dynamic(
            importlib.import_module("sdnq.training.layers.linear.linear_fp16.linear_fp16_dynamic_ckpt"),
            dtype_name="fp16",
            matmul_dtype="float16",
        )
    except ModuleNotFoundError:
        return False

    if logger is not None:
        logger.info("Applied SDNQ checkpointed backward compatibility patch for frozen quantized weights.")
    return True
