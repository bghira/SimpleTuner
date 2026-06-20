from typing import Optional

import torch
import torchao
from torch import Tensor
from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight


class _Int8WeightOnlyLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Int8QuantizedTrainingLinearWeight,
        bias: Optional[Tensor] = None,
    ):
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None

        # NOTE: we have to .T before .to(input.dtype) for torch.compile() mixed matmul to work
        out = (input @ weight.int_data.T.to(input.dtype)) * weight.scale
        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_input = (grad_output * weight.scale) @ weight.int_data.to(grad_output.dtype)
        # print(f"dtypes: grad_output {grad_output.dtype}, input {input.dtype}, weight {weight.dtype}")
        # here is the patch: we will cast the input to the grad_output dtype.
        grad_weight = grad_output.reshape(-1, weight.shape[0]).T @ input.to(grad_output.dtype).reshape(-1, weight.shape[1])
        grad_bias = grad_output.reshape(-1, weight.shape[0]).sum(0) if ctx.bias else None
        return grad_input, grad_weight, grad_bias


torchao.prototype.quantized_training.int8._Int8WeightOnlyLinear = _Int8WeightOnlyLinear

try:
    import torch
    from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight, implements

    # Check if cat is already implemented
    test_tensor = Int8QuantizedTrainingLinearWeight.from_float(torch.randn(2, 2))
    try:
        torch.cat([test_tensor, test_tensor], dim=0)
        print("aten.cat already implemented for Int8QuantizedTrainingLinearWeight")
    except NotImplementedError:
        # Need to monkeypatch
        @implements(torch.ops.aten.cat.default)
        def _(func, types, args, kwargs):
            """Implement concatenation for Int8QuantizedTrainingLinearWeight (needed for DDP)."""
            tensors = args[0]
            dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

            # First, check if we have any Int8QuantizedTrainingLinearWeight tensors
            has_int8 = any(isinstance(t, Int8QuantizedTrainingLinearWeight) for t in tensors)
            if not has_int8:
                # No int8 tensors, use regular cat
                return func(tensors, dim, **kwargs)

            # Check if all tensors have the same number of dimensions
            # This is important for DDP which might mix different tensor types
            ndims = set()
            for t in tensors:
                if isinstance(t, Int8QuantizedTrainingLinearWeight):
                    # Int8 weights are always 2D
                    ndims.add(2)
                else:
                    ndims.add(t.ndim)

            # If we have mixed dimensions, we need to handle this carefully
            if len(ndims) > 1:
                # For DDP's _broadcast_coalesced, tensors are often flattened
                # So we should flatten everything to 1D for concatenation
                flattened = []
                for t in tensors:
                    if isinstance(t, Int8QuantizedTrainingLinearWeight):
                        # Dequantize and flatten
                        flattened.append(t.dequantize().flatten())
                    else:
                        # Just flatten
                        flattened.append(t.flatten())

                # Cat along dim 0 (since everything is now 1D)
                return func(flattened, 0, **kwargs)

            # If all have same dimensions, proceed with type checking
            all_int8 = all(isinstance(t, Int8QuantizedTrainingLinearWeight) for t in tensors)

            if not all_int8:
                # Mixed types with same dimensions - dequantize int8 tensors
                dequantized = []
                for t in tensors:
                    if isinstance(t, Int8QuantizedTrainingLinearWeight):
                        dequantized.append(t.dequantize())
                    else:
                        dequantized.append(t)
                return func(dequantized, dim, **kwargs)

            # All are int8 weights with same dimensions
            if dim == 0:
                # Row-wise concat: both int_data and scale need concatenation
                int_data_list = [t.int_data for t in tensors]
                scale_list = [t.scale for t in tensors]

                cat_int_data = torch.cat(int_data_list, dim=0)
                cat_scale = torch.cat(scale_list, dim=0)

                return Int8QuantizedTrainingLinearWeight(cat_int_data, cat_scale)

            elif dim == 1:
                # Column-wise concat: scales stay the same (per-row quantization)
                int_data_list = [t.int_data for t in tensors]
                cat_int_data = torch.cat(int_data_list, dim=1)

                # All tensors should have the same scale for this case
                return Int8QuantizedTrainingLinearWeight(cat_int_data, tensors[0].scale)

            else:
                # For other dimensions, fall back to dequantize
                dequantized = [t.dequantize() for t in tensors]
                return func(dequantized, dim, **kwargs)

        print("✓ Monkeypatched aten.cat for Int8QuantizedTrainingLinearWeight - DDP enabled!")

except ImportError:
    # torchao int8 not being used
    pass
except Exception as e:
    print(f"Warning: Failed to monkeypatch int8 cat operation: {e}")
    print("DDP may not work with int8 quantization")


try:
    import peft.tuners.lora.model as peft_lora_model
    import peft.tuners.lora.torchao as peft_lora_torchao
    from peft.import_utils import is_torchao_available
    from peft.tuners.lora.torchao import TorchaoLoraLinear
    from peft.tuners.tuners_utils import BaseTunerLayer
    from torchao.prototype.quantized_training import int8_weight_only_quantized_training
    from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight
    from torchao.quantization import Float8Tensor, IntxUnpackedToInt8Tensor
    from torchao.quantization.linear_activation_quantized_tensor import LinearActivationQuantizedTensor
    from torchao.quantization.quant_api import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8DynamicActivationInt4WeightConfig,
        Float8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8DynamicActivationIntxWeightConfig,
    )
    from torchao.utils import TorchAOBaseTensor

    from simpletuner.helpers.training.state_tracker import StateTracker

    def _simpletuner_torchao_requantize_config(weight):
        args = StateTracker.get_args()
        model_precision = getattr(args, "base_model_precision", None) if args is not None else None
        if isinstance(weight, Int8QuantizedTrainingLinearWeight):
            return int8_weight_only_quantized_training
        if model_precision == "int8dq-torchao":
            return lambda: Int8DynamicActivationInt8WeightConfig(version=2)
        if isinstance(weight, LinearActivationQuantizedTensor):
            return Int8DynamicActivationInt8WeightConfig
        if model_precision == "int8dq-int4-torchao" or isinstance(weight, IntxUnpackedToInt8Tensor):
            return lambda: Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)
        if model_precision == "fp8-torchao" and isinstance(weight, Float8Tensor):
            return Float8DynamicActivationFloat8WeightConfig
        if model_precision == "fp8wo-torchao" and isinstance(weight, Float8Tensor):
            return Float8WeightOnlyConfig
        if model_precision == "fp8-int4-torchao":
            return Float8DynamicActivationInt4WeightConfig
        return None

    def _simpletuner_dispatch_torchao(target: torch.nn.Module, adapter_name: str, config, **kwargs):
        new_module = None
        target_base_layer = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
        if not hasattr(target_base_layer, "weight") or not is_torchao_available():
            return new_module
        get_apply_tensor_subclass = _simpletuner_torchao_requantize_config(target_base_layer.weight)
        if isinstance(target_base_layer.weight, TorchAOBaseTensor) and get_apply_tensor_subclass is not None:
            return TorchaoLoraLinear(
                target,
                adapter_name,
                config=config,
                get_apply_tensor_subclass=get_apply_tensor_subclass,
                **kwargs,
            )
        return peft_lora_torchao._simpletuner_original_dispatch_torchao(
            target,
            adapter_name,
            config=config,
            **kwargs,
        )

    if not hasattr(peft_lora_torchao, "_simpletuner_original_dispatch_torchao"):
        peft_lora_torchao._simpletuner_original_dispatch_torchao = peft_lora_torchao.dispatch_torchao
    peft_lora_torchao.dispatch_torchao = _simpletuner_dispatch_torchao
    peft_lora_model.dispatch_torchao = _simpletuner_dispatch_torchao
except ImportError:
    pass
