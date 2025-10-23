from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def wan_forward_origin(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    student=True,
    output_features=False,
    output_features_stride=2,
    final_layer=False,
    unpachify_layer=False,
    midfeat_layer=False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if student:
        self.disable_adapters()

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    if output_features:
        features_list = []

    # 4. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )
    else:
        for _, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

            if output_features and _ % output_features_stride == 0:
                features_list.append(hidden_states)

    # 5. Output norm, projection & unpatchify
    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if output_features:
        if final_layer:
            ori_features_list = torch.stack(features_list, dim=0)
            new_feat_list = []
            for xfeat in features_list:
                tmp = (self.norm_out(xfeat.float()) * (1 + scale) + shift).type_as(xfeat)
                tmp = self.proj_out(tmp)
                tmp = tmp.reshape(
                    batch_size,
                    post_patch_num_frames,
                    post_patch_height,
                    post_patch_width,
                    p_t,
                    p_h,
                    p_w,
                    -1,
                )
                tmp = tmp.permute(0, 7, 1, 4, 2, 5, 3, 6)
                tmp = tmp.flatten(6, 7).flatten(4, 5).flatten(2, 3)
                new_feat_list.append(tmp)
            features_list = torch.stack(new_feat_list, dim=0)
        else:
            ori_features_list = torch.stack(features_list, dim=0)
            features_list = torch.stack(features_list, dim=0)
    else:
        features_list = None
        ori_features_list = None

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output, features_list, ori_features_list)

    return Transformer2DModelOutput(sample=output)


def wan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    student=True,
    output_features=False,
    output_features_stride=2,
    final_layer=False,
    unpachify_layer=False,
    midfeat_layer=False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

    if student and int(timestep[0]) <= 981:
        self.set_adapter("lora1")
        self.enable_adapters()
    else:
        return self.wan_forward_origin(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            student=student,
            output_features=output_features,
            output_features_stride=output_features_stride,
            final_layer=final_layer,
            unpachify_layer=unpachify_layer,
            midfeat_layer=midfeat_layer,
        )

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder_lora(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    if output_features:
        features_list = []

    # 4. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )
    else:
        for _, block in enumerate(self.blocks):  # 30
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

            if output_features and _ % output_features_stride == 0:
                features_list.append(hidden_states)

    # 5. Output norm, projection & unpatchify
    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out_lora(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out_lora(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if output_features:
        if final_layer:
            ori_features_list = torch.stack(features_list, dim=0)
            new_feat_list = []
            for xfeat in features_list:
                tmp = (self.norm_out_lora(xfeat.float()) * (1 + scale) + shift).type_as(xfeat)
                tmp = self.proj_out_lora(tmp)

                tmp = tmp.reshape(
                    batch_size,
                    post_patch_num_frames,
                    post_patch_height,
                    post_patch_width,
                    p_t,
                    p_h,
                    p_w,
                    -1,
                )
                tmp = tmp.permute(0, 7, 1, 4, 2, 5, 3, 6)
                tmp = tmp.flatten(6, 7).flatten(4, 5).flatten(2, 3)
                new_feat_list.append(tmp)
            features_list = torch.stack(new_feat_list, dim=0)
        else:
            ori_features_list = torch.stack(features_list, dim=0)
            features_list = torch.stack(features_list, dim=0)
    else:
        features_list = None
        ori_features_list = None

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output, features_list, ori_features_list)

    return Transformer2DModelOutput(sample=output)


class DiscriminatorHead(nn.Module):
    """
    A single discriminator head that processes feature maps through convolutional layers.

    This module applies a series of 1x1 convolutions with normalization and activation
    to extract discriminative features from the input.

    Args:
        input_channel (int): Number of input channels
        output_channel (int): Number of output channels (default: 1)
    """

    def __init__(self, input_channel, output_channel=1):
        super().__init__()
        inner_channel = 1024

        # First convolutional block: projection + normalization + activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, inner_channel, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, inner_channel),
            nn.LeakyReLU(inplace=True),  # Note: Using LeakyReLU instead of GELU to save memory
        )

        # Second convolutional block: same structure for deeper feature extraction
        self.conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, inner_channel),
            nn.LeakyReLU(inplace=True),  # Note: Using LeakyReLU instead of GELU to save memory
        )

        # Output projection layer
        self.conv_out = nn.Conv2d(inner_channel, output_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass of the discriminator head.

        Args:
            x (torch.Tensor): Input tensor of shape:
                - 5D: (batch, channels, time, height, width) for video/3D data
                - 3D: (batch, time*height*width, channels) for flattened data

        Returns:
            torch.Tensor: Discriminator output features
        """

        # Handle 5D input (video or 3D data)
        if x.dim() == 5:
            b, ch, t, h, w = x.shape

            # Reorder dimensions: (b, ch, t, h, w) -> (b, t, ch, h, w)
            x = x.permute(0, 2, 1, 3, 4)

            # Reshape to merge batch and time dimensions
            # Note: Hard-coded values (1*3, 16, 98, 160) suggest specific expected dimensions
            x = x.reshape(1 * 3, 16, 98, 160)

            # Apply pixel unshuffle to reduce spatial dimensions while increasing channels
            pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)  # 2x2 blocks
            x = pixel_unshuffle(x)  # Output shape: (3, 16*4, 49, 80)

            # Add batch dimension back and rearrange for processing
            x = x.unsqueeze(0)  # Shape: (1, 3, 64, 49, 80)
            x = x.permute(0, 1, 3, 4, 2)  # Shape: (1, 3, 49, 80, 64)

            # Flatten spatial dimensions
            x = x.reshape(b, -1, ch * 4)

        # Process 3D input tensor
        # Expected shape: (batch, time*width*height, channels)
        b, twh, c = x.shape

        # Calculate time dimension assuming fixed spatial dimensions (30x52)
        t = twh // (30 * 52)

        # Reshape to separate spatial dimensions
        x = x.view(-1, 30 * 52, c)

        # Permute to channel-first format for convolutions
        x = x.permute(0, 2, 1)

        # Reshape to 4D tensor for 2D convolutions
        x = x.view(b * t, c, 30, 52)

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x) + x  # Residual connection
        x = self.conv_out(x)

        return x


class Discriminator(nn.Module):
    """
    Multi-head discriminator module that processes features from multiple layers.

    This discriminator applies multiple discriminator heads to features extracted
    from different layers of a model (e.g., from a transformer or U-Net).

    Args:
        stride (int): Sampling stride for selecting which layers to use (default: 8)
        num_h_per_head (int): Number of discriminator heads per feature channel (default: 1)
        adapter_channel_dims (list): List of channel dimensions for each adapter (default: [1536])
        total_layers (int): Total number of layers in the source model (default: 48)
    """

    def __init__(
        self,
        stride=8,
        num_h_per_head=1,
        adapter_channel_dims=[1536],
        total_layers=48,
    ):
        super().__init__()

        # Repeat adapter channels based on how many layers we're sampling
        adapter_channel_dims = adapter_channel_dims * (total_layers // stride)

        self.stride = stride
        self.num_h_per_head = num_h_per_head
        self.head_num = len(adapter_channel_dims)

        # Create nested ModuleList:
        # - Outer list: one entry per adapter channel dimension
        # - Inner list: num_h_per_head discriminator heads per channel
        self.heads = nn.ModuleList(
            [
                nn.ModuleList([DiscriminatorHead(adapter_channel) for _ in range(self.num_h_per_head)])
                for adapter_channel in adapter_channel_dims
            ]
        )

    def forward(self, features):
        """
        Forward pass applying all discriminator heads to their corresponding features.

        Args:
            features (list): List of feature tensors from different layers.
                            Length must match the number of head groups (self.head_num).

        Returns:
            list: Flattened list of discriminator outputs from all heads
        """
        outputs = []

        # Note: The create_custom_forward function is defined but not used
        # This might be for gradient checkpointing in the original implementation
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        # Ensure we have the expected number of features
        assert len(features) == len(self.heads), f"Expected {len(self.heads)} features, got {len(features)}"

        # Apply each head group to its corresponding feature
        for i in range(len(features)):
            # Apply all heads in this group to the same feature
            for h in self.heads[i]:
                out = h(features[i])
                outputs.append(out)

        return outputs
