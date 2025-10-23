import torch.nn as nn
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DiscriminatorHead(nn.Module):
    """
    A discriminator head module that processes feature maps through convolutional layers.

    This module applies spatial downsampling via PixelUnshuffle and processes features
    through two convolutional blocks with residual connections.

    Args:
        input_channel (int): Number of input channels
        output_channel (int): Number of output channels (default: 1)
    """

    def __init__(self, input_channel, output_channel=1):
        super().__init__()
        inner_channel = 1024

        # First convolutional block with GroupNorm and LeakyReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, inner_channel, 1, 1, 0),
            nn.GroupNorm(32, inner_channel),
            nn.LeakyReLU(inplace=True),  # Using LeakyReLU instead of GELU to save memory
        )

        # Second convolutional block with GroupNorm and LeakyReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, 1, 1, 0),
            nn.GroupNorm(32, inner_channel),
            nn.LeakyReLU(inplace=True),  # Using LeakyReLU instead of GELU to save memory
        )

        # Output convolution layer
        self.conv_out = nn.Conv2d(inner_channel, output_channel, 1, 1, 0)

    def forward(self, x):
        """
        Forward pass of the discriminator head.

        Args:
            x (torch.Tensor): Input tensor with shape (b, ch, t, h, w) for 5D inputs
                             or standard 4D tensor shape

        Returns:
            torch.Tensor: Processed output tensor
        """
        if x.dim() == 5:
            # Handle 5D input: (batch, channels, time, height, width)
            b, ch, t, h, w = x.shape

            # Reshape tensor for processing
            # Change from (b, ch, t, h, w) to (b, t, ch, h, w)
            x = x.permute(0, 2, 1, 3, 4)

            # Merge batch and time dimensions: (b*t, ch, h, w)
            x = x.reshape(b * t, ch, h, w)

            # Apply spatial downsampling using PixelUnshuffle
            # This rearranges spatial blocks into channel dimension
            pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
            x = pixel_unshuffle(x)  # Output shape: (b*t, ch*4, h/2, w/2)

            # Add batch dimension back
            x = x.unsqueeze(0)  # Shape: (1, b*t, ch*4, h/2, w/2)

            # Rearrange dimensions for further processing
            x = x.permute(0, 1, 3, 4, 2)  # Shape: (1, b*t, h/2, w/2, ch*4)

            # Reshape to prepare for convolution
            x = x.reshape(b, -1, ch * 4)

            # Process spatial and temporal dimensions
            b, twh, c = x.shape
            t = twh // (49 * 80)  # Calculate time dimension based on spatial size

            # Reshape for convolution processing
            x = x.view(-1, 49 * 80, c)
            x = x.permute(0, 2, 1)
            x = x.view(b * t, c, 49, 80)

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x) + x  # Residual connection
        x = self.conv_out(x)

        return x


class Discriminator(nn.Module):
    """
    Multi-scale discriminator with multiple heads for processing features at different layers.

    This discriminator processes features from multiple layers of a model (e.g., diffusion model)
    using separate discriminator heads for each feature scale.

    Args:
        stride (int): Sampling stride for selecting which layers to process (default: 8)
        num_h_per_head (int): Number of discriminator heads per feature layer (default: 1)
        adapter_channel_dims (list): List of channel dimensions for each adapter layer
        total_layers (int): Total number of layers in the source model (default: 48)
    """

    def __init__(
        self,
        stride=8,
        num_h_per_head=1,
        adapter_channel_dims=[3072],
        total_layers=48,
    ):
        super().__init__()

        # Repeat adapter dimensions based on the number of layers to process
        adapter_channel_dims = adapter_channel_dims * (total_layers // stride)

        self.stride = stride
        self.num_h_per_head = num_h_per_head
        self.head_num = len(adapter_channel_dims)

        # Create discriminator heads for each feature layer
        # Each layer can have multiple heads (num_h_per_head)
        self.heads = nn.ModuleList(
            [
                nn.ModuleList([DiscriminatorHead(adapter_channel) for _ in range(self.num_h_per_head)])
                for adapter_channel in adapter_channel_dims
            ]
        )

    def forward(self, features):
        """
        Forward pass processing features from multiple layers.

        Args:
            features (list): List of feature tensors from different layers

        Returns:
            list: List of discriminator outputs for each head
        """
        outputs = []

        # Helper function for creating custom forward passes (currently unused)
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        # Ensure we have the correct number of features
        assert len(features) == len(
            self.heads
        ), f"Number of features ({len(features)}) must match number of head groups ({len(self.heads)})"

        # Process each feature through its corresponding heads
        for i in range(len(features)):
            for h in self.heads[i]:
                out = h(features[i])
                outputs.append(out)

        return outputs
