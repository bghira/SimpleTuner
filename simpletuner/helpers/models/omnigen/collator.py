import torch
from diffusers.pipelines.omnigen.processor_omnigen import OmniGenCollator


class OmniGenTrainingCollator(OmniGenCollator):
    """
    A specialized collator that works with pre-cached latents instead of raw pixels.
    """

    def __init__(
        self,
        pad_token_id: int = 2,
        hidden_size: int = 3072,
        keep_raw_resolution: bool = True,
    ):
        super().__init__(pad_token_id, hidden_size)
        self.keep_raw_resolution = keep_raw_resolution

    def __call__(self, features):
        # Extract text processing part from mllm_inputs
        mllm_inputs = [f[0] for f in features]

        # Extract pre-computed latents instead of raw images
        output_latents = [f[1] for f in features]  # These are already latents
        output_latents = torch.stack(output_latents, dim=0)

        # Process text inputs normally
        target_img_size = [
            [x.shape[-2] * 8, x.shape[-1] * 8] for x in output_latents
        ]  # Convert latent size to image size
        (
            all_padded_input_ids,
            all_position_ids,
            all_attention_mask,
            all_padding_images,
            all_pixel_values,
            all_image_sizes,
        ) = self.process_mllm_input(mllm_inputs, target_img_size)

        # Handle input image latents if needed
        input_latents = None
        if len(all_pixel_values) > 0:
            # If we have input images that would normally go through VAE,
            # they should already be pre-encoded too
            input_latents = (
                torch.cat(all_pixel_values, dim=0)
                if not isinstance(all_pixel_values[0], list)
                else all_pixel_values
            )

        # Return the processed data with latents instead of pixel values
        data = {
            "input_ids": all_padded_input_ids,
            "attention_mask": all_attention_mask,
            "position_ids": all_position_ids,
            "input_img_latents": input_latents,  # Renamed to match transformer forward params
            "input_image_sizes": all_image_sizes,
            "padding_images": all_padding_images,
            "output_latents": output_latents,  # These are now latents, not images
        }
        return data
