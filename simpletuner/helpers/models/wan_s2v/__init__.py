# Reuse VAE normalization utilities from wan
from simpletuner.helpers.models.wan import compute_wan_posterior, normalize_wan_latents

# Wav2Vec2 model constants
WAV2VEC2_NUM_LAYERS = 25
WAV2VEC2_DIM = 1024
