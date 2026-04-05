"""Shared ACE-Step constants."""

V15_DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
V15_SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""
V15_TEXT_MAX_LENGTH = 256
V15_LYRIC_MAX_LENGTH = 2048
V15_SAMPLE_RATE = 48000
V15_AUDIO_SAMPLES_PER_LATENT = 1920
