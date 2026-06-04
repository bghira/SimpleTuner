from __future__ import annotations

import json
import importlib.util
import types
import warnings
from dataclasses import dataclass
from posixpath import dirname as _posix_dirname, join as _posix_join
from typing import Callable, Optional, Sequence

import torch
from diffusers.loaders.lora_base import LoraBaseMixin
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from simpletuner.helpers.models.ideogram.autoencoder import (
  AutoEncoder,
  AutoEncoderParams,
  convert_diffusers_state_dict,
)
from simpletuner.helpers.models.ideogram.caption_verifier import CaptionVerifier
from simpletuner.helpers.models.ideogram.constants import (
  IMAGE_POSITION_OFFSET,
  LLM_TOKEN_INDICATOR,
  OUTPUT_IMAGE_INDICATOR,
  SEQUENCE_PADDING_INDICATOR,
  QWEN3_VL_ACTIVATION_LAYERS,
)
from simpletuner.helpers.models.ideogram.transformer import Ideogram4Config, Ideogram4Transformer
from simpletuner.helpers.models.ideogram.quantized_loading import (
  FP8_TEXT_ENCODER_CONFIG_FLAG,
  is_bnb4bit_state_dict,
  is_fp8_state_dict,
  load_bnb4bit_state_dict,
  load_fp8_state_dict,
  swap_linears_to_bnb4bit,
  swap_linears_to_fp8,
)
from simpletuner.helpers.models.ideogram.prompt_enhancer import (
  PROMPT_UPSAMPLE_TEMPERATURE,
  Ideogram4PromptEnhancerHead,
  build_caption_logits_processor,
  build_prompt_enhancer,
  generate_captions,
)
from simpletuner.helpers.models.ideogram.scheduler import (
  LogitNormalSchedule,
  get_schedule_for_resolution,
  make_step_intervals,
)


def _load_subfolder_state_dict(
  repo_id: str, subfolder: str, basename: str
) -> dict[str, torch.Tensor]:
  """Download a component's weights, whether sharded (index) or a single file.

  ``basename`` is the safetensors stem (``model`` for transformers components,
  ``diffusion_pytorch_model`` for diffusers ones).
  """
  prefix = f"{subfolder}/" if subfolder else ""
  index_filename = f"{prefix}{basename}.safetensors.index.json"
  try:
    return _load_sharded_state_dict(repo_id, index_filename)
  except EntryNotFoundError:
    single_path = hf_hub_download(
      repo_id=repo_id, filename=f"{prefix}{basename}.safetensors"
    )
    return load_file(single_path)


def _load_fp8_text_encoder(
  repo_id: str,
  device: torch.device,
  dtype: torch.dtype,
  *,
  text_encoder_subfolder: str,
):
  """Rebuild the text encoder from its config and load weight-only FP8 weights.

  transformers' ``from_pretrained`` can't read our float8 layout, so we
  instantiate the architecture with ``from_config`` (which also computes the
  non-persistent buffers such as rotary caches), swap the quantized Linears, and
  load the FP8 state dict with ``assign=True``.
  """
  config = AutoConfig.from_pretrained(
    repo_id, subfolder=text_encoder_subfolder, trust_remote_code=True
  )
  model = AutoModel.from_config(config, trust_remote_code=True)
  state_dict = _load_subfolder_state_dict(repo_id, text_encoder_subfolder, "model")
  swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
  # assign=True so unquantized params take the loaded dtype and the computed
  # rotary buffers (absent from the checkpoint) survive; tied weights, if any,
  # surface as benign missing keys.
  load_fp8_state_dict(
    model, state_dict, device=device, dtype=dtype, assign=True, strict=False
  )
  model.eval()
  return model


def _load_qwen3_vl(
  repo_id: str,
  device: torch.device,
  dtype: torch.dtype,
  *,
  tokenizer_subfolder: str | None = None,
  text_encoder_subfolder: str | None = None,
):
  """Load the Qwen3-VL tokenizer + model, optionally from named subfolders of ``repo_id``.

  When the weights are published in diffusers layout the tokenizer lives at ``tokenizer/``
  and the model at ``text_encoder/`` within the same repo as the transformer weights, so
  there is no need to fetch them from a separate upstream repo.

  If the saved ``text_encoder/config.json`` carries a ``quantization_config`` (e.g.
  a bitsandbytes 4-bit checkpoint), transformers handles the bnb placement via
  ``device_map`` and we skip the explicit ``.to(device)`` move afterwards.
  """
  tokenizer_kwargs = {"subfolder": tokenizer_subfolder} if tokenizer_subfolder else {}
  model_kwargs = {"subfolder": text_encoder_subfolder} if text_encoder_subfolder else {}
  tokenizer = AutoTokenizer.from_pretrained(repo_id, **tokenizer_kwargs)

  cfg_path = hf_hub_download(
    repo_id=repo_id,
    filename=f"{text_encoder_subfolder}/config.json"
    if text_encoder_subfolder
    else "config.json",
  )
  with open(cfg_path) as f:
    cfg_data = json.load(f)
  is_quantized = "quantization_config" in cfg_data
  is_fp8 = bool(cfg_data.get(FP8_TEXT_ENCODER_CONFIG_FLAG, False))

  if is_fp8:
    model = _load_fp8_text_encoder(
      repo_id,
      device,
      dtype,
      text_encoder_subfolder=text_encoder_subfolder or "",
    )
  elif is_quantized:
    model = AutoModel.from_pretrained(
      repo_id,
      torch_dtype=dtype,
      trust_remote_code=True,
      device_map={"": device},
      **model_kwargs,
    )
    model.eval()
  else:
    model = AutoModel.from_pretrained(
      repo_id, torch_dtype=dtype, trust_remote_code=True, **model_kwargs
    )
    model.to(device)
    model.eval()
  return tokenizer, model


def _build_transformer(
  transformer_config: "Ideogram4Config",
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
) -> "Ideogram4Transformer":
  model = Ideogram4Transformer(transformer_config)
  if is_bnb4bit_state_dict(state_dict):
    if device.type != "cuda":
      raise ValueError(f"bnb 4-bit weights require a CUDA device, got device={device}")
    swap_linears_to_bnb4bit(model, compute_dtype=dtype)
    load_bnb4bit_state_dict(model, state_dict, device=device, dtype=dtype)
  elif is_fp8_state_dict(state_dict):
    # Weight-only FP8: cast the unquantized params to the compute dtype first,
    # then swap in Fp8Linear layers (which keep their weights as float8).
    model.to(dtype)
    swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
    load_fp8_state_dict(model, state_dict, device=device, dtype=dtype)
  else:
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
  model.eval()
  return model


def _load_autoencoder(weights_path: str, device: torch.device, dtype: torch.dtype):
  ae = AutoEncoder(AutoEncoderParams())
  state_dict = convert_diffusers_state_dict(load_file(weights_path))
  ae.load_state_dict(state_dict)
  ae.to(device=device, dtype=dtype)
  ae.eval()
  return ae


def _load_sharded_state_dict(
  repo_id: str, index_filename: str
) -> dict[str, torch.Tensor]:
  """Download a sharded safetensors checkpoint and merge it into one state dict.

  ``index_filename`` is the path of the safetensors index file inside the repo
  (e.g. ``conditional_model/model.safetensors.index.json``). Shard filenames in
  the index are interpreted relative to that index's directory, matching the
  layout written by ``huggingface_hub.save_torch_state_dict``.
  """
  index_path = hf_hub_download(repo_id=repo_id, filename=index_filename)
  with open(index_path) as f:
    index = json.load(f)
  weight_map: dict[str, str] = index["weight_map"]
  shard_dir = _posix_dirname(index_filename)
  shard_filenames = sorted(set(weight_map.values()))

  state_dict: dict[str, torch.Tensor] = {}
  for shard in shard_filenames:
    shard_repo_path = _posix_join(shard_dir, shard) if shard_dir else shard
    shard_path = hf_hub_download(repo_id=repo_id, filename=shard_repo_path)
    state_dict.update(load_file(shard_path))
  return state_dict


def _load_indexed_or_single_state_dict(
  repo_id: str, index_filename: str
) -> dict[str, torch.Tensor]:
  """Load a component whether published as a sharded index or a single file.

  Some repos publish each component as a single ``.safetensors`` file rather
  than a sharded checkpoint with an ``.index.json``. Try the index first and
  fall back to the single file (the index filename with ``.index.json``
  dropped) when it isn't present.
  """
  try:
    return _load_sharded_state_dict(repo_id, index_filename)
  except EntryNotFoundError:
    single_filename = index_filename.removesuffix(".index.json")
    single_path = hf_hub_download(repo_id=repo_id, filename=single_filename)
    return load_file(single_path)


@dataclass
class Ideogram4PipelineConfig:
  weights_repo: str = "ideogram-ai/ideogram-4-nf4"
  conditional_index_filename: str = (
    "transformer/diffusion_pytorch_model.safetensors.index.json"
  )
  unconditional_index_filename: str = (
    "unconditional_transformer/diffusion_pytorch_model.safetensors.index.json"
  )
  autoencoder_filename: str = "vae/diffusion_pytorch_model.safetensors"
  text_encoder_subfolder: str = "text_encoder"
  tokenizer_subfolder: str = "tokenizer"
  patch_size: int = 2
  ae_scale_factor: int = 8
  max_text_tokens: int = 2048


class Ideogram4LoraLoaderMixin(LoraBaseMixin):
  """LoRA save/load surface for the vendored Ideogram 4 pipeline."""

  _lora_loadable_modules = ["transformer", "text_encoder"]
  transformer_name = "transformer"
  text_encoder_name = "text_encoder"

  @classmethod
  def save_lora_weights(
    cls,
    save_directory: str,
    transformer_lora_layers: dict[str, torch.nn.Module | torch.Tensor] | None = None,
    text_encoder_lora_layers: dict[str, torch.nn.Module | torch.Tensor] | None = None,
    is_main_process: bool = True,
    weight_name: str | None = None,
    save_function: Callable | None = None,
    safe_serialization: bool = True,
    transformer_lora_adapter_metadata: dict | None = None,
    text_encoder_lora_adapter_metadata: dict | None = None,
  ) -> None:
    lora_layers = {}
    lora_metadata = {}

    if transformer_lora_layers:
      lora_layers[cls.transformer_name] = transformer_lora_layers
      lora_metadata[cls.transformer_name] = transformer_lora_adapter_metadata

    if text_encoder_lora_layers:
      lora_layers[cls.text_encoder_name] = text_encoder_lora_layers
      lora_metadata[cls.text_encoder_name] = text_encoder_lora_adapter_metadata

    if not lora_layers:
      raise ValueError("You must pass at least one of `transformer_lora_layers` or `text_encoder_lora_layers`.")

    cls._save_lora_weights(
      save_directory=save_directory,
      lora_layers=lora_layers,
      lora_metadata=lora_metadata,
      is_main_process=is_main_process,
      weight_name=weight_name,
      save_function=save_function,
      safe_serialization=safe_serialization,
    )


class Ideogram4Pipeline(Ideogram4LoraLoaderMixin):
  """Ideogram 4 text-to-image pipeline."""

  def __init__(
    self,
    conditional_transformer: Ideogram4Transformer,
    unconditional_transformer: Ideogram4Transformer | None,
    text_encoder,
    text_tokenizer,
    autoencoder,
    config: Ideogram4PipelineConfig,
    device: torch.device,
    dtype: torch.dtype,
    prompt_enhancer_head: Ideogram4PromptEnhancerHead | None = None,
  ) -> None:
    self.conditional_transformer = conditional_transformer
    self.unconditional_transformer = unconditional_transformer
    self.text_encoder = text_encoder
    self.text_tokenizer = text_tokenizer
    self.autoencoder = autoencoder
    self.prompt_enhancer_head = prompt_enhancer_head
    self.config = config
    self.device = device
    self.dtype = dtype
    self.caption_verifier = CaptionVerifier()
    self._progress_bar_config = {}
    self._prompt_enhancer = None
    self._caption_logits_processor = None
    self._prompt_upsample_unconstrained_warned = False

  def to(
    self,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> "Ideogram4Pipeline":
    if device is not None:
      device = torch.device(device)
      self.device = device
    if dtype is not None:
      self.dtype = dtype

    for module in (
      self.conditional_transformer,
      self.unconditional_transformer,
      self.text_encoder,
      self.prompt_enhancer_head,
      self.autoencoder,
    ):
      if module is not None:
        if dtype is not None:
          module.to(device=self.device, dtype=dtype)
        else:
          module.to(device=self.device)
    return self

  def set_progress_bar_config(self, **kwargs) -> None:
    self._progress_bar_config.update(kwargs)

  @classmethod
  def from_pretrained(
    cls,
    *,
    config: Optional[Ideogram4PipelineConfig] = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    transformer_config: Optional[Ideogram4Config] = None,
    prompt_enhancer_head: Ideogram4PromptEnhancerHead | None = None,
  ) -> "Ideogram4Pipeline":
    config = config or Ideogram4PipelineConfig()
    transformer_config = transformer_config or Ideogram4Config()
    device = torch.device(device)

    conditional_state_dict = _load_indexed_or_single_state_dict(
      config.weights_repo, config.conditional_index_filename
    )
    unconditional_state_dict = _load_indexed_or_single_state_dict(
      config.weights_repo, config.unconditional_index_filename
    )
    autoencoder_weights = hf_hub_download(
      repo_id=config.weights_repo, filename=config.autoencoder_filename
    )

    conditional_transformer = _build_transformer(
      transformer_config, conditional_state_dict, device, dtype
    )
    del conditional_state_dict
    unconditional_transformer = _build_transformer(
      transformer_config, unconditional_state_dict, device, dtype
    )
    del unconditional_state_dict

    text_tokenizer, text_encoder = _load_qwen3_vl(
      config.weights_repo,
      device,
      dtype,
      tokenizer_subfolder=config.tokenizer_subfolder,
      text_encoder_subfolder=config.text_encoder_subfolder,
    )
    autoencoder = _load_autoencoder(autoencoder_weights, device, dtype)

    return cls(
      conditional_transformer=conditional_transformer,
      unconditional_transformer=unconditional_transformer,
      text_encoder=text_encoder,
      text_tokenizer=text_tokenizer,
      autoencoder=autoencoder,
      config=config,
      device=device,
      dtype=dtype,
      prompt_enhancer_head=prompt_enhancer_head,
    )

  def upsample_prompt(
    self,
    prompt: str | list[str],
    height: int = 2048,
    width: int = 2048,
    temperature: float = PROMPT_UPSAMPLE_TEMPERATURE,
    max_new_tokens: int = 1024,
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | None = None,
  ) -> list[str]:
    """Rewrite prompt(s) into Ideogram 4's structured JSON caption schema."""
    if self.prompt_enhancer_head is None:
      raise ValueError(
        "Prompt upsampling requires a prompt_enhancer_head. Load "
        "Ideogram4PromptEnhancerHead and pass it to Ideogram4Pipeline."
      )
    if self._prompt_enhancer is None:
      self._prompt_enhancer = build_prompt_enhancer(self.text_encoder, self.prompt_enhancer_head)
    if self._caption_logits_processor is None and importlib.util.find_spec("outlines") is not None:
      self._caption_logits_processor = build_caption_logits_processor(self._prompt_enhancer, self.text_tokenizer)
    if self._caption_logits_processor is None and not self._prompt_upsample_unconstrained_warned:
      warnings.warn(
        "`outlines` is not installed; Ideogram prompt upsampling runs unconstrained and may not return schema-valid JSON.",
        stacklevel=2,
      )
      self._prompt_upsample_unconstrained_warned = True

    return generate_captions(
      self._prompt_enhancer,
      self.text_tokenizer,
      self._caption_logits_processor,
      prompt,
      height,
      width,
      temperature=temperature,
      max_new_tokens=max_new_tokens,
      generator=generator,
      device=device or self.device,
    )

  def _tokenize(self, prompt: str) -> tuple[torch.Tensor, int]:
    """Build chat-formatted token ids for a single prompt."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = self.text_tokenizer.apply_chat_template(
      messages, add_generation_prompt=True, tokenize=False
    )
    encoded = self.text_tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    num_text_tokens = int(token_ids.shape[0])
    if num_text_tokens > self.config.max_text_tokens:
      raise ValueError(
        f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={self.config.max_text_tokens}"
      )
    return token_ids, num_text_tokens

  def _build_inputs(
    self,
    prompts: list[str],
    height: int,
    width: int,
  ) -> dict[str, torch.Tensor]:
    """Build the packed sequence (text tokens + image tokens) for one batch."""
    tokenized = [self._tokenize(p) for p in prompts]
    batch_size = len(prompts)

    patch = self.config.patch_size * self.config.ae_scale_factor
    if height % patch != 0 or width % patch != 0:
      raise ValueError(
        f"height/width must be divisible by patch_size*ae_scale_factor={patch}"
      )
    grid_h = height // patch
    grid_w = width // patch
    num_image_tokens = grid_h * grid_w

    max_text_tokens = max(num_text for _, num_text in tokenized)
    total_seq_len = max_text_tokens + num_image_tokens

    # Image position ids (t=0, h, w) offset to keep them disjoint from text positions.
    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    token_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
    text_position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
    position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)

    segment_ids = torch.full(
      (batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long
    )
    indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

    for b, (toks, num_text) in enumerate(tokenized):
      pad_len = max_text_tokens - num_text
      total_unpadded = num_text + num_image_tokens

      # Layout: [pad_len zeros] [text tokens] [image tokens]
      offset = pad_len
      token_ids[b, offset : offset + num_text] = toks
      # Image token slots stay at 0.

      text_pos = torch.arange(num_text)
      text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
      text_position_ids[b, offset : offset + num_text] = text_pos_3d
      position_ids[b, offset : offset + num_text] = text_pos_3d
      position_ids[b, offset + num_text :] = image_pos

      indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
      indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

      # Segment id 1 for the (text+image) sample, padding stays at 0.
      segment_ids[b, offset : offset + total_unpadded] = 1

    return {
      "token_ids": token_ids.to(self.device),
      "text_position_ids": text_position_ids.to(self.device),
      "position_ids": position_ids.to(self.device),
      "segment_ids": segment_ids.to(self.device),
      "indicator": indicator.to(self.device),
      "num_image_tokens": num_image_tokens,  # type: ignore[dict-item]
      "grid_h": grid_h,  # type: ignore[dict-item]
      "grid_w": grid_w,  # type: ignore[dict-item]
      "max_text_tokens": max_text_tokens,  # type: ignore[dict-item]
    }

  def _build_inputs_from_embeds(
    self,
    prompt_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    height: int,
    width: int,
  ) -> dict[str, torch.Tensor]:
    batch_size, max_text_tokens, _ = prompt_embeds.shape
    if attention_mask is None:
      attention_mask = torch.ones(
        batch_size, max_text_tokens, dtype=torch.bool, device=prompt_embeds.device
      )
    else:
      attention_mask = attention_mask.to(device=prompt_embeds.device, dtype=torch.bool)

    patch = self.config.patch_size * self.config.ae_scale_factor
    if height % patch != 0 or width % patch != 0:
      raise ValueError(
        f"height/width must be divisible by patch_size*ae_scale_factor={patch}"
      )
    grid_h = height // patch
    grid_w = width // patch
    num_image_tokens = grid_h * grid_w
    total_seq_len = max_text_tokens + num_image_tokens

    h_idx = torch.arange(grid_h, device=prompt_embeds.device).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w, device=prompt_embeds.device).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    text_pos = torch.arange(max_text_tokens, device=prompt_embeds.device)
    text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
    position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long, device=prompt_embeds.device)
    text_position_ids = torch.zeros_like(position_ids)
    position_ids[:, :max_text_tokens] = text_pos_3d
    text_position_ids[:, :max_text_tokens] = text_pos_3d
    position_ids[:, max_text_tokens:] = image_pos

    segment_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long, device=prompt_embeds.device)
    segment_ids[:, :max_text_tokens] = attention_mask.to(torch.long)
    segment_ids[:, max_text_tokens:] = 1

    indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long, device=prompt_embeds.device)
    indicator[:, :max_text_tokens] = torch.where(
      attention_mask,
      torch.full_like(attention_mask, LLM_TOKEN_INDICATOR, dtype=torch.long),
      torch.zeros_like(attention_mask, dtype=torch.long),
    )
    indicator[:, max_text_tokens:] = OUTPUT_IMAGE_INDICATOR

    llm_features = torch.zeros(
      batch_size,
      total_seq_len,
      prompt_embeds.shape[-1],
      dtype=prompt_embeds.dtype,
      device=prompt_embeds.device,
    )
    llm_features[:, :max_text_tokens] = prompt_embeds

    return {
      "text_position_ids": text_position_ids.to(self.device),
      "position_ids": position_ids.to(self.device),
      "segment_ids": segment_ids.to(self.device),
      "indicator": indicator.to(self.device),
      "num_image_tokens": num_image_tokens,  # type: ignore[dict-item]
      "grid_h": grid_h,  # type: ignore[dict-item]
      "grid_w": grid_w,  # type: ignore[dict-item]
      "max_text_tokens": max_text_tokens,  # type: ignore[dict-item]
      "prompt_embeds": llm_features.to(device=self.device, dtype=torch.float32),
      "attention_mask": attention_mask.to(device=self.device),
    }

  def _get_qwen3_vl_embeddings(
    self,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_2d: torch.Tensor,
  ) -> list[torch.Tensor]:
    language_model = self.text_encoder.language_model

    inputs_embeds = language_model.embed_tokens(token_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    causal_mask = create_causal_mask(
      config=language_model.config,
      inputs_embeds=inputs_embeds,
      attention_mask=attention_mask,
      past_key_values=None,
      position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
      hidden_states = decoder_layer(
        hidden_states,
        attention_mask=causal_mask,
        position_ids=text_position_ids,
        past_key_values=None,
        position_embeddings=position_embeddings,
      )
      if layer_idx in tap_set:
        captured[layer_idx] = hidden_states

    return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]

  def _encode_text(
    self,
    token_ids: torch.Tensor,
    text_position_ids: torch.Tensor,
    indicator: torch.Tensor,
  ) -> torch.Tensor:
    """Run Qwen3-VL and stack hidden states from the activation layers.

    Returns a (B, L, hidden_size * num_layers) float32 tensor.
    """
    batch_size, seq_len = token_ids.shape

    # Real text positions are exactly the LLM_TOKEN_INDICATOR positions.
    attention_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)

    pos_2d = text_position_ids[..., 0].contiguous()

    with torch.no_grad():
      selected = self._get_qwen3_vl_embeddings(token_ids, attention_mask, pos_2d)
    stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
    stacked = torch.permute(stacked, (1, 2, 3, 0))
    stacked = stacked.reshape(batch_size, seq_len, -1)

    # Zero out non-LLM positions (left padding) so the transformer only sees real
    # text features at LLM_TOKEN_INDICATOR positions.
    text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
    stacked = stacked * text_mask
    return stacked.to(torch.float32)

  def _verify_prompts(
    self, prompts: list[str], *, raise_on_issues: bool = True
  ) -> None:
    """Run each prompt through the caption verifier.

    Raises ``ValueError`` if any prompt has issues. When ``raise_on_issues``
    is False, issues are emitted as warnings instead.
    """
    messages: list[str] = []
    for i, prompt in enumerate(prompts):
      issues = self.caption_verifier.verify_raw(prompt)
      if not issues:
        continue
      messages.append(f"caption verifier flagged prompt[{i}]:\n" + "\n".join(issues))
    if not messages:
      return
    combined = "\n".join(messages)
    if raise_on_issues:
      raise ValueError(combined)
    warnings.warn(combined, stacklevel=2)

  @staticmethod
  def _normalize_negative_prompts(
    negative_prompts: str | list[str] | None,
    batch_size: int,
  ) -> list[str]:
    if negative_prompts is None:
      return [""] * batch_size
    if isinstance(negative_prompts, str):
      return [negative_prompts] * batch_size
    if len(negative_prompts) != batch_size:
      raise ValueError(
        f"negative_prompts must have length {batch_size}, got {len(negative_prompts)}"
      )
    return negative_prompts

  @torch.no_grad()
  def __call__(
    self,
    prompts: str | list[str] | None = None,
    *,
    prompt_embeds: torch.Tensor | None = None,
    prompt_attention_mask: torch.Tensor | None = None,
    negative_prompts: str | list[str] | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_prompt_attention_mask: torch.Tensor | None = None,
    height: int = 1024,
    width: int = 1024,
    num_steps: int = 128,
    guidance_scale: float = 7.0,
    guidance_schedule: Optional[Sequence[float] | torch.Tensor] = None,
    mu: float = 0.0,
    std: float = 1.5,
    prompt_upsampling: bool = False,
    prompt_upsampling_temperature: float = PROMPT_UPSAMPLE_TEMPERATURE,
    seed: Optional[int] = None,
    schedule: Optional[LogitNormalSchedule] = None,
    raise_on_caption_issues: bool = True,
  ) -> list[Image.Image]:
    """Generate images for the given prompts."""
    if prompts is not None and isinstance(prompts, str):
      prompts = [prompts]

    if prompt_embeds is None:
      if prompts is None:
        raise ValueError("Either prompts or prompt_embeds must be provided.")
      if prompt_upsampling:
        prompts = self.upsample_prompt(
          prompts,
          height=height,
          width=width,
          temperature=prompt_upsampling_temperature,
          max_new_tokens=self.config.max_text_tokens,
          device=self.device,
        )
      self._verify_prompts(prompts, raise_on_issues=raise_on_caption_issues)

    schedule = schedule or get_schedule_for_resolution(
      (height, width), known_mean=mu, std=std
    )
    step_intervals = make_step_intervals(num_steps).to(self.device)

    if guidance_schedule is not None:
      gw_per_step = torch.as_tensor(
        guidance_schedule, dtype=torch.float32, device=self.device
      )
      if gw_per_step.shape != (num_steps,):
        raise ValueError(
          f"guidance_schedule must have shape ({num_steps},), "
          f"got {tuple(gw_per_step.shape)}"
        )
    else:
      gw_per_step = torch.full(
        (num_steps,), float(guidance_scale), dtype=torch.float32, device=self.device
      )
    do_cfg = bool(torch.any(gw_per_step > 1.0).item())

    if prompt_embeds is None:
      inputs = self._build_inputs(prompts, height=height, width=width)
      llm_features = self._encode_text(
        inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
      )
      batch_size = len(prompts)
    else:
      if prompt_embeds.dim() == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
      if prompt_attention_mask is not None and prompt_attention_mask.dim() == 1:
        prompt_attention_mask = prompt_attention_mask.unsqueeze(0)
      inputs = self._build_inputs_from_embeds(prompt_embeds, prompt_attention_mask, height=height, width=width)
      llm_features = inputs["prompt_embeds"]
      batch_size = prompt_embeds.shape[0]
    num_image_tokens = inputs["num_image_tokens"]
    grid_h, grid_w = inputs["grid_h"], inputs["grid_w"]
    max_text_tokens = inputs["max_text_tokens"]
    latent_dim = self.conditional_transformer.config.in_channels

    neg_inputs = None
    neg_llm_features = None
    neg_text_z_padding = None
    neg_max_text_tokens = 0
    if do_cfg and self.unconditional_transformer is None:
      if negative_prompt_embeds is not None:
        if negative_prompt_embeds.dim() == 2:
          negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)
        if negative_prompt_attention_mask is not None and negative_prompt_attention_mask.dim() == 1:
          negative_prompt_attention_mask = negative_prompt_attention_mask.unsqueeze(0)
        neg_inputs = self._build_inputs_from_embeds(
          negative_prompt_embeds,
          negative_prompt_attention_mask,
          height=height,
          width=width,
        )
        neg_llm_features = neg_inputs["prompt_embeds"]
        neg_max_text_tokens = neg_inputs["max_text_tokens"]
      else:
        negative_prompts = self._normalize_negative_prompts(negative_prompts, batch_size)
        self._verify_prompts(negative_prompts, raise_on_issues=False)
        neg_inputs = self._build_inputs(negative_prompts, height=height, width=width)
        neg_max_text_tokens = neg_inputs["max_text_tokens"]
        neg_llm_features = self._encode_text(
          neg_inputs["token_ids"], neg_inputs["text_position_ids"], neg_inputs["indicator"]
        )
      neg_text_z_padding = torch.zeros(  # type: ignore[call-overload]
        batch_size,
        neg_max_text_tokens,
        latent_dim,
        dtype=torch.float32,
        device=self.device,
      )
    elif do_cfg:
      # Separate unconditional branch is image-only (asymmetric CFG) with zeroed conditioning.
      neg_inputs = {
        "position_ids": inputs["position_ids"][:, max_text_tokens:],
        "segment_ids": inputs["segment_ids"][:, max_text_tokens:],
        "indicator": inputs["indicator"][:, max_text_tokens:],
      }
      neg_llm_features = torch.zeros(  # type: ignore[call-overload]
        batch_size,
        num_image_tokens,
        llm_features.shape[-1],
        dtype=llm_features.dtype,
        device=self.device,
      )

    generator = torch.Generator(device=self.device)
    if seed is not None:
      generator.manual_seed(seed)
    z = torch.randn(  # type: ignore[call-overload]
      batch_size,
      num_image_tokens,
      latent_dim,
      dtype=torch.float32,
      device=self.device,
      generator=generator,
    )

    text_z_padding = torch.zeros(  # type: ignore[call-overload]
      batch_size,
      max_text_tokens,
      latent_dim,
      dtype=torch.float32,
      device=self.device,
    )

    for i in range(num_steps - 1, -1, -1):
      t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
      s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
      t = torch.full((batch_size,), t_val, dtype=torch.float32, device=self.device)

      pos_z = torch.cat([text_z_padding, z], dim=1)
      pos_out = self.conditional_transformer(
        llm_features=llm_features,
        x=pos_z,
        t=t,
        position_ids=inputs["position_ids"],
        segment_ids=inputs["segment_ids"],
        indicator=inputs["indicator"],
      )
      pos_v = pos_out[:, max_text_tokens:]

      gw_i = gw_per_step[i]
      if do_cfg:
        if self.unconditional_transformer is None:
          neg_z = torch.cat([neg_text_z_padding, z], dim=1)
          neg_out = self.conditional_transformer(
            llm_features=neg_llm_features,
            x=neg_z,
            t=t,
            position_ids=neg_inputs["position_ids"],
            segment_ids=neg_inputs["segment_ids"],
            indicator=neg_inputs["indicator"],
          )
          neg_v = neg_out[:, neg_max_text_tokens:]
        else:
          neg_v = self.unconditional_transformer(
            llm_features=neg_llm_features,
            x=z,
            t=t,
            position_ids=neg_inputs["position_ids"],
            segment_ids=neg_inputs["segment_ids"],
            indicator=neg_inputs["indicator"],
          )
        v = gw_i * pos_v + (1.0 - gw_i) * neg_v
      else:
        v = pos_v
      delta = s_val - t_val
      z = z + v * delta

    return types.SimpleNamespace(images=self._decode(z, grid_h=grid_h, grid_w=grid_w))  # type: ignore[arg-type]

  def _decode(self, z: torch.Tensor, *, grid_h: int, grid_w: int) -> list[Image.Image]:
    """Unpatch and run the autoencoder decoder."""
    batch_size = z.shape[0]
    patch = self.config.patch_size

    ae_channels = z.shape[-1] // (patch * patch)
    z = z.view(batch_size, grid_h, grid_w, z.shape[-1]).permute(0, 3, 1, 2).contiguous()
    bn_mean = self.autoencoder.bn.running_mean.view(1, -1, 1, 1).to(device=z.device, dtype=z.dtype)
    bn_std = torch.sqrt(self.autoencoder.bn.running_var.view(1, -1, 1, 1) + self.autoencoder.bn.eps).to(
      device=z.device, dtype=z.dtype
    )
    z = z * bn_std + bn_mean

    z = z.view(batch_size, ae_channels, patch, patch, grid_h, grid_w)
    z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
    z = z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)

    z = z.to(self.dtype)
    decoded = self.autoencoder.decoder(z)

    decoded = decoded.float().clamp(-1.0, 1.0)
    decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
    decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(arr) for arr in decoded]
