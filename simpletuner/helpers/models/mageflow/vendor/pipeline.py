"""MageFlow text-to-image + image-edit inference pipeline.

Self-contained MageFlow t2i / edit inference: load a HuggingFace diffusers-style
repo (model_index.json + transformer/ vae/ scheduler/), then generate or edit
images. No training/eval deps.

Both ``generate_images`` and ``generate_edits`` support PACKED multi-resolution
inference: several samples (each at its own resolution) are concatenated into a
single varlen sequence and processed in one transformer forward per denoise
step. Per-sample ``cu_seqlens`` (inside the flash-attn varlen kernel) isolate
samples, exactly mirroring training-time packing. These packed functions are the
sole implementation — the single-image case is just a pack of size 1, exposed
via the ``MageFlowPipeline.generate`` / ``.edit`` convenience methods.
"""

from __future__ import annotations

import json
import os
import random

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from PIL import Image

from .models.mage_flow import MageFlowModel, ModelConfig
from .models.modules.mage_latent import encode_noise, resolve_gs_key
from .models.modules.mage_text import make_refusal_image
from .models.utils import PROMPT_TEMPLATE, get_noise, unpack


# ---------------------------------------------------------------------------
# Scheduler — diffusers FlowMatchEulerDiscreteScheduler
# ---------------------------------------------------------------------------
def build_scheduler(num_steps: int, device=None, shift: float = 6.0):
    """Construct a diffusers ``FlowMatchEulerDiscreteScheduler`` whose sigma
    schedule reproduces our default preset exactly.

    The base sigmas ``linspace(1, 1/num_steps, num_steps)`` fed to
    ``set_timesteps`` are run through the scheduler's built-in static shift
    ``shift·s/(1+(shift-1)·s)`` and a terminal 0 is appended — the static-shift
    schedule (the only supported schedule).
    """
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    base_sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps).tolist()
    scheduler.set_timesteps(sigmas=base_sigmas, device=device)
    return scheduler


def _get_scheduler(model, steps, device, static_shift):
    scheduler = getattr(model, "scheduler", None)
    if scheduler is None:
        return build_scheduler(steps, device=device, shift=(static_shift if static_shift is not None else 6.0))
    if static_shift is not None:
        scheduler.set_shift(static_shift)
    scheduler.set_timesteps(sigmas=torch.linspace(1.0, 1.0 / steps, steps).tolist(), device=device)
    return scheduler


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _template_info(name: str | None) -> dict:
    name = name or "mage-flow"
    if name not in PROMPT_TEMPLATE:
        raise ValueError(f"Unknown prompt template: {name}")
    return PROMPT_TEMPLATE[name]


def _as_list(val, default, n):
    """Broadcast a scalar/None to a length-n list, or validate a given list."""
    if val is None:
        return [default] * n
    if isinstance(val, (list, tuple)):
        if len(val) != n:
            raise ValueError(f"expected {n} values, got {len(val)}")
        return list(val)
    return [val] * n


def _lens_to_cu(lens, device):
    """Sequence lengths -> cumulative cu_seqlens [0, l0, l0+l1, ...] (int32)."""
    t = torch.tensor(lens, device=device, dtype=torch.int32)
    return torch.cat([torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(t, dim=0, dtype=torch.int32)])


def _make_divisible_by_16(size: int) -> int:
    return max(16, 16 * (size // 16))


def _compute_aspect_ratio_size(pil_img: Image.Image, max_size: int):
    """Longest side = ``max_size``, short side from aspect ratio, both /16."""
    w, h = pil_img.size
    if h >= w:
        new_h, new_w = max_size, int(round(w * max_size / h))
    else:
        new_w, new_h = max_size, int(round(h * max_size / w))
    return _make_divisible_by_16(new_h), _make_divisible_by_16(new_w)


def _edit_target_size(pil_img: Image.Image, max_size, height, width):
    """Output (H, W) for an edit sample, derived from its PRIMARY reference.

    Precedence: explicit ``height`` AND ``width`` (custom size) > ``max_size``
    (longest side, short side by aspect ratio) > the source image's own size.
    All rounded down to a multiple of 16.
    """
    if height and width:
        return _make_divisible_by_16(height), _make_divisible_by_16(width)
    if max_size:
        return _compute_aspect_ratio_size(pil_img, max_size)
    # Nothing specified: keep the source resolution (its own longest side).
    return _compute_aspect_ratio_size(pil_img, max(pil_img.size))


def _decode_one(model, tokens, height, width, dev):
    """Unpack one sample's image tokens [1, H*W, C] and VAE-decode to a PIL image."""
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        out = model.vae.decode(unpack(tokens.float(), height, width))
    out = rearrange(out.clamp(-1, 1), "b c h w -> b h w c")
    out = (127.5 * (out + 1.0)).cpu().byte().numpy()
    return Image.fromarray(out[0])


def _build_pack_ctx(
    img_ids,
    img_cu,
    img_shapes,
    img_lens,
    txt,
    txt_cu,
    txt_mask,
    vec,
    neg_txt,
    neg_cu,
    neg_mask,
    neg_vec,
    cfg,
    renormalization,
    batch_cfg,
    device,
):
    """Precompute the static per-step transformer inputs for a packed batch.

    When a negative branch is present and ``batch_cfg`` is True, the conditional
    and unconditional passes are fused into ONE varlen forward: the image tokens
    are duplicated (cond copy + uncond copy) and the positive/negative texts are
    concatenated, so cond sample i and uncond sample i become two independent
    varlen segments processed in a single kernel launch. flash_attn_varlen_func
    keeps every segment isolated via cu_seqlens, so this is numerically identical
    to two separate forwards — just one launch instead of two.
    """
    na = len(img_lens)
    ctx = {
        "na": na,
        "cfg": cfg,
        "renorm": renormalization,
        "batch_cfg": batch_cfg,
        "has_neg": neg_txt is not None,
        "img_ids": img_ids,
        "img_cu": img_cu,
        "img_shapes": img_shapes,
        "img_max": int(max(img_lens)),
        "txt": txt,
        "txt_ids": torch.zeros(1, txt.shape[1], 3, device=device),
        "txt_cu": txt_cu,
        "txt_mask": txt_mask,
        "vec": vec,
        "txt_max": int((txt_cu[1:] - txt_cu[:-1]).max().item()),
    }
    if neg_txt is None:
        return ctx
    ctx.update(
        {
            "neg_txt": neg_txt,
            "neg_ids": torch.zeros(1, neg_txt.shape[1], 3, device=device),
            "neg_cu": neg_cu,
            "neg_mask": neg_mask,
            "neg_vec": neg_vec,
            "neg_max": int((neg_cu[1:] - neg_cu[:-1]).max().item()),
        }
    )
    if batch_cfg:
        # Duplicate image segments (cond then uncond) and concat pos+neg text.
        d_txt = torch.cat([txt, neg_txt], dim=1)
        pos_lens = (txt_cu[1:] - txt_cu[:-1]).tolist()
        neg_lens = (neg_cu[1:] - neg_cu[:-1]).tolist()
        ctx.update(
            {
                "d_img_ids": torch.cat([img_ids, img_ids], dim=1),
                "d_img_cu": _lens_to_cu(list(img_lens) + list(img_lens), device),
                "d_img_shapes": [img_shapes[0] + img_shapes[0]],
                "d_txt": d_txt,
                "d_txt_ids": torch.zeros(1, d_txt.shape[1], 3, device=device),
                "d_txt_cu": _lens_to_cu(pos_lens + neg_lens, device),
                "d_txt_mask": torch.ones(1, d_txt.shape[1], device=device),
                "d_vec": torch.cat([vec, neg_vec], dim=0),
                "d_txt_max": int(max(pos_lens + neg_lens)),
            }
        )
    return ctx


def _velocity(transformer, img, ctx, sigma):
    """CFG-combined image-token velocity for a packed batch at noise level ``sigma``.

    Returns [1, sum_img_len, C] in the conditional sample order. When
    ``batch_cfg`` is set the cond+uncond passes share a single fused varlen
    forward; otherwise they are two forwards.
    """
    dev = img.device
    na = ctx["na"]

    def _fwd(x, n, img_ids, img_cu, img_max, img_shapes, txt, txt_ids, txt_cu, txt_mask, txt_max, vec):
        t_vec = torch.full((n,), sigma, dtype=x.dtype, device=dev)
        return transformer(
            img=x, txt=txt, timesteps=t_vec, img_shapes=img_shapes, img_cu_seqlens=img_cu, txt_cu_seqlens=txt_cu
        )

    if not ctx["has_neg"]:
        return _fwd(
            img,
            na,
            ctx["img_ids"],
            ctx["img_cu"],
            ctx["img_max"],
            ctx["img_shapes"],
            ctx["txt"],
            ctx["txt_ids"],
            ctx["txt_cu"],
            ctx["txt_mask"],
            ctx["txt_max"],
            ctx["vec"],
        )

    if ctx["batch_cfg"]:
        n_img = img.shape[1]
        out = _fwd(
            torch.cat([img, img], dim=1),
            2 * na,
            ctx["d_img_ids"],
            ctx["d_img_cu"],
            ctx["img_max"],
            ctx["d_img_shapes"],
            ctx["d_txt"],
            ctx["d_txt_ids"],
            ctx["d_txt_cu"],
            ctx["d_txt_mask"],
            ctx["d_txt_max"],
            ctx["d_vec"],
        )
        cond, unc = out[:, :n_img, :], out[:, n_img:, :]
    else:
        cond = _fwd(
            img,
            na,
            ctx["img_ids"],
            ctx["img_cu"],
            ctx["img_max"],
            ctx["img_shapes"],
            ctx["txt"],
            ctx["txt_ids"],
            ctx["txt_cu"],
            ctx["txt_mask"],
            ctx["txt_max"],
            ctx["vec"],
        )
        unc = _fwd(
            img,
            na,
            ctx["img_ids"],
            ctx["img_cu"],
            ctx["img_max"],
            ctx["img_shapes"],
            ctx["neg_txt"],
            ctx["neg_ids"],
            ctx["neg_cu"],
            ctx["neg_mask"],
            ctx["neg_max"],
            ctx["neg_vec"],
        )

    cfg = ctx["cfg"]
    if ctx["renorm"]:
        # CFG renormalization: rescale the guided velocity per token back to the
        # conditional velocity's norm (reduces oversaturation at high cfg).
        comb = unc + cfg * (cond - unc)
        return comb * (torch.norm(cond, dim=-1, keepdim=True) / (torch.norm(comb, dim=-1, keepdim=True) + 1e-6))
    return unc + cfg * (cond - unc)


def _encode_texts_packed(model, prompts, template, drop_idx, device):
    """Encode a LIST of templated text-only prompts in ONE packed varlen forward
    (``TextEncoder.forward`` — varlen cu_seqlens isolates each prompt,
    verified zero cross-contamination). Returns (txt_flat [ΣLi, D], vec [N, D],
    per-prompt token lengths list)."""
    tokenizer = model.txt_enc.tokenizer
    max_len = model.txt_enc.tokenizer_max_length + drop_idx
    ids_list = [
        tokenizer(template.format(p), max_length=max_len, truncation=True, return_tensors="pt").input_ids.squeeze(0)
        for p in prompts
    ]
    input_ids = torch.cat(ids_list).to(device)
    cu_seqlens = _lens_to_cu([int(t.numel()) for t in ids_list], device)
    res = model.txt_enc(input_ids, cu_seqlens, drop_idx_override=drop_idx)
    return res["txt"], res["vec"], res["txt_seq_lens"].tolist()


def _slice_packed(txt_flat, vec, lens, start, count, device):
    """Format a contiguous ``count``-prompt slice (starting at prompt ``start``) of a
    packed text encode into the (txt [1, ΣL, D], cu_seqlens, ones-mask, vec [count, D])
    tuple that ``_build_pack_ctx`` consumes."""
    seg_lens = lens[start : start + count]
    tok_start = sum(lens[:start])
    tok_end = tok_start + sum(seg_lens)
    txt = txt_flat[tok_start:tok_end].reshape(1, -1, txt_flat.shape[-1]).to(device)
    return (
        txt,
        _lens_to_cu(seg_lens, device),
        torch.ones(1, txt.shape[1], device=device),
        vec[start : start + count].to(device),
    )


# ---------------------------------------------------------------------------
# Text-to-image (packed, multi-resolution)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_images(
    model,
    prompts,
    neg_prompts=None,
    seeds=None,
    steps=30,
    cfg=5.0,
    heights=None,
    widths=None,
    device="cuda",
    prompt_template="mage-flow",
    static_shift=None,
    gs_key=None,
    renormalization=False,
    batch_cfg=True,
):
    """Generate one image per prompt. Prompts may request DIFFERENT resolutions;
    all are packed into a single varlen forward per denoise step — samples are
    kept isolated by ``flash_attn_varlen_func`` via per-sample ``cu_seqlens`` (no
    cross-sample attention), mirroring training-time packing. When ``cfg > 1`` and
    ``batch_cfg`` is set, the positive and negative passes are fused into that
    same varlen forward. Returns a list of PIL images aligned with ``prompts``.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    n = len(prompts)
    neg_prompts = _as_list(neg_prompts, " ", n)
    seeds = _as_list(seeds, 42, n)
    heights = _as_list(heights, 1024, n)
    widths = _as_list(widths, 1024, n)
    info = _template_info(prompt_template)
    template = info.get("template", "{}")
    drop_idx = int(info.get("start_idx", 0))
    dev = torch.device(device)

    # Content-policy gate per sample (MANDATORY — runs on the same text-encoder
    # weights as conditioning, no opt-out). Violating prompts get a refusal
    # placeholder and are dropped from the pack.
    results = [None] * n
    active = []
    for i in range(n):
        if seeds[i] == -1:
            seeds[i] = random.randint(0, 2**32 - 1)
        verdict = model.txt_enc.screen_text(prompts[i])
        if verdict.violates:
            h_, w_ = _make_divisible_by_16(heights[i]), _make_divisible_by_16(widths[i])
            print(verdict.banner())
            results[i] = make_refusal_image(verdict, height=h_, width=w_)
            continue
        active.append(i)
    if not active:
        return results

    gs_key_int = resolve_gs_key(gs_key)
    # Per-sample noise tokens + position ids + shapes (MageVAE: flatten, no packing).
    ch = model.vae.latent_channels
    img_list, ids_list, lens, shapes, hw = [], [], [], [], []
    for i in active:
        h_, w_ = _make_divisible_by_16(heights[i]), _make_divisible_by_16(widths[i])
        torch.manual_seed(seeds[i])
        x = get_noise(num_samples=1, channel=ch, height=h_, width=w_, device=dev, dtype=torch.bfloat16, seed=seeds[i])
        # Distribution-preserving watermark in the initial noise (same shape,
        # still ~N(0,1)); detect by inverting the flow ODE back to noise.
        x = encode_noise(tuple(x.shape[1:]), key=gs_key_int, seed=seeds[i], device=dev, dtype=torch.bfloat16)
        _, _, gh, gw = x.shape
        img_list.append(rearrange(x, "b c h w -> b (h w) c")[0])
        ids = torch.zeros(gh, gw, 3, device=dev)
        ids[..., 1] = ids[..., 1] + torch.arange(gh, device=dev)[:, None]
        ids[..., 2] = ids[..., 2] + torch.arange(gw, device=dev)[None, :]
        ids_list.append(rearrange(ids, "h w c -> (h w) c"))
        lens.append(gh * gw)
        shapes.append((1, gh, gw))
        hw.append((h_, w_))
    img = torch.cat(img_list, 0).unsqueeze(0)
    img_ids = torch.cat(ids_list, 0).unsqueeze(0)
    img_cu = _lens_to_cu(lens, dev)
    img_shapes = [shapes]

    # Packed text: positive prompts AND (for CFG) negative prompts are encoded
    # TOGETHER in ONE varlen forward, then split back — cu_seqlens keeps every
    # prompt isolated (verified zero cross-contamination).
    pos_prompts = [prompts[i] for i in active]
    na = len(active)
    use_neg = cfg > 1.0 and any(neg_prompts[i] for i in active)
    if use_neg:
        neg_list = [neg_prompts[i] or " " for i in active]
        txt_flat, vec_all, lens_t = _encode_texts_packed(model, pos_prompts + neg_list, template, drop_idx, dev)
        txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, lens_t, 0, na, dev)
        neg_txt, neg_cu, neg_mask, neg_vec = _slice_packed(txt_flat, vec_all, lens_t, na, na, dev)
    else:
        txt_flat, vec_all, lens_t = _encode_texts_packed(model, pos_prompts, template, drop_idx, dev)
        txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, lens_t, 0, na, dev)
        neg_txt = neg_cu = neg_mask = neg_vec = None

    ctx = _build_pack_ctx(
        img_ids,
        img_cu,
        img_shapes,
        lens,
        txt,
        txt_cu,
        txt_mask,
        vec,
        neg_txt,
        neg_cu,
        neg_mask,
        neg_vec,
        cfg,
        renormalization,
        batch_cfg,
        dev,
    )
    scheduler = _get_scheduler(model, steps, device, static_shift)
    for si, t in enumerate(scheduler.timesteps):
        pred = _velocity(model.transformer, img, ctx, scheduler.sigmas[si].item())
        img = scheduler.step(pred, t, img, return_dict=False)[0]

    off = 0
    for k, i in enumerate(active):
        L = lens[k]
        h_, w_ = hw[k]
        results[i] = _decode_one(model, img[:, off : off + L, :], h_, w_, dev)
        off += L
    return results


# ---------------------------------------------------------------------------
# Image edit (packed, multi-resolution)
# ---------------------------------------------------------------------------
def _preprocess_ref_image(pil_img: Image.Image, height: int, width: int, device) -> torch.Tensor:
    """Resize an RGB reference image to (height, width) and normalize to [-1, 1]."""
    from torchvision.transforms import functional as TF

    img = pil_img.convert("RGB")
    img = TF.resize(img, [height, width], interpolation=TF.InterpolationMode.BICUBIC)
    t = TF.to_tensor(img)  # [3, H, W] in [0, 1]
    t = TF.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1, 1]
    return t.to(device)


def _resize_long_edge(image: Image.Image, max_long_edge: int | None) -> Image.Image:
    """Cap the VL conditioning image's long edge, preserving aspect ratio.

    Matches training's data.processor._resize_long_edge (BICUBIC). Without this,
    inference feeds a full-resolution image to the Qwen-VL processor whose
    default max_pixels is far larger than 384**2 — a train/test mismatch.
    """
    if max_long_edge is None or max_long_edge <= 0:
        return image
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.BICUBIC)


# Fixed image placeholder used at edit training time (one per reference image).
_EDIT_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _edit_prompt_body(instruction, num_refs):
    """Training-time multi-reference prompt body: ``Image 1: <ph>Image 2: <ph>…{instruction}``."""
    prefix = "".join(f"Image {j}: {_EDIT_IMAGE_PLACEHOLDER}" for j in range(1, num_refs + 1))
    return prefix + instruction


def _encode_edits_packed(model, ref_pils_per_sample, instructions, template, drop_idx, device):
    """Encode ALL image-conditioned edit instructions in ONE packed multimodal
    varlen forward (pixel_values/image_grid_thw concatenated across samples,
    cu_seqlens isolates each). Returns (txt_flat [ΣLi, D], vec [N, D], per-sample lens)."""
    processor = model.txt_enc.processor
    ids_list, pv_list, thw_list = [], [], []
    for ref_pils, instr in zip(ref_pils_per_sample, instructions, strict=False):
        formatted = template.format(_edit_prompt_body(instr, len(ref_pils)))
        vl = processor(text=[formatted], images=list(ref_pils), padding=True, return_tensors="pt")
        vl = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in vl.items()}
        ids_list.append(vl["input_ids"].squeeze(0))
        if vl.get("pixel_values") is not None:
            pv_list.append(vl["pixel_values"])
            thw_list.append(vl["image_grid_thw"])
    input_ids = torch.cat(ids_list).to(device)
    cu = _lens_to_cu([int(t.numel()) for t in ids_list], device)
    inputs = {"input_ids": input_ids, "cu_seqlens": cu}
    if pv_list:
        inputs["pixel_values"] = torch.cat(pv_list, dim=0)
        inputs["image_grid_thw"] = torch.cat(thw_list, dim=0)
    res = model.txt_enc(input_ids, cu, inputs=inputs, drop_idx_override=drop_idx)
    return res["txt"], res["vec"], res["txt_seq_lens"].tolist()


@torch.no_grad()
def generate_edits(
    model,
    prompts,
    ref_images,
    neg_prompts=None,
    seeds=None,
    steps=30,
    cfg=5.0,
    max_size=None,
    heights=None,
    widths=None,
    device="cuda",
    prompt_template="mage-flow-edit",
    static_shift=None,
    gs_key=None,
    vl_cond_long_edge=384,
    renormalization=False,
    batch_cfg=True,
):
    """Edit reference image(s) per prompt. Each ``ref_images[i]`` may be a single
    image/path OR a list of source images (multi-image edit, like training —
    trained with up to 3, but more are accepted) — all produce ONE edited output. Each sample's
    ``[target, ref_1, …, ref_N]`` latent tokens are sequence-concatenated, and
    all samples are packed into one varlen forward per denoise step.

    Output resolution (derived from the first/primary reference of each sample):
    if both ``heights[i]`` and ``widths[i]`` are given, use them; else if
    ``max_size`` is given, the longest side is ``max_size`` and the short side
    follows the reference's aspect ratio; otherwise the output keeps the source
    image's own resolution. All references are VAE-encoded at that target size.
    Returns a list of PIL images.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
        ref_images = [ref_images]
    n = len(prompts)
    neg_prompts = _as_list(neg_prompts, " ", n)
    seeds = _as_list(seeds, 42, n)
    heights = _as_list(heights, None, n)
    widths = _as_list(widths, None, n)
    info = _template_info(prompt_template)
    template = info.get("template", "{}")
    drop_idx = int(info.get("start_idx", 0))
    dev = torch.device(device)

    # Normalize each sample's references to a list of 1..3 PIL images.
    def _load_pil(r):
        if isinstance(r, str):
            r = Image.open(r)
        return r.convert("RGB")

    pils_per_sample = []
    for r in ref_images:
        refs = list(r) if isinstance(r, (list, tuple)) else [r]
        if not refs:
            raise ValueError("each edit sample needs at least one reference image")
        pils_per_sample.append([_load_pil(x) for x in refs])

    # Per-sample output resolution (from the first/primary reference) + content gate.
    results = [None] * n
    res_hw = [None] * n
    active = []
    for i in range(n):
        res_hw[i] = _edit_target_size(pils_per_sample[i][0], max_size, heights[i], widths[i])
        if seeds[i] == -1:
            seeds[i] = random.randint(0, 2**32 - 1)
        # Multimodal gate (MANDATORY): inspect the source image(s) AND the
        # instruction, so NSFW / copyrighted-character / real-public-figure
        # source photos are blocked even under an innocuous instruction.
        verdict = model.txt_enc.screen_edit(prompts[i], pils_per_sample[i])
        if verdict.violates:
            h_, w_ = res_hw[i]
            print(verdict.banner())
            results[i] = make_refusal_image(verdict, height=h_, width=w_)
            continue
        active.append(i)
    if not active:
        return results

    gs_key_int = resolve_gs_key(gs_key)

    # Per sample: reference latent tokens (clean) + target noise tokens, plus the
    # combined [target, ref_1, …, ref_N] position ids and shapes. ``target_idx``
    # records where each sample's target tokens land in the packed sequence so we
    # can slice the velocity and step only the target portion.
    ch = model.vae.latent_channels
    targets, refs, ids_list, shape_seq, samp_lens, tgt_lens, hw = [], [], [], [], [], [], []
    target_idx_parts = []
    off = 0
    for i in active:
        h_, w_ = res_hw[i]
        torch.manual_seed(seeds[i])  # MageVAE.encode samples the posterior (global RNG)
        # All references resized to the target resolution and VAE-encoded together.
        ref_tensors = [_preprocess_ref_image(p, h_, w_, dev) for p in pils_per_sample[i]]
        ref_tok, ref_shapes, ref_ids = model.compute_vae_encodings(ref_tensors, with_ids=True)
        ref_tok = ref_tok.to(torch.bfloat16)  # [1, N*Lr, C]
        x = get_noise(num_samples=1, channel=ch, height=h_, width=w_, device=dev, dtype=torch.bfloat16, seed=seeds[i])
        x = encode_noise(tuple(x.shape[1:]), key=gs_key_int, seed=seeds[i], device=dev, dtype=torch.bfloat16)
        _, _, gh, gw = x.shape
        tgt = rearrange(x, "b c h w -> b (h w) c")  # [1, Lt, C]
        tgt_ids = torch.zeros(gh, gw, 3, device=dev)
        tgt_ids[..., 1] = tgt_ids[..., 1] + torch.arange(gh, device=dev)[:, None]
        tgt_ids[..., 2] = tgt_ids[..., 2] + torch.arange(gw, device=dev)[None, :]
        tgt_ids = rearrange(tgt_ids, "h w c -> (h w) c").unsqueeze(0)
        lt, lr = tgt.shape[1], ref_tok.shape[1]
        targets.append(tgt)
        refs.append(ref_tok)
        ids_list.append(torch.cat([tgt_ids, ref_ids.to(dev)], dim=1)[0])  # [Lt + N*Lr, 3]
        shape_seq.append((1, gh, gw))  # target frame idx 0
        shape_seq.extend(s[0] for s in ref_shapes)  # ref_j frame idx j
        samp_lens.append(lt + lr)
        tgt_lens.append(lt)
        hw.append((h_, w_))
        target_idx_parts.append(torch.arange(off, off + lt, device=dev))
        off += lt + lr
    img_ids = torch.cat(ids_list, 0).unsqueeze(0)
    img_cu = _lens_to_cu(samp_lens, dev)
    img_shapes = [shape_seq]
    target_idx = torch.cat(target_idx_parts)

    # Packed edit text — positive AND (for CFG) negative are encoded TOGETHER in
    # ONE packed multimodal forward, then split. Both branches share the same
    # reference images; cu_seqlens isolates every sequence (zero cross-contamination).
    # The VL conditioning image's long edge is capped (default 384) to match
    # training preprocessing — the VAE path above keeps the full target resolution.
    na = len(active)
    edit_refs = [[_resize_long_edge(p, vl_cond_long_edge) for p in pils_per_sample[i]] for i in active]
    if cfg > 1.0:
        pos_instr = [prompts[i] for i in active]
        neg_instr = [neg_prompts[i] or " " for i in active]
        txt_flat, vec_all, lens_t = _encode_edits_packed(
            model, edit_refs + edit_refs, pos_instr + neg_instr, template, drop_idx, dev
        )
        txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, lens_t, 0, na, dev)
        neg_txt, neg_cu, neg_mask, neg_vec = _slice_packed(txt_flat, vec_all, lens_t, na, na, dev)
    else:
        txt_flat, vec_all, lens_t = _encode_edits_packed(
            model, edit_refs, [prompts[i] for i in active], template, drop_idx, dev
        )
        txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, lens_t, 0, na, dev)
        neg_txt = neg_cu = neg_mask = neg_vec = None

    ctx = _build_pack_ctx(
        img_ids,
        img_cu,
        img_shapes,
        samp_lens,
        txt,
        txt_cu,
        txt_mask,
        vec,
        neg_txt,
        neg_cu,
        neg_mask,
        neg_vec,
        cfg,
        renormalization,
        batch_cfg,
        dev,
    )
    scheduler = _get_scheduler(model, steps, device, static_shift)
    for si, t in enumerate(scheduler.timesteps):
        parts = []
        for k in range(na):
            parts.append(targets[k])
            parts.append(refs[k])
        img = torch.cat(parts, dim=1)  # [1, sum(Lt+Lr), C], ref clean
        vel = _velocity(model.transformer, img, ctx, scheduler.sigmas[si].item())
        pred_t = vel[:, target_idx, :]  # [1, sum Lt, C] — target tokens only
        tgt_packed = torch.cat(targets, dim=1)  # [1, sum Lt, C]
        stepped = scheduler.step(pred_t, t, tgt_packed, return_dict=False)[0]
        o = 0
        new_targets = []
        for k in range(na):
            lt = tgt_lens[k]
            new_targets.append(stepped[:, o : o + lt, :])
            o += lt
        targets = new_targets

    for k, i in enumerate(active):
        h_, w_ = hw[k]
        results[i] = _decode_one(model, targets[k], h_, w_, dev)
    return results


# ---------------------------------------------------------------------------
# Flow-ODE inversion (Gaussian-Shading watermark detection)
# ---------------------------------------------------------------------------
@torch.no_grad()
def invert_to_noise(
    model, z0, height, width, steps=30, device="cuda", prompt_template="mage-flow", static_shift=None, prompt=""
):
    """Reverse the flow ODE from a clean latent ``z0`` back to the initial noise.

    This is the detection primitive for the Gaussian-Shading watermark: VAE-encode
    the image to ``z0`` (posterior MEAN — deterministic), run this to recover the
    initial noise, then read the signs via ``mage_latent.decode_bits``.

    Inversion uses an empty prompt at cfg=1 (the standard Tree-Ring /
    Gaussian-Shading setup). Reverse Euler recovers ``x_i`` from ``x_{i+1}`` with
    the velocity evaluated at the point in hand; the sign-only watermark tolerates
    the resulting approximation error (see the module's redundancy).

    Args:
        z0: clean latent ``[1, C, gh, gw]`` (e.g. the mean of ``model.vae.encode``).
    Returns:
        recovered initial-noise latent ``[1, C, gh, gw]`` (float32).
    """
    dev = torch.device(device)
    info = _template_info(prompt_template)
    template = info.get("template", "{}")
    drop_idx = int(info.get("start_idx", 0))

    z0 = z0.to(dev)
    _, ch, gh, gw = z0.shape
    img = rearrange(z0, "b c h w -> b (h w) c").to(torch.bfloat16)  # [1, gh*gw, C]

    ids = torch.zeros(gh, gw, 3, device=dev)
    ids[..., 1] = ids[..., 1] + torch.arange(gh, device=dev)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(gw, device=dev)[None, :]
    img_ids = rearrange(ids, "h w c -> (h w) c").unsqueeze(0)
    lens = [gh * gw]
    img_cu = _lens_to_cu(lens, dev)
    img_shapes = [[(1, gh, gw)]]

    # Empty-prompt conditioning, no negative branch, cfg=1 (single forward).
    txt_flat, vec_all, lens_t = _encode_texts_packed(model, [prompt], template, drop_idx, dev)
    txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, lens_t, 0, 1, dev)
    ctx = _build_pack_ctx(
        img_ids, img_cu, img_shapes, lens, txt, txt_cu, txt_mask, vec, None, None, None, None, 1.0, False, False, dev
    )

    scheduler = _get_scheduler(model, steps, device, static_shift)
    sigmas = scheduler.sigmas
    n = len(scheduler.timesteps)
    # Forward step si: x_{si+1} = x_si + (s_{si+1}-s_si)·v(x_si, s_si).
    # Reverse it from clean (x_n, sigma 0) up to noise (x_0), using x_{si+1} as the
    # proxy for x_si at the forward eval sigma s_si.
    for si in range(n - 1, -1, -1):
        s_cur = sigmas[si].item()
        s_next = sigmas[si + 1].item()
        vel = _velocity(model.transformer, img, ctx, s_cur)
        img = img - (s_next - s_cur) * vel
    return unpack(img.float(), height, width)  # [1, C, gh, gw]


# ---------------------------------------------------------------------------
# High-level pipeline wrapper
# ---------------------------------------------------------------------------
class MageFlowPipeline:
    """``MageFlowPipeline.from_pretrained(repo).generate(...) / .edit(...)``.

    ``generate`` / ``edit`` are packed multi-resolution calls: they take a list
    of prompts (a single string is accepted and treated as a pack of size 1) and
    return a list of PIL images. Per-sample ``heights``/``widths``/``seeds`` are
    lists. Every prompt is screened by the text encoder's mandatory content
    gate (no opt-out); banned prompts come back as refusal placeholders
    interleaved with the real images. Real outputs always carry a Gaussian-Shading
    watermark in the initial noise (no toggle), using the configured secret key.
    """

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(cls, repo_dir: str, device: str = "cuda"):
        """Load a Mage-Flow diffusers-style repo (``model_index.json`` +
        ``transformer/`` ``vae/`` ``scheduler/`` ``text_encoder/``).

        ``repo_dir`` may be a local directory OR a Hugging Face Hub repo id
        (e.g. ``"microsoft/Mage-Flow-4B"``), which is downloaded and cached
        automatically on first use.
        """
        return cls(load_from_repo(repo_dir, device), device)

    def generate(self, prompts, **kw) -> list[Image.Image]:
        """Packed multi-resolution t2i. ``prompts`` is a list (or a single
        string); pass per-sample ``heights``/``widths``/``seeds`` as lists."""
        kw.setdefault("device", self.device)
        return generate_images(self.model, prompts, **kw)

    def edit(self, prompts, ref_images, **kw) -> list[Image.Image]:
        """Packed multi-resolution edit. ``prompts`` is a list (or a single
        string); each ``ref_images[i]`` is one reference or a list of references."""
        kw.setdefault("device", self.device)
        return generate_edits(self.model, prompts, ref_images, **kw)

    def invert_to_noise(self, z0, height, width, **kw):
        """Recover the initial noise from a clean latent (Gaussian-Shading detect)."""
        kw.setdefault("device", self.device)
        return invert_to_noise(self.model, z0, height, width, **kw)


def _safe_subpath(root: str, *parts: str) -> str:
    """Join ``parts`` under ``root`` and confirm the result stays inside ``root``.

    ``root`` is normalized up front; the joined path is normalized **lexically**
    (``os.path.normpath`` — symlinks are *not* followed, so a Hugging Face cache
    whose weight files are symlinks into the shared blob store still loads) and
    rejected if it escapes ``root``. This guards the user-supplied model path
    against path traversal (CWE-22 / CodeQL ``py/path-injection``).
    """
    root = os.path.realpath(root)
    full = os.path.normpath(os.path.join(root, *parts))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError(f"Resolved path {os.path.join(*parts)!r} escapes repo directory {root!r}")
    return full


def _resolve_repo_dir(repo_dir: str) -> str:
    """Return a local directory for ``repo_dir``.

    If ``repo_dir`` is an existing local path it is returned as a normalized
    absolute path; otherwise it is treated as a Hugging Face Hub repo id (e.g.
    ``microsoft/Mage-Flow``) and downloaded/cached via
    ``huggingface_hub.snapshot_download``.
    """
    candidate = os.path.realpath(repo_dir)
    if os.path.isdir(candidate):
        return candidate
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_dir)


def load_from_repo(repo_dir: str, device: str = "cuda") -> MageFlowModel:
    """Load a Mage-Flow diffusers-style repo (model_index.json + transformer/
    vae/ scheduler/). Transformer weights come from the bf16 safetensors;
    VAE + text encoder are built from the sources recorded in model_index.json.

    ``repo_dir`` may be a local directory OR a Hugging Face Hub repo id (e.g.
    ``microsoft/Mage-Flow-4B``), which is downloaded/cached automatically.
    """
    from safetensors.torch import load_file

    repo_dir = _resolve_repo_dir(repo_dir)
    mi = json.load(open(_safe_subpath(repo_dir, "model_index.json")))
    tcfg = json.load(open(_safe_subpath(repo_dir, "transformer", "config.json")))
    # Keys stripped from the checkpoint config before it becomes model_structure.
    # ``schedule_mode`` is a legacy field still present in some config.json files;
    # Keys of the checkpoint config that are NOT MageFlowParams constructor args
    # (legacy/unused fields). Everything else becomes model_structure. The DiT only
    # reads: in_channels, out_channels, context_in_dim, hidden_size, num_heads,
    # depth, axes_dim, checkpoint, patch_size.
    _meta = {
        "_class_name",
        "txt_max_length",
        "max_sequence_length",
        "param_dtype",
        "packing",
        "schedule_mode",
        "static_shift",
        "use_time_shift",
        "rope_type",
        "apply_text_rotary_emb",
        "mlp_ratio",
        "depth_single_blocks",
        "theta",
        "qkv_bias",
        "guidance_embed",
        "vec_in_dim",
        "vec_type",
        "time_type",
        "double_block_type",
    }
    structure = {k: v for k, v in tcfg.items() if k not in _meta}

    def _resolve(p):
        return p if os.path.isabs(p) else _safe_subpath(repo_dir, p)

    cfg = ModelConfig(
        vae_path=_resolve(mi.get("_vae_source")),
        txt_enc_path=_resolve(mi.get("_text_encoder_path")),
        model_structure=structure,
        txt_max_length=tcfg.get("txt_max_length", 2048),
        packing=tcfg.get("packing", True),
        static_shift=tcfg.get("static_shift", 6.0),
    )
    model = MageFlowModel(cfg)
    sd = load_file(_safe_subpath(repo_dir, "transformer", "diffusion_pytorch_model.safetensors"), device="cpu")
    model.transformer.load_state_dict(sd, strict=False, assign=True)
    model.to(device)
    model.transformer.to(torch.bfloat16)
    model.txt_enc.to(torch.bfloat16)
    if model.vae is not None:
        model.vae.to(torch.bfloat16)
    model.eval()
    # Diffusers FlowMatchEulerDiscreteScheduler (scheduler/scheduler_config.json).
    model.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(_safe_subpath(repo_dir, "scheduler"))
    return model
