from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt2EffectFamilySpec:
    family: str
    label: str
    component_prefix: str
    component_subfolder: str
    default_target_modules: tuple[str, ...]
    default_model_flavour: str
    known_model_repos: dict[str, str]
    i2v_flavours: tuple[str, ...] = ()


FAMILY_SPECS: dict[str, Prompt2EffectFamilySpec] = {
    "ltxvideo2": Prompt2EffectFamilySpec(
        family="ltxvideo2",
        label="LTXVideo2",
        component_prefix="transformer",
        component_subfolder="transformer",
        default_target_modules=("to_k", "to_q", "to_v", "to_out.0"),
        default_model_flavour="dev",
        known_model_repos={
            "dev": "Lightricks/LTX-2",
            "dev-fp4": "Lightricks/LTX-2",
            "dev-fp8": "Lightricks/LTX-2",
            "2.3-dev": "dg845/LTX-2.3-Diffusers",
            "2.3-distilled": "dg845/LTX-2.3-Distilled-Diffusers",
        },
    ),
    "wan": Prompt2EffectFamilySpec(
        family="wan",
        label="Wan",
        component_prefix="transformer",
        component_subfolder="transformer",
        default_target_modules=("to_k", "to_q", "to_v", "to_out.0"),
        default_model_flavour="i2v-14b-2.1",
        known_model_repos={
            "i2v-14b-2.1": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            "i2v-14b-2.1-720p": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            "i2v-14b-2.2-high": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "i2v-14b-2.2-low": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        },
        i2v_flavours=("i2v-14b-2.1", "i2v-14b-2.1-720p", "i2v-14b-2.2-high", "i2v-14b-2.2-low"),
    ),
    "hunyuanvideo": Prompt2EffectFamilySpec(
        family="hunyuanvideo",
        label="HunyuanVideo",
        component_prefix="transformer",
        component_subfolder="transformer",
        default_target_modules=(
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ),
        default_model_flavour="i2v-480p",
        known_model_repos={
            "t2v-480p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
            "t2v-720p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
            "t2v-480p-distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",
            "i2v-480p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
            "i2v-720p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
            "i2v-480p-distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled",
            "i2v-720p-distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled",
        },
        i2v_flavours=("i2v-480p", "i2v-720p", "i2v-480p-distilled", "i2v-720p-distilled"),
    ),
}

ALIASES = {
    "ltx2": "ltxvideo2",
    "ltx-video2": "ltxvideo2",
    "ltx_video2": "ltxvideo2",
    "wan_i2v": "wan",
    "wan-i2v": "wan",
    "hunyuan": "hunyuanvideo",
    "hunyuan_video": "hunyuanvideo",
    "hunyuan-video": "hunyuanvideo",
}


def resolve_family_spec(model_family: str) -> Prompt2EffectFamilySpec:
    normalized = str(model_family).strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    try:
        return FAMILY_SPECS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(FAMILY_SPECS))
        raise ValueError(
            f"Unsupported Prompt2Effect model family `{model_family}`. Supported families: {supported}."
        ) from exc


def resolve_model_repo(spec: Prompt2EffectFamilySpec, model_ref: str | None, model_flavour: str | None) -> str:
    if model_ref not in (None, "", "None"):
        return str(model_ref)
    flavour = str(model_flavour or spec.default_model_flavour)
    try:
        return spec.known_model_repos[flavour]
    except KeyError as exc:
        supported = ", ".join(sorted(spec.known_model_repos))
        raise ValueError(
            f"Unknown {spec.label} model flavour `{flavour}`. Use --base_model or one of: {supported}."
        ) from exc


def normalize_target_modules(raw: str | None, spec: Prompt2EffectFamilySpec) -> list[str]:
    if raw in (None, "", "None", "default"):
        return list(spec.default_target_modules)
    if str(raw).strip() == "all-linear":
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def module_matches_target(module_name: str, target_modules: list[str]) -> bool:
    if not target_modules:
        return True
    return any(module_name == target or module_name.endswith(f".{target}") for target in target_modules)


def module_type_for_name(module_name: str, target_modules: list[str]) -> str:
    for target in sorted(target_modules, key=len, reverse=True):
        if module_name == target or module_name.endswith(f".{target}"):
            return target
    return module_name.rsplit(".", 1)[-1]
