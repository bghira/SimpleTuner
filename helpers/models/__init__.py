upstream_config_sources = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "kolors": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd3": "stabilityai/stable-diffusion-3-large",
    "sana": "terminusresearch/sana-1.6b-1024px",
    "flux": "black-forest-labs/flux.1-dev",
    "legacy": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}


def get_model_config_path(model_family: str, model_path: str):
    print(f"model_path: {model_path}")
    if model_path.endswith(".safetensors"):
        print(f"found safetensors file: {model_path}")
        if model_family in upstream_config_sources:
            print(
                f"found model family: {model_family} ({upstream_config_sources[model_family]})"
            )
            return upstream_config_sources[model_family]
        else:
            raise ValueError(
                "Cannot find noise schedule config for .safetensors file in architecture {}".format(
                    model_family
                )
            )

    print(f"No safetensors file found: {model_path}")
    return model_path
