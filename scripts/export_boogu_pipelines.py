#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import save_file
from torch.serialization import safe_globals
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict
from torchao.quantization import Float8Tensor


REPOS = {
    "v0.1-base": ("Boogu/Boogu-Image-0.1-Base", "SimpleTuner/Boogu-Image-0.1-Base"),
    "v0.1-base-fp8": ("Boogu/Boogu-Image-0.1-Base-fp8", "SimpleTuner/Boogu-Image-0.1-Base-fp8"),
    "v0.1-turbo": ("Boogu/Boogu-Image-0.1-Turbo", "SimpleTuner/Boogu-Image-0.1-Turbo"),
    "v0.1-turbo-fp8": ("Boogu/Boogu-Image-0.1-Turbo-fp8", "SimpleTuner/Boogu-Image-0.1-Turbo-fp8"),
    "v0.1-edit": ("Boogu/Boogu-Image-0.1-Edit", "SimpleTuner/Boogu-Image-0.1-Edit"),
    "v0.1-edit-fp8": ("Boogu/Boogu-Image-0.1-Edit-fp8", "SimpleTuner/Boogu-Image-0.1-Edit-fp8"),
}


def copy_snapshot(src_repo: str, export_dir: Path) -> Path:
    snapshot_path = Path(snapshot_download(src_repo))
    target_dir = export_dir / src_repo.replace("/", "__")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(snapshot_path, target_dir, symlinks=False)
    return target_dir


def convert_transformer_bins_to_safetensors(repo_dir: Path) -> None:
    transformer_dir = repo_dir / "transformer"
    index_path = transformer_dir / "diffusion_pytorch_model.bin.index.json"
    if not index_path.exists():
        return

    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    bin_filenames = sorted(set(weight_map.values()))
    filename_map = {
        filename: filename.replace(".bin", ".safetensors")
        for filename in bin_filenames
        if filename.endswith(".bin")
    }

    for bin_name, safetensors_name in filename_map.items():
        bin_path = transformer_dir / bin_name
        safetensors_path = transformer_dir / safetensors_name
        print(f"Converting {bin_path.name} -> {safetensors_path.name}", flush=True)
        with safe_globals([Float8Tensor]):
            state_dict = torch.load(bin_path, map_location="cpu", mmap=True)
        state_dict, metadata = flatten_tensor_state_dict(state_dict)
        metadata["format"] = "pt"
        save_file(state_dict, safetensors_path, metadata=metadata)
        del state_dict
        bin_path.unlink()

    index["weight_map"] = {key: filename_map.get(value, value) for key, value in weight_map.items()}
    new_index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    new_index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    index_path.unlink()


def upload_repo(api: HfApi, local_dir: Path, dst_repo: str, private: bool, dry_run: bool) -> None:
    print(f"Uploading {local_dir} -> {dst_repo}", flush=True)
    if dry_run:
        return
    api.create_repo(repo_id=dst_repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=dst_repo,
        repo_type="model",
        folder_path=str(local_dir),
        commit_message="Export Boogu-Image pipeline with safetensors weights",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="/tmp/simpletuner-boogu-export")
    parser.add_argument("--flavour", action="append", choices=sorted(REPOS), help="Limit export to one or more flavours.")
    parser.add_argument("--private", action="store_true", help="Create target repositories as private.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    selected = args.flavour or list(REPOS)
    api = HfApi()

    for flavour in selected:
        src_repo, dst_repo = REPOS[flavour]
        print(f"Preparing {flavour}: {src_repo} -> {dst_repo}", flush=True)
        local_dir = copy_snapshot(src_repo, export_dir)
        convert_transformer_bins_to_safetensors(local_dir)
        upload_repo(api, local_dir, dst_repo, args.private, args.dry_run)


if __name__ == "__main__":
    main()
