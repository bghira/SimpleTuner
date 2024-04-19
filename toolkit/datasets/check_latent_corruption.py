"""
2024-04-05 17:19:44,198 [DEBUG] (LocalDataBackend) Checking if /Volumes/models/training/vae_cache/sdxl/photo-concept-bucket/image_data/1027365.pt exists = True
2024-04-05 17:19:44,198 [DEBUG] (LocalDataBackend) Checking if /Volumes/models/training/vae_cache/sdxl/photo-concept-bucket/image_data/10064767.pt exists = True
2024-04-05 17:19:44,223 [DEBUG] (LocalDataBackend) Checking if /Volumes/models/training/vae_cache/sdxl/photo-concept-bucket/image_data/13997787.pt exists = True
2024-04-05 17:19:44,223 [DEBUG] (LocalDataBackend) Checking if /Volumes/models/training/vae_cache/sdxl/photo-concept-bucket/image_data/13565183.pt exists = True
"""

latent_file_paths = ["1027365", "10064767", "13997787", "13565183"]

prefix = "/Volumes/models/training/vae_cache/sdxl/photo-concept-bucket/image_data/"

# load the latent_file_paths
import torch

for latent_file_path in latent_file_paths:
    print(f"{prefix}{latent_file_path}.pt")
    latent = torch.load(
        f"{prefix}{latent_file_path}.pt", map_location=torch.device("cpu")
    )
    print(f"Shape: {latent.shape}")
    print(f"Mean: {latent.mean()}")
    print(f"Std: {latent.std()}")
    print(f"Min: {latent.min()}")
    print(f"Is corrupt: {torch.isnan(latent).any() or torch.isinf(latent).any()}")
