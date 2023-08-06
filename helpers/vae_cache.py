import hashlib, os, torch, logging
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger('VAECache')
logger.setLevel('DEBUG')

class VAECache:
    def __init__(self, vae, accelerator, cache_dir="vae_cache", resolution: int = 1024):
        self.vae = vae
        self.vae.enable_slicing()
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        self.resolution = resolution
        os.makedirs(self.cache_dir, exist_ok=True)

    def create_hash(self, filename):
        # Create a sha256 hash
        sha256_hash = hashlib.sha256()

        # Feed the hash function with the filename
        sha256_hash.update(filename.encode())

        # Get the hexadecimal representation of the hash
        return sha256_hash.hexdigest()

    def save_to_cache(self, filename, embeddings):
        torch.save(embeddings, filename)

    def load_from_cache(self, filename):
        return torch.load(filename)

    def encode_image(self, pixel_values, file_hash: str):
        filename = os.path.join(self.cache_dir, file_hash + ".pt")
        if os.path.exists(filename):
            latents = self.load_from_cache(filename)
            logger.debug(f'Loading latents of shape {latents.shape} from existing cache file: {filename}')
        else:
            with torch.no_grad():
                latents = self.vae.encode(pixel_values.unsqueeze(0).to(self.accelerator.device, dtype=torch.bfloat16)).latent_dist.sample()
                logger.debug(f'Using shape {latents.shape}, creating new latent cache: {filename}')
            latents = latents * self.vae.config.scaling_factor
            logger.debug(f'Latent shape after re-scale: {latents.shape}')
            self.save_to_cache(filename, latents.squeeze())

        output_latents = latents.squeeze().to(self.accelerator.device, dtype=self.vae.dtype)
        logger.debug(f'Output latents shape: {output_latents.shape}')
        return output_latents

    def process_directory(self, directory):
        # Define a transform to convert the image to tensor
        transform = transforms.ToTensor()

        # Get a list of all the files to process (customize as needed)
        files_to_process = []
        logger.debug(f'Beginning processing of VAECache directory {directory}')
        for subdir, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    logger.debug(f'Discovered image: {os.path.join(subdir, file)}')
                    files_to_process.append(os.path.join(subdir, file))

        # Iterate through the files, displaying a progress bar
        for filepath in tqdm(files_to_process, desc='Processing images'):
            # Open the image using PIL
            try:
                logger.debug(f'Loading image: {filepath}')
                image = Image.open(filepath)
                image = image.convert('RGB')
                image = self._resize_for_condition_image(image, self.resolution)
                # Create a hash based on the filename
                file_hash = self.create_hash(filepath)
            except Exception as e:
                logger.error(f'Encountered error opening image: {e}')
                os.remove(filepath)
                continue

            # Convert the image to a tensor
            try:
                pixel_values = transform(image).to(self.accelerator.device, dtype=self.vae.dtype)
            except OSError as e:
                logger.error(f'Encountered error converting image to tensor: {e}')
                continue

            # Process the image with the VAE
            self.encode_image(pixel_values, file_hash)

            logger.debug(f'Processed image {filepath}')
    def _resize_for_condition_image(self, input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        aspect_ratio = round(W / H, 3)
        msg = f"Inspecting image of aspect {aspect_ratio} and size {W}x{H} to "
        if W < H:
            W = resolution
            H = int(resolution / aspect_ratio)  # Calculate the new height
        elif H < W:
            H = resolution
            W = int(resolution * aspect_ratio)  # Calculate the new width
        if W == H:
            W = resolution
            H = resolution
        msg = f"{msg} {W}x{H}."
        logger.debug(msg)
        img = input_image.resize((W, H), resample=Image.BICUBIC)
        return img
