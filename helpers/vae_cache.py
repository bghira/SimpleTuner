import hashlib, os, torch, logging
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger('VAECache')
logger.setLevel('INFO')

class VAECache:
    def __init__(self, vae, accelerator, cache_dir="vae_cache"):
        self.vae = vae
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def create_hash(self, image):
        return hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest()

    def save_to_cache(self, filename, embeddings):
        torch.save(embeddings, filename)

    def load_from_cache(self, filename):
        return torch.load(filename)

    def encode_image(self, pixel_values):
        image_hash = self.create_hash(pixel_values)
        filename = os.path.join(self.cache_dir, image_hash + ".pt")

        if os.path.exists(filename):
            latents = self.load_from_cache(filename)
        else:
            with torch.no_grad():
                latents = self.vae.encode(pixel_values.unsqueeze(0)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            self.save_to_cache(filename, latents.squeeze())

        return latents.squeeze()

    def process_directory(self, directory):
        # Define a transform to convert the image to tensor
        transform = transforms.ToTensor()

        # Get a list of all the files to process (customize as needed)
        files_to_process = []
        for subdir, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    files_to_process.append(os.path.join(subdir, file))

        # Iterate through the files, displaying a progress bar
        for filepath in tqdm(files_to_process, desc='Processing images'):
            # Open the image using PIL
            try:
                image = Image.open(filepath)
            except Exception as e:
                logger.error(f'Encountered error opening image: {e}')
                os.remove(filepath)
                continue

            # Convert the image to a tensor
            try:
                pixel_values = transform(image).float().to(self.accelerator.device)
            except OSError as e:
                logger.error(f'Encountered error converting image to tensor: {e}')
                continue

            # Process the image with the VAE
            self.encode_image(pixel_values)

            logger.debug(f'Processed image {filepath}')
