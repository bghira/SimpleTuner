import torch, logging
from diffusers import AutoencoderKL
from helpers.vae_cache import VAECache

logger = logging.getLogger("VAEManager")
logger.setLevel("DEBUG")


class VAEManager:
    def __init__(self, vae_path, subfolder, revision, force_upcast, device):
        self.vae_path = vae_path
        self.subfolder = subfolder
        self.revision = revision
        self.force_upcast = force_upcast
        self.device = device

    def initialise_vae(self, args):
        vae = AutoencoderKL.from_pretrained(
            self.vae_path,
            subfolder=self.subfolder
            if args.pretrained_vae_model_name_or_path is None
            else None,
            revision=self.revision,
            force_upcast=self.force_upcast,
        )
        vae.enable_slicing()
        return vae

    def set_vae_dtype(self, args, vae):
        vae_dtype = torch.float32
        if hasattr(args, "vae_dtype"):
            logger.info(f"Configured VAE type to {args.vae_dtype}")
            if args.vae_dtype == "bf16":
                vae_dtype = torch.bfloat16
            elif args.vae_dtype == "fp16":
                vae_dtype = torch.float16
            elif args.vae_dtype == "fp32":
                vae_dtype = torch.float32
            elif args.vae_dtype == "none" or args.vae_dtype == "default":
                vae_dtype = torch.float32
            else:
                raise ValueError(f"Unknown dtype: `--vae_dtype {args.vae_dtype}`")
        logger.info(f"Moving VAE to GPU {self.gpu} with dtype {vae_dtype}")
        if args.pretrained_vae_model_name_or_path is not None:
            vae.to(self.device, dtype=vae_dtype)
        else:
            vae.to(self.device, dtype=vae_dtype)

        return vae

    def process_directory(self, vae, accelerator, args):
        if accelerator.is_main_process:
            logger.info(f"Pre-computing VAE latent space.")
            vaecache = VAECache(vae, accelerator)
            vaecache.process_directory(args.instance_data_dir)
