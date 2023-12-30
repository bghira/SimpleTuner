from helpers.data_backend.local import LocalDataBackend
from helpers.data_backend.aws import S3DataBackend

from helpers.training.exceptions import MultiDatasetExhausted
from helpers.multiaspect.bucket import BucketManager
from helpers.multiaspect.dataset import MultiAspectDataset
from helpers.multiaspect.sampler import MultiAspectSampler
from helpers.prompts import PromptHandler
from helpers.caching.vae import VAECache
from helpers.training.multi_process import rank_info
from helpers.training.collate import collate_fn
from helpers.training.state_tracker import StateTracker

import json, os, torch, logging, random

logger = logging.getLogger("DataBackendFactory")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


def init_backend_config(backend: dict, args: dict, accelerator) -> dict:
    output = {"id": backend["id"], "config": {}}
    if "crop" in backend:
        output["config"]["crop"] = backend["crop"]
    else:
        output["config"]["crop"] = args.crop
    if "crop_aspect" in backend:
        output["config"]["crop_aspect"] = backend["crop_aspect"]
    else:
        output["config"]["crop_aspect"] = args.crop_aspect
    if "crop_style" in backend:
        output["config"]["crop_style"] = backend["crop_style"]
    else:
        output["config"]["crop_style"] = args.crop_style
    if "resolution" in backend:
        output["config"]["resolution"] = backend["resolution"]
    else:
        output["config"]["resolution"] = args.resolution
    if "resolution_type" in backend:
        output["config"]["resolution_type"] = backend["resolution_type"]
    else:
        output["config"]["resolution_type"] = args.resolution_type

    return output


def print_bucket_info(bucket_manager):
    # Print table header
    print(f"{rank_info()} | {'Bucket':<10} | {'Image Count':<12}")

    # Print separator
    print("-" * 30)

    # Print each bucket's information
    for bucket in bucket_manager.aspect_ratio_bucket_indices:
        image_count = len(bucket_manager.aspect_ratio_bucket_indices[bucket])
        print(f"{rank_info()} | {bucket:<10} | {image_count:<12}")


def configure_multi_databackend(args: dict, accelerator):
    """
    Configure a multiple dataloaders based on the provided commandline args.
    """
    if args.data_backend_config is None:
        raise ValueError(
            "Must provide a data backend config file via --data_backend_config"
        )
    if not os.path.exists(args.data_backend_config):
        raise FileNotFoundError(
            f"Data backend config file {args.data_backend_config} not found."
        )
    with open(args.data_backend_config, "r") as f:
        data_backend_config = json.load(f)
    if len(data_backend_config) == 0:
        raise ValueError(
            "Must provide at least one data backend in the data backend config file."
        )
    all_captions = []
    for backend in data_backend_config:
        if "disabled" in backend and backend["disabled"]:
            logger.info(
                f"Skipping disabled data backend {backend['id']} in config file."
            )
            continue
        # For each backend, we will create a dict to store all of its components in.
        if "id" not in backend:
            raise ValueError(
                "No identifier was given for one more of your data backends. Add a unique 'id' field to each one."
            )
        # Retrieve some config file overrides for commandline arguments, eg. cropping
        init_backend = init_backend_config(backend, args, accelerator)
        if backend["type"] == "local":
            init_backend["data_backend"] = get_local_backend(
                accelerator, init_backend["id"]
            )
            init_backend["instance_data_root"] = backend["instance_data_dir"]
        elif backend["type"] == "aws":
            check_aws_config(backend)
            init_backend["data_backend"] = get_aws_backend(
                identifier=init_backend["id"],
                aws_bucket_name=backend["aws_bucket_name"],
                aws_region_name=backend["aws_region_name"],
                aws_endpoint_url=backend["aws_endpoint_url"],
                aws_access_key_id=backend["aws_access_key_id"],
                aws_secret_access_key=backend["aws_secret_access_key"],
                accelerator=accelerator,
            )
            # S3 buckets use the aws_data_prefix as their prefix/ for all data.
            init_backend["instance_data_root"] = backend["aws_data_prefix"]
        else:
            raise ValueError(f"Unknown data backend type: {backend['type']}")

        init_backend["bucket_manager"] = BucketManager(
            id=init_backend["id"],
            instance_data_root=init_backend["instance_data_root"],
            data_backend=init_backend["data_backend"],
            accelerator=accelerator,
            resolution=backend.get("resolution", args.resolution),
            minimum_image_size=backend.get(
                "minimum_image_size", args.minimum_image_size
            ),
            resolution_type=backend.get("resolution_type", args.resolution_type),
            batch_size=args.train_batch_size,
            metadata_update_interval=backend.get(
                "metadata_update_interval", args.metadata_update_interval
            ),
            cache_file=os.path.join(
                init_backend["instance_data_root"], "aspect_ratio_bucket_indices.json"
            ),
            metadata_file=os.path.join(
                init_backend["instance_data_root"], "aspect_ratio_bucket_metadata.json"
            ),
            delete_problematic_images=args.delete_problematic_images or False,
        )
        prev_config = {}
        if hasattr(init_backend["bucket_manager"], "config"):
            prev_config = init_backend["bucket_manager"].config
        logger.debug(f"Loaded previous data backend config: {prev_config}")
        StateTracker.set_data_backend_config(
            data_backend_id=init_backend["id"],
            config=prev_config,
        )
        if init_backend["bucket_manager"].has_single_underfilled_bucket():
            raise Exception(
                f"Cannot train using a dataset that has a single bucket with fewer than {args.train_batch_size} images."
                f" You have to reduce your batch size, or increase your dataset size (id={init_backend['id']})."
            )
        if "aspect" not in args.skip_file_discovery:
            if accelerator.is_local_main_process:
                init_backend["bucket_manager"].refresh_buckets(rank_info())
        accelerator.wait_for_everyone()
        init_backend["bucket_manager"].reload_cache()
        # Now split the contents of these buckets between all processes
        init_backend["bucket_manager"].split_buckets_between_processes(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        # Check if there is an existing 'config' in the bucket_manager.config
        if prev_config != {}:
            logger.debug(f"Found existing config: {prev_config}")
            # Check if any values differ between the 'backend' values and the 'config' values:
            for key, _ in prev_config.items():
                logger.debug(f"Checking config key: {key}")
                if key in backend and prev_config[key] != backend[key]:
                    if not args.override_dataset_config:
                        raise Exception(
                            f"Dataset {init_backend['id']} has inconsistent config, and --override_dataset_config was not provided."
                            f"\n-> Expected value {key}={prev_config[key]} differs from current value={backend[key]}."
                            f"\n-> Recommended action is to correct the current config values to match the values that were used to create this dataset:"
                            f"\n{prev_config}"
                        )
                    else:
                        logger.warning(
                            f"Overriding config value {key}={prev_config[key]} with {backend[key]}"
                        )
                        prev_config[key] = backend[key]

        print_bucket_info(init_backend["bucket_manager"])
        if len(init_backend["bucket_manager"]) == 0:
            raise Exception(
                "No images were discovered by the bucket manager in the dataset."
            )

        use_captions = True
        if "only_instance_prompt" in backend and backend["only_instance_prompt"]:
            use_captions = False
        elif args.only_instance_prompt:
            use_captions = False
        init_backend["train_dataset"] = MultiAspectDataset(
            id=init_backend["id"],
            datasets=[init_backend["bucket_manager"]],
        )

        # full filename path:
        seen_state_path = args.seen_state_path
        # split the filename by extension, append init_backend["id"] to the end of the filename, reassemble with extension:
        seen_state_path = ".".join(
            seen_state_path.split(".")[:-1]
            + [init_backend["id"], seen_state_path.split(".")[-1]]
        )
        state_path = args.state_path
        state_path = ".".join(
            state_path.split(".")[:-1] + [init_backend["id"], state_path.split(".")[-1]]
        )

        init_backend["sampler"] = MultiAspectSampler(
            id=init_backend["id"],
            bucket_manager=init_backend["bucket_manager"],
            data_backend=init_backend["data_backend"],
            accelerator=accelerator,
            batch_size=args.train_batch_size,
            seen_images_path=backend.get("seen_state_path", seen_state_path),
            state_path=backend.get("state_path", state_path),
            debug_aspect_buckets=args.debug_aspect_buckets,
            delete_unwanted_images=backend.get(
                "delete_unwanted_images", args.delete_unwanted_images
            ),
            resolution=backend.get("resolution", args.resolution),
            resolution_type=backend.get("resolution_type", args.resolution_type),
            caption_strategy=backend.get("caption_strategy", args.caption_strategy),
            use_captions=use_captions,
            prepend_instance_prompt=backend.get(
                "prepend_instance_prompt", args.prepend_instance_prompt
            ),
        )

        init_backend["train_dataloader"] = torch.utils.data.DataLoader(
            init_backend["train_dataset"],
            batch_size=1,  # The sampler handles batching
            shuffle=False,  # The sampler handles shuffling
            sampler=init_backend["sampler"],
            collate_fn=lambda examples: collate_fn(examples),
            num_workers=0,
            persistent_workers=False,
        )

        with accelerator.main_process_first():
            all_captions.extend(
                PromptHandler.get_all_captions(
                    data_backend=init_backend["data_backend"],
                    instance_data_root=init_backend["instance_data_root"],
                    prepend_instance_prompt=backend.get(
                        "prepend_instance_prompt", args.prepend_instance_prompt
                    ),
                    use_captions=use_captions,
                )
            )

        logger.info(f"Pre-computing VAE latent space.")
        init_backend["vaecache"] = VAECache(
            id=init_backend["id"],
            vae=StateTracker.get_vae(),
            accelerator=accelerator,
            bucket_manager=init_backend["bucket_manager"],
            data_backend=init_backend["data_backend"],
            instance_data_root=init_backend["instance_data_root"],
            delete_problematic_images=backend.get(
                "delete_problematic_images", args.delete_problematic_images
            ),
            resolution=backend.get("resolution", args.resolution),
            resolution_type=backend.get("resolution_type", args.resolution_type),
            minimum_image_size=backend.get(
                "minimum_image_size", args.minimum_image_size
            ),
            vae_batch_size=args.vae_batch_size,
            write_batch_size=args.write_batch_size,
            cache_dir=backend.get("cache_dir_vae", args.cache_dir_vae),
        )

        if accelerator.is_local_main_process:
            init_backend["vaecache"].discover_all_files()
        accelerator.wait_for_everyone()

        if (
            "metadata" not in args.skip_file_discovery
            and accelerator.is_main_process
            and backend.get("scan_for_errors", False)
        ):
            logger.info(
                f"Beginning error scan for dataset {init_backend['id']}. Set 'scan_for_errors' to False in the dataset config to disable this."
            )
            init_backend["bucket_manager"].handle_vae_cache_inconsistencies(
                vae_cache=init_backend["vaecache"],
                vae_cache_behavior=backend.get(
                    "vae_cache_behaviour", args.vae_cache_behaviour
                ),
            )
            init_backend["bucket_manager"].scan_for_metadata()
        elif not backend.get("scan_for_errors", False):
            logger.info(
                f"Skipping error scan for dataset {init_backend['id']}. Set 'scan_for_errors' to True in the dataset config to enable this if your training runs into mismatched latent dimensions."
            )
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            init_backend["bucket_manager"].load_image_metadata()
        accelerator.wait_for_everyone()

        if "vae" not in args.skip_file_discovery:
            init_backend["vaecache"].split_cache_between_processes()
            init_backend["vaecache"].process_buckets()
            accelerator.wait_for_everyone()

        StateTracker.register_data_backend(init_backend)
        init_backend["bucket_manager"].save_cache()

    # After configuring all backends, register their captions.
    StateTracker.set_caption_files(all_captions)

    if len(StateTracker.get_data_backends()) == 0:
        raise ValueError(
            "Must provide at least one data backend in the data backend config file."
        )
    return StateTracker.get_data_backends()


def get_local_backend(accelerator, identifier: str) -> LocalDataBackend:
    """
    Get a local disk backend.

    Args:
        accelerator (Accelerator): A Huggingface Accelerate object.
        identifier (str): An identifier that links this data backend to its other components.
    Returns:
        LocalDataBackend: A LocalDataBackend object.
    """
    return LocalDataBackend(accelerator=accelerator, id=identifier)


def check_aws_config(backend: dict) -> None:
    """
    Check the configuration for an AWS backend.

    Args:
        backend (dict): A dictionary of the backend configuration.
    Returns:
        None
    """
    required_keys = [
        "aws_bucket_name",
        "aws_region_name",
        "aws_endpoint_url",
        "aws_access_key_id",
        "aws_secret_access_key",
    ]
    for key in required_keys:
        if key not in backend:
            raise ValueError(f"Missing required key {key} in AWS backend config.")


def get_aws_backend(
    aws_bucket_name: str,
    aws_region_name: str,
    aws_endpoint_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    accelerator,
    identifier: str,
) -> S3DataBackend:
    return S3DataBackend(
        id=identifier,
        bucket_name=aws_bucket_name,
        accelerator=accelerator,
        region_name=aws_region_name,
        endpoint_url=aws_endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def get_dataset(args: dict, accelerator) -> list:
    """Retrieve a dataset based on the provided commandline args.

    Args:
        args (dict): A dictionary from parseargs.
        accelerator (Accelerator): A Huggingface Accelerate object.
    Returns:
        list: A list of DataBackend objects.
    """
    if args.data_backend == "multi":
        return configure_multi_databackend(args)
    elif args.data_backend == "local":
        if not os.path.exists(args.instance_data_dir):
            raise FileNotFoundError(
                f"Instance {args.instance_data_root} images root doesn't exist. Cannot continue."
            )
        return [get_local_backend(args, accelerator)]
    elif args.data_backend == "aws":
        return [
            get_aws_backend(
                aws_bucket_name=args.aws_bucket_name,
                aws_region_name=args.aws_region_name,
                aws_endpoint_url=args.aws_endpoint_url,
                aws_access_key_id=args.aws_access_key_id,
                aws_secret_access_key=args.aws_secret_access_key,
                accelerator=accelerator,
            )
        ]


step = None


def random_dataloader_iterator(dataloaders):
    global step
    data_backends = StateTracker.get_data_backends()

    # Remove any 'dataloaders' who have been exhausted.
    for backend_id, backend in data_backends.items():
        if StateTracker.backend_status(backend_id):
            logger.info(
                f"Dataset (name={backend_id}) was detected as exhausted from a previous run."
                " Removing from list, it will not be sampled for the remainder of this epoch."
            )
            # Remove corresponding 'dataloaders' entry:
            for i, dataloader in enumerate(dataloaders):
                if dataloader.dataset.id == backend_id:
                    dataloaders.pop(i)
                    break

    iterator_indices = list(range(len(dataloaders)))
    iterators = [iter(dataloader) for dataloader in dataloaders]

    initial_probabilities = [
        backend["config"].get("probability", 1) for _, backend in data_backends.items()
    ]
    disable_steps = [
        backend["config"].get("disable_after_epoch_step", float("inf"))
        for _, backend in data_backends.items()
    ]

    if step is None:
        step = StateTracker.get_epoch_step()
    else:
        step = 0

    gradient_accumulation_steps = StateTracker.get_args().gradient_accumulation_steps

    while iterators:
        step += 1
        epoch_step = int(step / gradient_accumulation_steps)
        StateTracker.set_epoch_step(epoch_step)

        chosen_index = select_dataloader_index(
            step, iterator_indices, initial_probabilities, disable_steps
        )

        if chosen_index is None:
            logger.info("No dataloader iterators were available.")
            break

        chosen_iter = iterators[iterator_indices.index(chosen_index)]
        backend_id = list(data_backends)[chosen_index]

        if StateTracker.backend_status(backend_id):
            logger.info(
                f"Dataset (name={backend_id}) was detected as exhausted from a previous run."
                " Removing from list, it will not be sampled for the remainder of this epoch."
            )
            remove_index = iterator_indices.index(chosen_index)
            iterator_indices.pop(remove_index)
            iterators.pop(remove_index)
            initial_probabilities.pop(remove_index)
            disable_steps.pop(remove_index)
            continue

        try:
            yield (step, next(chosen_iter))
        except MultiDatasetExhausted:
            logger.info(
                f"Dataset (name={backend_id}) is now exhausted. Removing from list."
            )
            remove_index = iterator_indices.index(chosen_index)
            iterator_indices.pop(remove_index)
            iterators.pop(remove_index)
            initial_probabilities.pop(remove_index)
            disable_steps.pop(remove_index)
            StateTracker.backend_exhausted(backend_id)
            if not iterator_indices:
                logger.info(
                    "All dataloaders exhausted. Moving to next epoch in main training loop."
                )
                for backend_id in data_backends:
                    StateTracker.backend_enable(backend_id)
                return None


def select_dataloader_index(
    step, iterator_indices, initial_probabilities, disable_steps
):
    adjusted_probabilities = []
    for i in iterator_indices:
        prob, disable_step = initial_probabilities[i], disable_steps[i]
        adjusted_prob = (
            0 if step > disable_step else max(0, prob * (1 - step / disable_step))
        )
        adjusted_probabilities.append(adjusted_prob)

    total_prob = sum(adjusted_probabilities)
    if total_prob == 0:
        return None

    rnd = random.uniform(0, total_prob)
    cumulative_prob = 0
    for i, prob in enumerate(adjusted_probabilities):
        cumulative_prob += prob
        if rnd < cumulative_prob:
            return iterator_indices[i]

    return None
