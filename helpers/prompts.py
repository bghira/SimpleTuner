import json
import regex as re
from pathlib import Path
from helpers.training.state_tracker import StateTracker

prompts = {
    "alien_landscape": "Alien planet, strange rock formations, glowing plants, bizarre creatures, surreal atmosphere",
    "alien_market": "Alien marketplace, bizarre creatures, exotic goods, vibrant colors, otherworldly atmosphere",
    "child_balloon": "Child holding a balloon, happy expression, colorful balloons, sunny day, high detail",
    "comic_strip": "a 4-panel comic strip showing an orange cat saying the words 'HELP' and 'LASAGNA'",
    "comic_book": "a hand is holding a comic book with a cover that reads 'The Adventures of Superhero'",
    "crystal_cave": "Underground cave filled with crystals, glowing lights, reflective surfaces, fantasy environment, high detail",
    "cyberpunk_bazaar": "Bustling cyberpunk bazaar, vendors, neon signs, advanced tech, crowded, high detail",
    "cyberpunk_hacker": "Cyberpunk hacker in a dark room, neon glow, multiple screens, intense focus, high detail",
    "cybernetic_anne": "a cybernetic anne of green gables with neural implant and bio mech augmentations",
    "dystopian_city": "Post-apocalyptic cityscape, ruined buildings, overgrown vegetation, dark and gritty, high detail",
    "enchanted_castle": "Magical castle in a lush forest, glowing windows, fantasy architecture, high resolution, detailed textures",
    "enchanted_forest_ruins": "Ruins of an ancient temple in an enchanted forest, glowing runes, mystical creatures, high detail",
    "enchanted_forest": "Mystical forest, glowing plants, fairies, magical creatures, fantasy art, high detail",
    "enchanted_garden": "Magical garden with glowing flowers, fairies, serene atmosphere, detailed plants, high resolution",
    "fairy_garden": "Whimsical garden filled with fairies, magical plants, sparkling lights, serene atmosphere, high detail",
    "fantasy_dragon": "Majestic dragon soaring through the sky, detailed scales, dynamic pose, fantasy art, high resolution",
    "floating_islands": "Fantasy world, floating islands in the sky, waterfalls, lush vegetation, detailed landscape, high resolution",
    "futuristic_cityscape": "Futuristic city skyline at night, neon lights, cyberpunk style, high contrast, sharp focus",
    "galactic_battle": "Space battle scene, starships fighting, laser beams, explosions, cosmic background",
    "haunted_fairground": "Abandoned fairground at night, eerie rides, ghostly figures, fog, dark atmosphere, high detail",
    "haunted_mansion": "Spooky haunted mansion on a hill, dark and eerie, glowing windows, ghostly atmosphere, high detail",
    "hardcover_textbook": "a hardcover physics textbook that is called PHYSICS FOR DUMMIES",
    "medieval_battle": "Epic medieval battle, knights in armor, dynamic action, detailed landscape, high resolution",
    "medieval_market": "Bustling medieval market with merchants, knights, and jesters, vibrant colors, detailed",
    "medieval_tavern": "Cozy medieval tavern, warm firelight, adventurers drinking, detailed interior, rustic atmosphere",
    "neon_cityscape": "Futuristic city skyline at night, neon lights, cyberpunk style, high contrast, sharp focus",
    "neon_forest": "Forest with neon-lit trees, glowing plants, bioluminescence, surreal atmosphere, high detail",
    "neon_sign": "Bright neon sign in a busy city street, 'Open 24 Hours', bold typography, glowing lights",
    "neon_typography": "Vibrant neon sign, 'Bar', bold typography, dark background, glowing lights, detailed design",
    "pirate_ship": "Pirate ship on the high seas, stormy weather, detailed sails, dramatic waves, photorealistic",
    "pirate_treasure": "Pirate discovering a treasure chest, detailed gold coins, tropical island, dramatic lighting",
    "psychedelic": "a photograph of a woman experiencing a psychedelic trip. trippy, 8k, uhd, fractal",
    "rainy_cafe": "Cozy cafe on a rainy day, people sipping coffee, warm lights, reflections on wet pavement, photorealistic",
    "retro_arcade": "1980s arcade, neon lights, vintage game machines, kids playing, vibrant colors, nostalgic atmosphere",
    "retro_game_room": "1980s game room with vintage arcade machines, neon lights, vibrant colors, nostalgic feel",
    "robot_blacksmith": "Robot blacksmith forging metal, sparks flying, detailed workshop, futuristic and medieval blend",
    "robot_dancer": "Sleek robot performing a dance, futuristic theater, holographic effects, detailed, high resolution",
    "robot_factory": "High-tech factory where robots are assembled, detailed machinery, futuristic setting, high detail",
    "robotic_garden": "Garden tended by robots, mechanical plants, colorful flowers, futuristic setting, high detail",
    "robotic_pet": "Cute robotic pet, futuristic home, sleek design, detailed features, friendly and animated",
    "security_footage": "cctv trail camera night time security picture of a wendigo in the woods",
    "space_explorer": "Astronaut exploring an alien planet, detailed landscape, futuristic suit, cosmic background",
    "space_station": "Futuristic space station orbiting a distant exoplanet, sleek design, detailed structures, cosmic backdrop",
    "soon": "a person holding a sign that reads 'SOON'",
    "steampunk_airship": "Steampunk airship in the sky, intricate design, Victorian aesthetics, dynamic scene, high detail",
    "steampunk_inventor": "Steampunk inventor in a workshop, intricate gadgets, Victorian attire, mechanical arm, goggles",
    "stormy_ocean": "Stormy ocean with towering waves, dramatic skies, detailed water, intense atmosphere, high resolution",
    "stormy_sea": "Dramatic stormy sea, lighthouse in the distance, lightning striking, dark clouds, high detail",
    "urban_art": "Graffiti artist creating a mural, vibrant colors, urban setting, dynamic action, high resolution",
    "urban_graffiti": "Urban alleyway filled with vibrant graffiti art, tags and murals, realistic textures",
    "urban_street_sign": "Urban street sign, 'Main Street', bold typography, realistic textures, weathered look",
    "vintage_car_show": "Classic car show with vintage vehicles, vibrant colors, nostalgic atmosphere, high detail",
    "vintage_diner_sign": "Retro diner sign, 'Joe's Diner', classic 1950s design, neon lights, weathered look",
    "vintage_store_sign": "Vintage store sign with elaborate typography, 'Antique Shop', hand-painted, weathered look",
}


def prompt_library_injection(new_prompts: dict) -> dict:
    """
    Add more prompts to the built-in SimpleTuner Prompt library.

    Args:
        new_prompts (dict): A dict of shortnames matching the existing prompt library format:
        {
            "nickname_here": "prompt goes here",
            ...
        }

    Returns:
        dict: Completed prompt library.
    """

    # Unpack the new prompts into the library.
    global prompts
    return {**prompts, **new_prompts}


import logging
from helpers.data_backend.base import BaseDataBackend
from tqdm import tqdm
import os

logger = logging.getLogger("PromptHandler")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class PromptHandler:
    def __init__(
        self,
        args: dict,
        text_encoders: list,
        tokenizers: list,
        accelerator,
        model_type: str = "sdxl",
    ):
        if args.disable_compel:
            raise Exception(
                "--disable_compel was provided, but the Compel engine was still attempted to be initialised."
            )

        from compel import Compel, ReturnedEmbeddingsType

        self.accelerator = accelerator
        self.encoder_style = model_type
        if (
            len(text_encoders) == 2
            and text_encoders[1] is not None
            and text_encoders[0] is not None
        ):
            # SDXL Refiner and Base can both use the 2nd tokenizer/encoder.
            logger.debug("Initialising Compel prompt manager with dual text encoders.")
            self.compel = Compel(
                tokenizer=tokenizers,
                text_encoder=text_encoders,
                truncate_long_prompts=False,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[
                    False,  # CLIP-L does not produce pooled embeds.
                    True,  # CLIP-G produces pooled embeds.
                ],
                device=accelerator.device,
            )
        elif len(text_encoders) == 2 and text_encoders[0] is None:
            # SDXL Refiner has ONLY the 2nd tokenizer/encoder, which needs to be the only one in Compel.
            logger.debug(
                "Initialising Compel prompt manager with just the 2nd text encoder."
            )
            self.compel = Compel(
                tokenizer=tokenizers[1],
                text_encoder=text_encoders[1],
                truncate_long_prompts=False,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=True,
                device=accelerator.device,
            )
            self.encoder_style = "sdxl-refiner"
        else:
            # Any other pipeline uses the first tokenizer/encoder.
            logger.debug(
                "Initialising the Compel prompt manager with a single text encoder."
            )
            pipe_tokenizer = tokenizers[0]
            pipe_text_encoder = text_encoders[0]
            self.compel = Compel(
                tokenizer=pipe_tokenizer,
                text_encoder=pipe_text_encoder,
                truncate_long_prompts=False,
                returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                device=accelerator.device,
            )
            self.encoder_style = "legacy"
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

    @staticmethod
    def retrieve_prompt_column_from_parquet(
        sampler_backend_id: str,
    ) -> str:
        parquetdb = StateTracker.get_parquet_database(sampler_backend_id)
        dataframe = parquetdb[0]
        if dataframe is None:
            raise ValueError(
                f"Parquet database not found for sampler {sampler_backend_id}."
            )
        caption_column = (
            StateTracker.get_data_backend_config(sampler_backend_id)
            .get("parquet", {})
            .get("caption_column", None)
        )
        if not caption_column:
            raise ValueError(
                f"Caption column not found for sampler {sampler_backend_id}. Config: {StateTracker.get_data_backend_config(sampler_backend_id)}"
            )
        # Return just that column
        all_captions = dataframe[caption_column].values
        fallback_caption_column = (
            StateTracker.get_data_backend_config(sampler_backend_id)
            .get("parquet", {})
            .get("fallback_caption_column")
        )
        if fallback_caption_column is not None and all_captions is not None:
            # Combine the lists
            fallback_captions = dataframe[fallback_caption_column].values
            all_captions = [
                x if x else y for x, y in zip(all_captions, fallback_captions)
            ]
        return all_captions

    @staticmethod
    def prepare_instance_prompt_from_parquet(
        image_path: str,
        use_captions: bool,
        prepend_instance_prompt: bool,
        data_backend: BaseDataBackend,
        instance_prompt: str = None,
        sampler_backend_id: str = None,
    ) -> str:
        if sampler_backend_id is None:
            raise ValueError("Sampler backend ID is required.")
        if not use_captions:
            if not instance_prompt:
                raise ValueError(
                    "Instance prompt is required when instance_prompt_only is enabled."
                )
            return instance_prompt
        metadata_backend = StateTracker.get_data_backend(sampler_backend_id)[
            "metadata_backend"
        ]
        if metadata_backend is None:
            raise ValueError(
                f"Could not find metadata backend for sampler {sampler_backend_id}: {StateTracker.get_data_backend(sampler_backend_id)}"
            )
        (
            parquet_db,
            filename_column,
            caption_column,
            fallback_caption_column,
            identifier_includes_extension,
        ) = StateTracker.get_parquet_database(sampler_backend_id)
        backend_config = StateTracker.get_data_backend_config(
            data_backend_id=data_backend.id
        )
        instance_data_root = backend_config.get("instance_data_root")
        image_filename_stem = image_path
        if instance_data_root is not None and instance_data_root in image_filename_stem:
            image_filename_stem = image_filename_stem.replace(instance_data_root, "")
            if image_filename_stem.startswith("/"):
                image_filename_stem = image_filename_stem[1:]

        if not identifier_includes_extension:
            image_filename_stem = os.path.splitext(image_filename_stem)[0]
        image_caption = metadata_backend.caption_cache_entry(image_filename_stem)
        if instance_prompt is None and fallback_caption_column and not image_caption:
            raise ValueError(
                f"Could not locate caption for image {image_path} in sampler_backend {sampler_backend_id} with filename column {filename_column}, caption column {caption_column}, and a parquet database with {len(parquet_db)} entries."
            )
        elif (
            instance_prompt is None
            and not fallback_caption_column
            and not image_caption
        ):
            raise ValueError(
                f"Could not locate caption for image {image_path} in sampler_backend {sampler_backend_id} with filename column {filename_column}, caption column {caption_column}, and a parquet database with {len(parquet_db)} entries."
            )
        if type(image_caption) == bytes:
            image_caption = image_caption.decode("utf-8")
        if image_caption:
            image_caption = image_caption.strip()
        if prepend_instance_prompt:
            if type(image_caption) == list:
                image_caption = [instance_prompt + " " + x for x in image_caption]
            else:
                image_caption = instance_prompt + " " + image_caption
        return image_caption

    @staticmethod
    def prepare_instance_prompt_from_filename(
        image_path: str,
        use_captions: bool,
        prepend_instance_prompt: bool,
        instance_prompt: str = None,
    ) -> str:
        if not use_captions:
            if not instance_prompt:
                raise ValueError(
                    "Instance prompt is required when instance_prompt_only is enabled."
                )
            return instance_prompt
        image_caption = Path(image_path).stem
        # Underscores to spaces.
        image_caption = image_caption.replace("_", " ")
        if prepend_instance_prompt:
            image_caption = instance_prompt + " " + image_caption
        return image_caption

    @staticmethod
    def prepare_instance_prompt_from_textfile(
        image_path: str,
        use_captions: bool,
        prepend_instance_prompt: bool,
        data_backend: BaseDataBackend,
        instance_prompt: str = None,
    ) -> str:
        if not use_captions:
            if not instance_prompt:
                raise ValueError(
                    "Instance prompt is required when instance_prompt_only is enabled."
                )
            return instance_prompt
        # caption_file = os.path.splitext(image_path)[0] + ".txt"
        caption_file = image_path + '.txt'
        if not data_backend.exists(caption_file):
            raise FileNotFoundError(f"Caption file {caption_file} not found.")
        try:
            image_caption = data_backend.read(caption_file)
            # Convert from bytes to str:
            if type(image_caption) == bytes:
                image_caption = image_caption.decode("utf-8")

            # any newlines? split into array
            if "\n" in image_caption:
                image_caption = image_caption.split("\n")
                # Remove any empty strings
                image_caption = [x for x in image_caption if x]

            if prepend_instance_prompt:
                if type(image_caption) is list:
                    image_caption = [instance_prompt + " " + x for x in image_caption]
                else:
                    image_caption = instance_prompt + " " + image_caption

            return image_caption
        except Exception as e:
            logger.error(f"Could not read caption file {caption_file}: {e}")

    @staticmethod
    def magic_prompt(
        image_path: str,
        use_captions: bool,
        caption_strategy: str,
        prepend_instance_prompt: bool,
        data_backend: BaseDataBackend,
        instance_prompt: str = None,
        sampler_backend_id: str = None,
    ) -> str:
        """Pull a prompt for an image file like magic, using one of the available caption strategies.

        Args:
            image_path (str): The image path.
            caption_strategy (str): Currently, 'filename' or 'textfile'.
            use_captions (bool): If false, the folder containing the image is used as an instance prompt.
            prepend_instance_prompt (bool): If true, the folder name of the image is prepended to the caption.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if caption_strategy == "filename":
            instance_prompt = PromptHandler.prepare_instance_prompt_from_filename(
                image_path=image_path,
                use_captions=use_captions,
                prepend_instance_prompt=prepend_instance_prompt,
                instance_prompt=instance_prompt,
            )
        elif caption_strategy == "textfile":
            # Can return multiple captions, if the file has newlines.
            instance_prompt = PromptHandler.prepare_instance_prompt_from_textfile(
                image_path,
                use_captions=use_captions,
                prepend_instance_prompt=prepend_instance_prompt,
                instance_prompt=instance_prompt,
                data_backend=data_backend,
            )
        elif caption_strategy == "parquet":
            # Can return multiple captions, if the field is a list.
            instance_prompt = PromptHandler.prepare_instance_prompt_from_parquet(
                image_path,
                use_captions=use_captions,
                prepend_instance_prompt=prepend_instance_prompt,
                instance_prompt=instance_prompt,
                data_backend=data_backend,
                sampler_backend_id=sampler_backend_id,
            )
        elif caption_strategy == "instanceprompt":
            return instance_prompt
        else:
            raise ValueError(
                f"Unsupported caption strategy: {caption_strategy}. Supported: 'filename', 'textfile', 'parquet', 'instanceprompt'"
            )

        if type(instance_prompt) is list:
            instance_prompt = instance_prompt[0]
            logger.debug(
                f"Multiple captions found for image {image_path}. Using just the first caption: {instance_prompt}"
            )

        return instance_prompt

    @staticmethod
    def get_all_captions(
        instance_data_root: str,
        use_captions: bool,
        prepend_instance_prompt: bool,
        data_backend: BaseDataBackend,
        caption_strategy: str,
        instance_prompt: str = None,
    ) -> list:
        captions = []
        all_image_files = StateTracker.get_image_files(
            data_backend_id=data_backend.id
        ) or data_backend.list_files(
            instance_data_root=instance_data_root, str_pattern="*.[jJpP][pPnN][gG]"
        )
        backend_config = StateTracker.get_data_backend_config(
            data_backend_id=data_backend.id
        )
        if type(all_image_files) == list and type(all_image_files[0]) == tuple:
            all_image_files = all_image_files[0][2]
        from tqdm import tqdm

        # if caption_strategy == "parquet":
        #     return PromptHandler.retrieve_prompt_column_from_parquet(
        #         sampler_backend_id=data_backend.id
        #     )

        for image_path in tqdm(
            all_image_files,
            desc="Loading captions",
            total=len(all_image_files),
            ncols=125,
        ):
            if caption_strategy == "filename":
                caption = PromptHandler.prepare_instance_prompt_from_filename(
                    image_path=str(image_path),
                    use_captions=use_captions,
                    prepend_instance_prompt=prepend_instance_prompt,
                    instance_prompt=instance_prompt,
                )
            elif caption_strategy == "textfile":
                caption = PromptHandler.prepare_instance_prompt_from_textfile(
                    image_path,
                    use_captions=use_captions,
                    prepend_instance_prompt=prepend_instance_prompt,
                    instance_prompt=instance_prompt,
                    data_backend=data_backend,
                )
            elif caption_strategy == "parquet":
                try:
                    caption = PromptHandler.prepare_instance_prompt_from_parquet(
                        image_path,
                        use_captions=use_captions,
                        prepend_instance_prompt=prepend_instance_prompt,
                        instance_prompt=instance_prompt,
                        data_backend=data_backend,
                        sampler_backend_id=data_backend.id,
                    )
                except:
                    continue
            elif caption_strategy == "instanceprompt":
                return [instance_prompt]
            else:
                raise ValueError(
                    f"Unsupported caption strategy: {caption_strategy}. Supported: 'filename', 'textfile', 'parquet', 'instanceprompt'"
                )

            if type(caption) not in [tuple, list, dict]:
                captions.append(caption)
            else:
                # allow caching of multiple captions, if returned by the backend.
                captions.extend(caption)

        return captions

    @staticmethod
    def filter_caption(data_backend: BaseDataBackend, caption: str) -> str:
        """Just filter a single caption.

        Args:
            data_backend (BaseDataBackend): The data backend for the instance.
            caption (str): The caption to filter.

        Raises:
            e: If caption filter list can not be loaded.
            ValueError: If we have an invalid filter list.
            FileNotFoundError: If the filter list can not be found.

        Returns:
            str: The filtered caption.
        """
        return PromptHandler.filter_captions(data_backend, [caption])[0]

    @staticmethod
    def filter_captions(data_backend: BaseDataBackend, captions: list) -> list:
        """
        If the data backend config contains the entry "caption_filter_list", this function will filter the captions.

        The caption_filter file contains strings or regular expressions, one per line.

        If a line doesn't have any regex control characters in it, we'll treat it as a string.
        """
        data_backend_config = StateTracker.get_data_backend_config(
            data_backend_id=data_backend.id
        )
        caption_filter_list = data_backend_config.get("caption_filter_list", None)
        if not caption_filter_list or caption_filter_list == "":
            return captions
        if (
            type(caption_filter_list) == str
            and os.path.splitext(caption_filter_list)[1] == ".json"
        ):
            # It's a path to a filter list. Load it in JSON format.
            caption_filter_list_path = Path(caption_filter_list)
            try:
                with open(caption_filter_list_path, "r") as caption_filter_list:
                    caption_filter_list = json.load(caption_filter_list)
            except Exception as e:
                logger.error(
                    f"Caption filter list for data backend '{data_backend.id}' could not be loaded: {e}"
                )
                raise e
        elif (
            type(caption_filter_list) == str
            and os.path.splitext(caption_filter_list)[1] == ".txt"
        ):
            # We have a plain text list of filter strings/regex. Load them into an array:
            caption_filter_list_path = Path(caption_filter_list)
            try:
                with open(caption_filter_list_path, "r") as caption_filter_list:
                    caption_filter_list = caption_filter_list.readlines()
                    # Strip newlines from the ends:
                    caption_filter_list = [x.strip("\n") for x in caption_filter_list]
            except Exception as e:
                logger.error(
                    f"Caption filter list for data backend '{data_backend.id}' could not be loaded: {e}"
                )
                raise e
        # We have the filter list. Is it valid and non-empty?
        if type(caption_filter_list) != list or len(caption_filter_list) == 0:
            logger.debug(
                f"Data backend '{data_backend.id}' has an invalid or empty caption filter list."
            )
            return captions
        elif type(caption_filter_list) is not list:
            raise ValueError(
                f"Data backend '{data_backend.id}' has an invalid caption filter list: {caption_filter_list}"
            )
        # Iterate through each caption
        filtered_captions = []
        for caption in tqdm(
            captions,
            desc="Filtering captions",
            total=len(captions),
            ncols=125,
            disable=True if len(captions) < 10 else False,
        ):
            if type(caption) is list:
                caption = caption[0]
            modified_caption = caption
            # Apply each filter to the caption
            logger.debug(f"Filtering caption: {modified_caption}")
            if modified_caption is None:
                logger.error(
                    f"Encountered a None caption in the list, data backend: {data_backend.id}"
                )
                continue
            for filter_item in caption_filter_list:
                # Check for special replace pattern 's/replace/entry/'
                if filter_item.startswith("s/") and filter_item.count("/") == 2:
                    _, search, replace = filter_item.split("/")
                    regex_modified_caption = re.sub(search, replace, modified_caption)
                    if regex_modified_caption != modified_caption:
                        # logger.debug(
                        #     f"Applying regex SEARCH {filter_item} to caption: {modified_caption}"
                        # )
                        modified_caption = regex_modified_caption
                else:
                    # Treat as plain string and remove occurrences
                    if modified_caption is not None:
                        modified_caption = str(modified_caption).replace(
                            filter_item, ""
                        )
                try:
                    # Assume all filters as regex patterns for flexibility
                    pattern = re.compile(filter_item)
                    try:
                        regex_modified_caption = pattern.sub("", modified_caption)
                    except:
                        regex_modified_caption = modified_caption
                    if regex_modified_caption != modified_caption:
                        # logger.debug(
                        #     f"Applying regex FILTER {filter_item} to caption: {modified_caption}"
                        # )
                        modified_caption = regex_modified_caption
                except re.error as e:
                    logger.error(f"Regex error with pattern {filter_item}: {e}")

            # Add the modified caption to the filtered list
            # if caption != modified_caption:
            #     logger.debug(
            #         f"After all filters have finished, here is the modified caption: {modified_caption}"
            #     )
            filtered_captions.append(modified_caption)

        # Return the list of modified captions
        return filtered_captions

    @staticmethod
    def load_user_prompts(user_prompt_path: str = None):
        if not user_prompt_path:
            return {}
        # Does the file exist?
        user_prompt_path = Path(user_prompt_path)
        if not user_prompt_path.exists():
            raise FileNotFoundError(f"User prompt file {user_prompt_path} not found.")
        # Load the file.
        try:
            with user_prompt_path.open("r") as f:
                user_prompts = json.load(f)
                # logger.debug(f"Loaded user prompts: {user_prompts}")
            return user_prompts
        except Exception as e:
            logger.error(f"Could not read user prompt file {user_prompt_path}: {e}")
            return {}

    def process_long_prompt(self, prompt: str, num_images_per_prompt: int = 1):
        if "sdxl" in self.encoder_style:
            logger.debug(
                f"Running dual encoder Compel pipeline for batch size {num_images_per_prompt}."
            )
            # We need to make a list of prompt * num_images_per_prompt count.
            prompt = [prompt] * num_images_per_prompt
            conditioning, pooled_embed = self.compel(prompt)
        else:
            logger.debug("Running single encoder Compel pipeline.")
            conditioning = self.compel.build_conditioning_tensor(prompt)
        [conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
            [conditioning]
        )
        if "sdxl" in self.encoder_style:
            logger.debug("Returning pooled embeds along with hidden states.")
            return (conditioning, pooled_embed)
        logger.debug("Returning legacy style hidden states without pooled embeds")
        return conditioning
