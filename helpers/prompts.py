import json, io
from pathlib import Path
from helpers.training.state_tracker import StateTracker

prompts = {
    "zen_garden_snowfall": "A serene Zen garden during a gentle snowfall.",
    "yoyo_ma": "world yo-yo champion movie kid played by Yo Yo Ma",
    "jester": "stunning photographs of jesters at the twisted carnival",
    "woman": "Blonde female model, white shirt, white pants, white background, studio",
    "woman2": "autumn portrait of a russian young woman inside cafe, knitted fox-hat, fog close-up, storm clouds, wearing red sunglasses, red shirt, (artstation, deviantart)",
    "woman3": "cinematic portrait of girl, award winning photography, short fluffy auburn hair, freckles, photorealistic, hyperrealistic, green eyes, (ultra detailed face), athletic petite toned body, huge boobs, belly button , smile, Best quality, masterpiece, white wet tight spacesuit, surprised, in space capsule",
    "woman4": "best quality, ultra highres, 8k, RAW photo, knee up shot, sharp focus, insanely detailed, highly detailed photo, photorealistic, hyperrealism, continuous angles, perfect composition, natural geometry, rule of thirds, centered, upright , film grain young woman looking past the camera, red lipstick, hot af, freckle, blue eyes, slim petite body, huge breast long fade black flowing hair, 35mm analog film photo, sitting reading a book in a library in a green tank top & blazer, pantyhose, choker",
    "woman5": "epic cinematic 8k wallpaper of Ellen Ripley, from Alien movie, astronaut, standing in Nostromo space shuttle intricate interiors, photorealistic, cinemeatic, movie quality, vfx post production, rtx ray tracing lighting, ultra realistc, hyper-detailed, hd, hdr, 4k, 8k, shot on mitchell bnc camera, octane render, 8K",
    "woman6": "serene Girl with a Pearl Earring delicately gazing, bathed in ethereal moonlight, with intricate details capturing every strand of hair and every glimmer in her eye, presented in a mesmerizing fusion of oil painting, charcoal sketch, and watercolor, blending the essence of Baroque, Impressionism, and Surrealism.",
    "woman7": "furry art, uploaded on e621, by Pino Daeni, furry, anthro, calico cat, solo, standing, stormy day background, long blue hair, breasts, soft, shirt, rain coat, pants, green eyes, sharp focus, furry calico cat, bloom, illumination, shadows, reflections, shine, glare, ambient light, ultra realistic, ultra detailed, best quality, smiling face",
    "woman8": "extremely detailed 4k digital painting of lady Dracula, highly detailed face cute but mischievous, hyper detailed blonde loose hair, face and lips are in blood stains, detailed pupils, highly detailed symmetrical eyes, highly rendered eyes, smiling, natural skin texture, perfect composition, half body shot teeth sharp, standing, centered, lots of fine details vampire horror movie style), trending on Coppola's Dracula movie, giger, artgerm, greg rutkowski, highly detailed, insanely realistic, ornate, epic realistic, uhd, 4k, award winning photograph, standing, concept art, cinematic lighting, lots of fine details, hyper realistic, focused, extreme details, real shadow, vogue, dark setting, depth of field, f/1.8, 85mm, ultra detailed, highest detail quality, overcast reflection mapping, ultraphotorealistic, movie quality, vfx post production, tx ray tracing lighting, Shot on IMAX 70mm, 8K post-processing, weta FX, dark background, cinematic shot on Nikon Z9, unreal engine 5, octane render, 8k",
    "woman9": "portrait photography of a beautiful Japanese young geisha with make-up looking at the camera, intimate portrait composition, high detail, a hyper-realistic close-up portrait, symmetrical portrait, in-focus, portrait photography, hard edge lighting photography, essay, portrait photo man, photorealism, serious eyes, leica 50mm, DSLRs, f1.8, artgerm, dramatic, moody lighting, post-processing highly detailed",
    "wesanderson": "a stunning portrait of a soviet television news show in a 1977 wes anderson style 70mm film shoot",
    "wizarddd": "digital art, fantasy, portrait of an old wizard warlock man, detailed",
    "wizard": "a handsome mage wizard man, bearded and gray hair, blue  star hat with wand and mystical haze",
    "wizard_tower": "An old wizard's tower, filled with magical artifacts and spellbooks.",
    "wild_west": "A tense standoff in a dusty Wild West town.",
    "wild_horses": "Wild horses galloping across a dusty plain at sunset, sharp",
    "white": "a solid white background",
    "vaping": "a handsome man, vaping a massive cloud in a coffee shop, black and white, sharp",
    "vampire_castle": "A vampire's castle on a stormy night.",
    "urban_graffiti": "An urban alleyway filled with vibrant graffiti",
    "underwater_city": "A thriving city under the sea, inhabited by merpeople.",
    "time_travel": "A time traveler stepping out of their machine into an unknown era.",
    "target": "an embarassing family portrait, photography from 1980s kodachrome style realistic",
    "swimmingpool_jello": " a stunning portrait of a swimming pool full of shiny emerald jello cubes",
    "superhero_hideout": "The hidden underground hideout of a superhero.",
    "stunning_portrait": "a stunning portrait of a stunning portrait of a stunning portrait of a stunning portrait of a stunning portrait",
    "stubble": "a portrait of a freckled man with 5-o'clock shadow",
    "stolen_robot": "an epically massive mecha robot in a fighting stance against a skyscraper in Manhattan, destruction, explosions, professional, masterpiece, majestic",
    "steampunk_city": "A steampunk city in the midst of an industrial revolution.",
    "spielberg": "photograph of stephen speilberg meeting an extra terrestrial",
    "spaghetti_coaster": "a rollercoaster made of spaghetti",
    "space_station": "An advanced space station orbiting a distant exoplanet.",
    "smoking": "a child smoking a cigarette in a coffee shop, black and white, 1960, kodachrome",
    "serene_lake": "A serene lake surrounded by snow-capped mountains at sunrise",
    "scientist_discovery": "A scientist in a lab, just as they make a groundbreaking discovery.",
    "robot": "a melodramatic android serving coffee in a cafe, photography, kodachrome",
    "robot_human_stars": "A robot and a human, sitting together and observing the stars.",
    "robin2": "robin williams, starring in a 1985 television show on VHS with film grain",
    "robin": "robin williams, starring in a 1985 television show on VHS film, MASH, korean war, sharp focus",
    "rainbow_waterfall": "A stunning waterfall with a rainbow arcing through the mist",
    "pomeranian": "image of a sleepy pomeranian",
    "post_apocalyptic": "A post-apocalyptic landscape with nature reclaiming urban spaces.",
    "pirate_treasure": "A pirate discovering an ancient treasure on a deserted island.",
    "neon_metropolis": "A futuristic metropolis lit by neon lights.",
    "mystic_forest": "A mystical forest inhabited by mythical creatures.",
    "mysterious_figure": "A mysterious figure standing at the edge of a dark forest.",
    "movie_description": "A psychological comedy that plays with time, merging the suspenseful mystery of a Jim Carrey classic with the mind-bending narrative of a contemporary science fiction film.",
    "movie_cover": "DVD cover of the movie 'Paw Patrol:Crime Scene Investigation'",
    "moon_colony": "A thriving colony on the moon, domes, people in space suits, lunar rovers",
    "micro": "macro close-up view of RNA and other molecular machinery of life in ultra high detail with realistic textures, sharp focus",
    "menn2": "a group of handsome man, standing proudly outside of a pub in Ireland at night, illuminated by neon lights",
    "menn": "photograph of a group of handsome men, standing proudly outside of a pub in Ireland during nighttime, illuminated by parking lot lighting",
    "medieval_battle": "An epic medieval battle, knights charging on horseback",
    "mash": "scene from a 1985 television show starring danny devito and robin williams, MASH, korean war, sharp focus",
    "mars_rover": "A Mars rover exploring the rugged Martian terrain",
    "man": "a happy and smiling man playing guitar in a park at nighttime while party lanterns and balloons are visible around him",
    "magical_library": "A magical library with books that fly off the shelves.",
    "macro": "a wide angle view of a dramatic city skyline during sunrise",
    "larry": "look at my cousin larry in his room, he is such a mess",
    "last_dinosaur": "The last dinosaur on Earth looking at the incoming meteor.",
    "kodachrome": "a 1960s scene in kodachrome where hippies hang out in san francisco's haight area",
    "knight": "a handsome knight protecting a castle, brilliant masterpiece, oil painting",
    "knight_dragon_battle": "A medieval knight and a dragon, engaged in an epic battle.",
    "kanagawa": "The Great Wave off Kanagawa",
    "harry_houdini": "harry houdini locking his keys in his car",
    "haunted_mansion": "A haunted mansion at the end of a winding path.",
    "greek_mythology": "A scene from Greek mythology, with gods and mythical creatures.",
    "giant_robot": "A giant robot protecting a city from a natural disaster.",
    "gecko": "a leopard gecko walking toward a grasshopper, sharp focus",
    "garfield_cat": "garfield the cat's inauguration",
    "garfield_president": "garfield's inauguration",
    "garbagepailkids": "garbage pail kids",
    "future_cityscape": "A cityscape of a utopian future with advanced technology.",
    "fantasy_map": "A map of a fantasy world with various terrains and kingdoms.",
    "family_picnic": "A family enjoying a picnic in a blooming spring meadow",
    "fairy_glen": "A hidden glen where fairies gather to dance in the moonlight.",
    "elijah": "elijah wood looking at himself in the mirror trying to decide his chaotic alignment for the day",
    "dystopian_city": "A dystopian city under a gloomy, polluted sky, sharp focus",
    "dinosaur_park": "A modern-day park where genetically engineered dinosaurs roam freely.",
    "dust": "a construction worker using a jackhammer with dust settling",
    "devito2": "danny devito, starring in 1985 televison show on VHS with film grain",
    "devito": "danny devito, starring in a 1985 television show on VHS, MASH, korean war, sharp focus",
    "darktv": "a glowing television screen in a dark room",
    "cyborg_concert": "A cyborg playing a futuristic instrument in a crowded concert.",
    "cyberpunk_street": "A bustling street in a cyberpunk city.",
    "cosmic": "cosmic entity sitting in an impossible position, quantum reality, colours, fractals",
    "coral_reef": "A vibrant coral reef teeming with life, underwater, sharp focus",
    "comparison": "Film still, mid shot, Fashion Photography, a very pretty American girl, colourful straight hairstyle, plain colour skirt, realistic skin texture, photo realistic, raining, night time, Low saturation, masterpiece, global illumination, cinematic realism",
    "childmid": "a school portrait of a happy child, sharp focus",
    "child": "a happy child flying a kite on a sunny day",
    "child_dreaming": "A child dreaming of fantastical creatures and far-off worlds.",
    "cake_monalisa": "photo of a cake mona lisa",
    "busy_market": "A bustling market scene in Marrakech, vibrant colors, sharp",
    "black": "a solid black background",
    "bicycle": "a downhill mountain bike on a mountain road on a sunny day, sharp",
    "beach_sunset": "A sunset over a tranquil beach, with dolphins leaping from the water.",
    "baked_combo": "the pope as a geisha",
    "avocado": "An illustration of an avocado sitting in a therapist's chair, saying 'I just feel so empty inside' with a pit-sized hole in its center. The therapist, a spoon, scribbles notes",
    "aurora_night": "A dazzling display of the aurora borealis over a snow-covered landscape, sharp focus",
    "astronaut_space": "An astronaut floating in the vast expanse of space, Earth in the background.",
    "artful_dodger": "the artful dodger, cool dog in sunglasses sitting on a recliner in the dark, with the white noise reflecting on his sunglasses",
    "architecture": "a beautiful parametric architecture building",
    "ancient_temple": "An ancient temple, overgrown with vines, deep in the jungle.",
    "ancient_ruins": "Ancient ruins swallowed by a dense jungle",
    "ancient_egypt": "A view of ancient Egypt, with pyramids and the Sphinx.",
    "alien": "an ancient alien exploring the Martian surface, photorealistic",
    "alien_marketplace": "A bustling marketplace in an alien world.",
    "alien_invasion": "The first moments of an alien invasion from a civilian's perspective.",
    "butterfly": "photo of a yellow butterfly perched on a colorful flower, taken with a Canon EOS Rebel T6 camera on aperture f/4",
    "spiderweb": "a close-up of a delicate spider web with morning dew drops, captured with a Sony Alpha A7 III on f/8",
    "cityscape_sunset": "a striking black and white image of a cityscape at sunset, shot with a Nikon D7500 on f/8",
    "dronecam": "an aerial view of a lush green forest, photographed with a DJI Mavic Pro drone at 200 feet",
    "otters": "an adorable shot of a family of otters playing in the waves, captured with a Nikon D7500 on f/8",
    "mountains": "a majestic portrait of a snow-capped mountain range, taken with a Canon EOS RP on f/16",
    "rainbow_park": "a vibrant shot of a rainbow over a colorful park, captured with an iPhone 11 Pro Max on f/16",
    "stormy_sky_night": "a powerful image of a stormy sky at night, photographed with a Sony Alpha A7 III on f/8",
    "masked_dancer": "a thought-provoking portrait of a masked dancer, shot with a Canon EOS 5D Mark IV on f/2",
    "mist_over_lake": "an ethereal image of mist floating over a serene lake, captured with a Fujifilm X-T3 on f/11",
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
from helpers.data_backend.aws import S3DataBackend
from pathlib import Path
import os

logger = logging.getLogger("PromptHandler")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING"))


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
        if len(text_encoders) == 2 and text_encoders[1] is not None:
            # SDXL Refiner and Base can both use the 2nd tokenizer/encoder.
            logger.debug(f"Initialising Compel prompt manager with dual text encoders.")
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
                f"Initialising Compel prompt manager with just the 2nd text encoder."
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
                f"Initialising the Compel prompt manager with a single text encoder."
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

    @staticmethod
    def prepare_instance_prompt(
        image_path: str,
        use_captions: bool,
        data_backend: BaseDataBackend,
        prepend_instance_prompt: bool,
        instance_prompt: str = None,
    ) -> str:
        instance_prompt = Path(image_path).stem
        if not instance_prompt and prepend_instance_prompt:
            # If we did not get a specific instance prompt, use the folder name.
            logger.debug(f"Prepending instance prompt: {instance_prompt}")
            if type(data_backend) == S3DataBackend:
                raise ValueError(
                    "S3 data backend is not yet compatible with --prepend_instance_prompt"
                )
        if use_captions:
            # Underscores to spaces.
            instance_prompt = instance_prompt.replace("_", " ")
            # Remove some midjourney messes.
            instance_prompt = instance_prompt.split("upscaled by")[0]
            instance_prompt = instance_prompt.split("upscaled beta")[0]
            if prepend_instance_prompt:
                instance_prompt = instance_prompt + " " + instance_prompt
        return instance_prompt

    @staticmethod
    def prepare_instance_prompt_from_textfile(image_path) -> str:
        caption_file = Path(image_path).with_suffix(".txt")
        if not caption_file.exists():
            raise FileNotFoundError(f"Caption file {caption_file} not found.")
        try:
            with caption_file.open("r") as f:
                instance_prompt = f.read()
            return instance_prompt
        except Exception as e:
            logger.error(f"Could not read caption file {caption_file}: {e}")

    @staticmethod
    def magic_prompt(
        image_path: str,
        caption_strategy: str,
        use_captions: bool,
        prepend_instance_prompt: bool,
        data_backend: BaseDataBackend,
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
            instance_prompt = PromptHandler.prepare_instance_prompt(
                image_path=image_path,
                use_captions=use_captions,
                prepend_instance_prompt=prepend_instance_prompt,
                data_backend=data_backend,
            )
        elif caption_strategy == "textfile":
            instance_prompt = PromptHandler.prepare_instance_prompt_from_textfile(
                image_path
            )
        else:
            raise ValueError(f"Unsupported caption strategy: {caption_strategy}")
        return instance_prompt

    @staticmethod
    def get_all_captions(
        instance_data_root: str,
        use_captions: bool,
        prepend_instance_prompt: bool,
        data_backend: BaseDataBackend,
    ) -> list:
        captions = []
        all_image_files = StateTracker.get_image_files(
            data_backend_id=data_backend.id
        ) or data_backend.list_files(
            instance_data_root=instance_data_root, str_pattern="*.[jJpP][pPnN][gG]"
        )
        if type(all_image_files) == list and type(all_image_files[0]) == tuple:
            all_image_files = all_image_files[0][2]
        for image_path in all_image_files:
            caption = PromptHandler.prepare_instance_prompt(
                image_path=str(image_path),
                use_captions=use_captions,
                prepend_instance_prompt=prepend_instance_prompt,
                data_backend=data_backend,
            )
            captions.append(caption)

        return captions

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
            logger.debug(f"Running single encoder Compel pipeline.")
            conditioning = self.compel.build_conditioning_tensor(prompt)
        [conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
            [conditioning]
        )
        if "sdxl" in self.encoder_style:
            logger.debug(f"Returning pooled embeds along with hidden states.")
            return (conditioning, pooled_embed)
        logger.debug("Returning legacy style hidden states without pooled embeds")
        return conditioning
