import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class PromptExpander:
    # Class variables to hold the model, tokenizer, and generator
    model = None
    tokenizer = None
    generator = None

    @staticmethod
    def initialize_model(model_path="meta-llama/Llama-3.2-1B-Instruct"):
        """
        Initializes the language model, tokenizer, and text generation pipeline.
        """
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        PromptExpander.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        PromptExpander.tokenizer = AutoTokenizer.from_pretrained(model_path)
        PromptExpander.generator = pipeline(
            "text-generation",
            model=PromptExpander.model,
            tokenizer=PromptExpander.tokenizer,
            device=0 if device == "cuda" else -1,
        )

    @staticmethod
    def generate_prompts(trigger_phrase, num_prompts=25):
        """
        Generates expanded prompts based on the provided trigger phrase.

        Args:
            trigger_phrase (str): The trigger phrase to include in the prompts.
            num_prompts (int): The number of prompts to generate.
        """
        # Check if the model is initialized
        if PromptExpander.generator is None:
            print("Model not initialized. Please call initialize_model() first.")
            return

        # Define the list of prompt templates and styles
        prompt_templates = [
            "An image of TRIGGERPHRASE in the style of {style}.",
            "A painting of {style}, featuring TRIGGERPHRASE.",
            "TRIGGERPHRASE as depicted in {style} art.",
            "{style} illustration showing TRIGGERPHRASE.",
            "An abstract representation of TRIGGERPHRASE in {style}.",
            "A realistic portrayal of TRIGGERPHRASE, inspired by {style}.",
            "An artistic depiction of TRIGGERPHRASE using {style} techniques.",
            "A {style} photograph of TRIGGERPHRASE.",
            "TRIGGERPHRASE captured in the style of {style}.",
            "An image featuring TRIGGERPHRASE with {style} elements.",
            "A {style}-inspired scene with TRIGGERPHRASE.",
            "TRIGGERPHRASE illustrated in {style} style.",
            "An artistic rendering of TRIGGERPHRASE in {style} fashion.",
            "A {style}-themed artwork of TRIGGERPHRASE.",
            "TRIGGERPHRASE portrayed in {style} aesthetics.",
            "{style} art featuring TRIGGERPHRASE.",
            "An expressive {style} depiction of TRIGGERPHRASE.",
            "A creative {style} representation of TRIGGERPHRASE.",
            "TRIGGERPHRASE shown in a {style} setting.",
            "An imaginative {style} image of TRIGGERPHRASE.",
            "A {style} design including TRIGGERPHRASE.",
            "TRIGGERPHRASE in a {style} composition.",
            "An illustration of TRIGGERPHRASE with {style} influence.",
            "A {style}-inspired portrait of TRIGGERPHRASE.",
            "An image where TRIGGERPHRASE meets {style}.",
            "TRIGGERPHRASE blended into a {style} background.",
            "A {style} collage featuring TRIGGERPHRASE.",
            "An artistic scene of TRIGGERPHRASE in {style} mood.",
            "A {style} depiction of TRIGGERPHRASE in motion.",
            "TRIGGERPHRASE rendered in {style} tones.",
            "An atmospheric {style} image of TRIGGERPHRASE.",
            "An expressive portrait of TRIGGERPHRASE in {style} style.",
            "A surreal {style} painting of TRIGGERPHRASE.",
            "TRIGGERPHRASE integrated into a {style} landscape.",
            "An abstract {style} representation of TRIGGERPHRASE.",
            "A {style} sketch of TRIGGERPHRASE.",
            "TRIGGERPHRASE depicted in {style} illustration.",
            "An image of TRIGGERPHRASE with {style} patterns.",
            "A {style} poster featuring TRIGGERPHRASE.",
            "An iconic {style} image of TRIGGERPHRASE.",
            "TRIGGERPHRASE in a {style} artwork.",
            "A vibrant {style} depiction of TRIGGERPHRASE.",
            "An ethereal {style} image of TRIGGERPHRASE.",
            "A dynamic {style} scene with TRIGGERPHRASE.",
            "TRIGGERPHRASE portrayed through {style} art.",
            "A {style} mural of TRIGGERPHRASE.",
            "An imaginative {style} illustration of TRIGGERPHRASE.",
            "TRIGGERPHRASE set in a {style} environment.",
            "A {style}-inspired depiction of TRIGGERPHRASE.",
            "An image of TRIGGERPHRASE with {style} motifs.",
        ]

        styles = [
            "Impressionism",
            "Cubism",
            "Surrealism",
            "Pop Art",
            "Futurism",
            "Baroque",
            "Gothic",
            "Abstract Expressionism",
            "Renaissance",
            "Minimalism",
            "Digital Art",
            "Vintage Photography",
            "Sci-Fi",
            "Fantasy",
            "Steampunk",
            "Cyberpunk",
            "Art Deco",
            "Graffiti",
            "Watercolor",
            "Oil Painting",
            "Black and White",
            "Colorful",
            "Retro",
            "Comic Book",
            "Manga",
            "3D Rendering",
            "Low Poly",
            "Pixel Art",
            "Line Art",
            "Flat Design",
            "Concept Art",
            "Photorealism",
            "High Contrast",
            "Monochrome",
            "Collage",
            "Typography",
            "Street Art",
            "Ukiyo-e",
            "Pop Surrealism",
            "Digital Illustration",
            "Neon",
            "Expressionism",
            "Anime",
            "Realism",
            "Dadaism",
            "Constructivism",
            "Avant-Garde",
            "Hyperrealism",
            "Symbolism",
            "Fauvism",
        ]

        used_templates = []
        used_styles = []
        user_prompt_library = {}
        idx = 0

        for _ in range(num_prompts):
            idx += 1
            # Randomly select a prompt template and style
            prompt_template = None
            style = None
            while (prompt_template is None or prompt_template in used_templates) or (style is None or style in used_styles):
                prompt_template = random.choice(prompt_templates)
                style = random.choice(styles)

            used_templates.append(prompt_template)
            used_styles.append(style)

            # Replace placeholders in the template
            prompt = prompt_template.replace("{style}", style.lower())
            prompt = prompt.replace("TRIGGERPHRASE", trigger_phrase)

            # Generate the text
            input_prompt = (
                "You are a text-to-text interface that returns improved prompts. "
                "The captions should be expanded to be more descriptive. "
                "Captions look like sentence fragments and tags.\n\n"
                f"WITHOUT CHANGING ANY SPELLINGS: Clean this prompt, and return NOTHING but the upgraded prompt: {prompt}"
            )

            import time

            begin = time.time()
            output_prompt = None

            def refused(output_prompt):
                triggers = [
                    "i cannot",
                    "can you",
                    "what kind",
                    "i can't",
                    "i won't",
                    "am unable to",
                    "here is the",
                    "here's the",
                    "improving the caption",
                    "improving the prompt",
                    "should ",
                    "improving the text",
                    "improving the sentence",
                    "improving the description",
                    "remember:",
                    "please describe",
                    "please",
                    "improving the tag",
                    "improving the fragment",
                    "i am unable to",
                    "i am not able to",
                    "i am not capable of",
                    "i am not capable to",
                    "i am not capable",
                    "i am not able",
                ]
                if output_prompt is None:
                    return None
                output_prompt_lower = output_prompt.lower()
                for trigger in triggers:
                    if trigger in output_prompt_lower:
                        return trigger
                return None

            refused_term = None
            while output_prompt is None or refused_term is not None:
                if output_prompt is not None and refused_term:
                    print(
                        f"-> (REFUSAL) Prompt contains '{refused_term}'. Generating a new prompt.",
                        end="\n\n",
                    )
                generated_caption = PromptExpander.generator(
                    input_prompt,
                    temperature=0.4,
                    max_new_tokens=77,
                    return_full_text=False,
                )

                end = time.time()
                output_prompt = (
                    generated_caption[0]["generated_text"]
                    .strip()
                    .lower()
                    .replace("create a ", "")
                    .replace("explore the", "")
                    .replace("imagine a", "")
                    .replace("find a", "")
                    .replace('"', "")
                )
                refused_term = refused(output_prompt)

            time_taken = end - begin
            print("Prompt expanded in", round(time_taken, 2), "seconds:")
            print(f"Original prompt: {prompt}")
            print(f"Expanded prompt: {output_prompt}", end="\n\n")
            user_prompt_library[f"prompt_{idx}"] = output_prompt

        # Output the generated prompts as JSON
        return user_prompt_library


if __name__ == "__main__":
    # Example usage:
    # Initialize the model first
    PromptExpander.initialize_model()

    # Generate prompts with your trigger phrase
    user_prompt_library = PromptExpander.generate_prompts(trigger_phrase="your_trigger_phrase_here", num_prompts=25)

    # Write to disk
    with open("config/user_prompt_library.json", "w") as f:
        json.dump(user_prompt_library, f, indent=4)
