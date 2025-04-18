import os
try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image, ImageDraw, ImageFont
from helpers.prompts import prompts

grid_dir = "/notebooks/SimpleTuner/grid"
output_dir = "/notebooks/datasets/test_results"

image_files = [
    os.path.join(output_dir, file)
    for file in os.listdir(output_dir)
    if file.endswith(".png")
]


def create_image_grid(images, num_columns):
    # Determine the grid size
    num_rows = len(images) // num_columns
    if len(images) % num_columns != 0:
        num_rows += 1

    # Determine the size of each image
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)

    # Create a new image for the grid
    grid_img = Image.new("RGB", (num_columns * max_width, num_rows * max_height))

    # Paste the images into the grid
    for i, img in enumerate(images):
        row = i // num_columns
        col = i % num_columns
        grid_img.paste(img, (col * max_width, row * max_height))

    return grid_img


def get_wrapped_text(text: str, font: ImageFont.ImageFont, line_length: int):
    lines = [""]
    for word in text.split():
        line = f"{lines[-1]} {word}".strip()
        if font.getlength(line) <= line_length:
            lines[-1] = line
        else:
            lines.append(word)
    return "\n".join(lines)


shortname_to_files = {}
for file in image_files:
    shortname = os.path.basename(file).split("-")[0]
    logging.info(f"Shortname: {shortname}")
    if shortname not in shortname_to_files:
        shortname_to_files[shortname] = []
    shortname_to_files[shortname].append(file)

for shortname, files in shortname_to_files.items():
    files.sort()
    images = [Image.open(file) for file in files]
    labels = [os.path.basename(file) for file in files]  # get labels from filenames
    prompt = shortname

    # Concatenate images horizontally with black borders in between
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths) + 10 * (len(images) - 1)
    max_height = max(heights)

    # Create a new image for the prompt
    prompt_img = Image.new(
        "RGB", (200, max_height), "white"
    )  # 200 is the width of the prompt frame
    prompt_draw = ImageDraw.Draw(prompt_img)
    prompt_font = ImageFont.truetype("arial.ttf", 16)  # Change the font size here
    font = ImageFont.load_default()
    prompt_draw.text(
        (10, max_height // 2),
        get_wrapped_text(prompt, prompt_font, 25),
        font=prompt_font,
        fill="black",
    )

    # Create a new image for the grid
    grid_img = Image.new(
        "RGB", (total_width + 210, max_height + 50)
    )  # +210 for prompt frame and border
    grid_draw = ImageDraw.Draw(grid_img)
    x_offset = 210  # start from the right of the prompt frame
    logging.info(f"Loop begin")
    for img, label in zip(images, labels):
        grid_draw.text(
            (x_offset, 0), get_wrapped_text(label, font, 25), font=font, fill="white"
        )
        grid_img.paste(img, (x_offset, 50))
        x_offset += img.width + 10  # +10 for border
    logging.info(f"Loop end")

    # Paste the prompt image to the grid image
    grid_img.paste(prompt_img, (0, 50))
    logging.info(f"Saving...")
    grid_img.save(f"{grid_dir}/{shortname}_grid.png")
exit(0)

grouped_files = {}
for file in image_files:
    # get parameters from filename
    basename = ".".join(os.path.basename(file).split(".")[:-1])
    logging.info(f"Found basename: {basename}")
    (
        shortname,
        TIMESTEP_TYPE,
        RESCALE_BETAS_ZEROS_SNR,
        GUIDANCE,
        GUIDANCE_RESCALE,
    ) = basename.split("_")
    GUIDANCE = int(GUIDANCE[:-1])  # remove 'g' and convert to int
    GUIDANCE_RESCALE = float(GUIDANCE_RESCALE[:-1])  # remove 'r' and convert to float

    # group files by shortname, TIMESTEP_TYPE, and RESCALE_BETAS_ZEROS_SNR
    key = (shortname, TIMESTEP_TYPE, RESCALE_BETAS_ZEROS_SNR)
    if key not in grouped_files:
        grouped_files[key] = {}
    if GUIDANCE not in grouped_files[key]:
        grouped_files[key][GUIDANCE] = {}
    grouped_files[key][GUIDANCE][GUIDANCE_RESCALE] = file


def stack_images_vertically(images):
    # Determine the total height of all images
    total_height = sum(i.size[1] for i in images)

    # Determine the width of the widest image
    max_width = max(i.size[0] for i in images)

    # Create a new image for the stack
    stack_img = Image.new("RGB", (max_width, total_height))

    # Paste the images into the stack
    y_offset = 0
    for img in images:
        stack_img.paste(img, (0, y_offset))
        y_offset += img.size[1]

    return stack_img


for key, GUIDANCE_dict in grouped_files.items():
    shortname, TIMESTEP_TYPE, RESCALE_BETAS_ZEROS_SNR = key
    # sort GUIDANCE_dict by GUIDANCE
    GUIDANCE_dict = {k: GUIDANCE_dict[k] for k in sorted(GUIDANCE_dict)}
    all_guidance_images = []  # Store all grid images for each GUIDANCE level
    for GUIDANCE, GUIDANCE_RESCALE_dict in GUIDANCE_dict.items():
        # sort GUIDANCE_RESCALE_dict by GUIDANCE_RESCALE
        logging.info(f"Working on GUIDANCE {GUIDANCE}")
        GUIDANCE_RESCALE_dict = {
            k: GUIDANCE_RESCALE_dict[k] for k in sorted(GUIDANCE_RESCALE_dict)
        }
        images = [
            Image.open(GUIDANCE_RESCALE_dict[GUIDANCE_RESCALE])
            for GUIDANCE_RESCALE in GUIDANCE_RESCALE_dict
        ]
        image_grid = create_image_grid(
            images, len(GUIDANCE_RESCALE_dict)
        )  # create grid for each GUIDANCE
        logging.info(f"Saving image..")
        image_grid.save(
            f"{grid_dir}/{shortname}_{TIMESTEP_TYPE}_{RESCALE_BETAS_ZEROS_SNR}_{GUIDANCE}g_grid.png"
        )
        all_guidance_images.append(image_grid)

    # Stack all GUIDANCE level grids vertically
    logging.info(f"Stacking images like shit on a log.")
    stacked_image = stack_images_vertically(all_guidance_images)
    logging.info(f"Saving stack")
    stacked_image.save(
        f"{grid_dir}/{shortname}_{TIMESTEP_TYPE}_{RESCALE_BETAS_ZEROS_SNR}_stacked.png"
    )
