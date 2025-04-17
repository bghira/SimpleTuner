import os
import pillow_jxl
from PIL import Image

# Define the image size
img_size = (768, 768)

# Define the directory
directory = os.getcwd()

# Get a list of all the image files
files = [f for f in os.listdir(directory) if f.endswith(".png")]

# Extract subjects from files
subjects = list(set([f.split("-")[0] for f in files]))

# For each subject, sort the files and combine them
for subject in subjects:
    # Get all images of the current subject
    subject_files = [f for f in files if f.startswith(subject)]
    subject_files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))

    # Create a new blank image to paste the others onto
    new_image = Image.new("RGB", (len(subject_files) * img_size[0], img_size[1]))

    # For each image file
    for i, file in enumerate(subject_files):
        # Open the image file
        img = Image.open(file)

        # Paste the image into the new image
        new_image.paste(img, (i * img_size[0], 0))

    # Save the new image
    new_image.save(f"{subject}-combined.png")
