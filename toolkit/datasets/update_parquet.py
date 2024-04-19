import os
import pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# set 'fork' spawn mode
import multiprocessing as mp

mp.set_start_method("fork")

PARQUET_FILE = "photo-concept-bucket.parquet"
IMAGE_DATA = "output_dir"

# Load the parquet file
df = pd.read_parquet(PARQUET_FILE, engine="pyarrow")

# Function to process a chunk of IDs
import json


def process_images(ids_chunk):
    summary = {
        "id": [],
        "old_width": [],
        "new_width": [],
        "old_height": [],
        "new_height": [],
        "old_aspect_ratio": [],
        "new_aspect_ratio": [],
    }

    for id in ids_chunk:
        metadata_path = os.path.join(IMAGE_DATA, f"{id}.json")
        if not os.path.exists(metadata_path):
            continue
        # Use the simpletuner data if the image is not found
        try:
            with open(metadata_path) as f:
                row = json.load(f)
            width, height = row["image_size"]
            aspect_ratio = row["aspect_ratio"]
        except KeyError:
            print(f"Image {metadata_path} not found in simpletuner data")
            continue

        # Locate the row in the DataFrame
        row = df.loc[df["id"] == id]

        # Check for differences
        if not row.empty and (
            row.iloc[0]["width"] != width or row.iloc[0]["height"] != height
        ):
            print(
                f"Updated image {id}: {row.iloc[0]['width']}x{row.iloc[0]['height']} -> {width}x{height}"
            )
            summary["id"].append(id)
            summary["old_width"].append(row.iloc[0]["width"])
            summary["new_width"].append(width)
            summary["old_height"].append(row.iloc[0]["height"])
            summary["new_height"].append(height)
            summary["old_aspect_ratio"].append(row.iloc[0]["aspect_ratio"])
            summary["new_aspect_ratio"].append(aspect_ratio)

    return summary


# Split IDs into chunks for parallel processing
ids = df["id"].values
num_processes = os.cpu_count()
chunk_size = len(ids) // num_processes + (len(ids) % num_processes > 0)
id_chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]

# Process the images in parallel
with ProcessPoolExecutor(max_workers=num_processes) as executor:
    results = list(tqdm(executor.map(process_images, id_chunks), total=len(id_chunks)))

# Combine results from all processes
combined_summary = pd.DataFrame()
for result in results:
    combined_summary = pd.concat([combined_summary, pd.DataFrame(result)])

# Update the DataFrame based on the combined summary
for index, row in combined_summary.iterrows():
    idx = df.index[df["id"] == row["id"]].tolist()[0]
    df.at[idx, "width"] = row["new_width"]
    df.at[idx, "height"] = row["new_height"]
    df.at[idx, "aspect_ratio"] = row["new_aspect_ratio"]

# Save the updated DataFrame to the parquet file
df.to_parquet(PARQUET_FILE, engine="pyarrow")
