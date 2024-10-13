'''
This script is used to recrop your JSON image metadata dataset so that you can
safely delete the VAE cache and then recreate it with new crops. Just point it
at your multidatabackend.json file and it will take care of the rest, then
delete your VAE cache folder and let ST recache.
'''

import sys
import json
import random
import shutil
import os


def update_crop_coordinates_from_multidatabackend(multidatabackend_file):
    # Read the multidatabackend.json file
    with open(multidatabackend_file, 'r') as f:
        datasets = json.load(f)
    
    # Ensure datasets is a list
    if not isinstance(datasets, list):
        datasets = [datasets]
    
    for dataset in datasets:
        # Skip datasets that are disabled
        if dataset.get('disabled', False):
            continue

        # Get required fields
        instance_data_dir = dataset.get('instance_data_dir')
        cache_file_suffix = dataset.get('cache_file_suffix')

        if not instance_data_dir or not cache_file_suffix:
            print(f"Skipping dataset {dataset.get('id', 'unknown')} due to missing 'instance_data_dir' or 'cache_file_suffix'")
            continue

        # Build the metadata file path
        metadata_file = os.path.join(instance_data_dir, f'aspect_ratio_bucket_metadata_{cache_file_suffix}.json')

        # Check if metadata file exists
        if not os.path.exists(metadata_file):
            print(f"Metadata file {metadata_file} does not exist, skipping")
            continue

        # Now process the metadata file
        with open(metadata_file, 'r') as f:
            data = json.load(f)

        for key in data:
            metadata = data[key]
            inter_size = metadata.get('intermediary_size')
            target_size = metadata.get('target_size')
            if inter_size is None or target_size is None:
                continue

            # Assuming sizes are in (height, width) format
            inter_height, inter_width = inter_size
            target_height, target_width = target_size

            max_crop_top = max(inter_height - target_height, 0)
            max_crop_left = max(inter_width - target_width, 0)

            crop_top = random.randint(0, max_crop_top)
            crop_left = random.randint(0, max_crop_left)

            # Update the crop_coordinates
            metadata['crop_coordinates'] = [crop_top, crop_left]

        # Backup the original metadata file
        backup_file = metadata_file + '.bak'
        shutil.copyfile(metadata_file, backup_file)

        # Write the updated data back to the metadata file
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Updated crop_coordinates in {metadata_file}, backup saved as {backup_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_crop_coordinates.py multidatabackend.json")
        sys.exit(1)
    multidatabackend_file = sys.argv[1]
    update_crop_coordinates_from_multidatabackend(multidatabackend_file)