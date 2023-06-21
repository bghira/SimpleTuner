import torch, logging, random, time
from PIL import Image
from .state_tracker import StateTracker
import os, json


class BalancedBucketSampler(torch.utils.data.Sampler):
    def __init__(self, aspect_ratio_bucket_indices, batch_size=15):
        self.aspect_ratio_bucket_indices = aspect_ratio_bucket_indices  # A dictionary of string float buckets and their image paths.
        self.buckets = list(
            aspect_ratio_bucket_indices.keys()
        )  # These keys are a float value, eg. 1.78.
        self.exhausted_buckets = (
            []
        )  # Buckets that have been exhausted, eg. all samples have been used.
        self.batch_size = batch_size  # How many images per sample during training. They MUST all be the same aspect.
        self.current_bucket = 0
        self.seen_images_path = "/notebooks/SimpleTuner/seen_images.json"
        self.seen_images = self.load_seen_images()

    def load_seen_images(self):
        if os.path.exists(self.seen_images_path):
            with open(self.seen_images_path, "r") as f:
                seen_images = json.load(f)
        else:
            seen_images = {}
        return seen_images

    def save_seen_images(self):
        with open(self.seen_images_path, "w") as f:
            json.dump(self.seen_images, f)

    def __iter__(self):
        while True:
            if not self.buckets:
                logging.info(f"All buckets are exhausted. Exiting...")
                break

            bucket = self.buckets[self.current_bucket]
            bucket_length = len(self.aspect_ratio_bucket_indices[bucket])

            if bucket_length < self.batch_size:
                if bucket not in self.exhausted_buckets:
                    self.move_to_exhausted()
                self.change_bucket()
                continue

            available_images = [
                image
                for image in self.aspect_ratio_bucket_indices[bucket]
                if image not in self.seen_images
            ]
            if len(available_images) < self.batch_size:
                logging.warn(f"Not enough unseen images in the bucket: {bucket}")
                self.move_to_exhausted()
                self.change_bucket()
                continue

            samples = random.choices(available_images, k=self.batch_size)
            to_yield = []
            for i in range(len(samples)):
                # Reassign the bucket an image is in if its aspect ratio is incorrect
                image_path = samples[i]
                # Does the path exist?
                if not os.path.exists(image_path):
                    logging.warn(f"Image path does not exist: {image_path}")
                    # Remove from the bucket:
                    if image_path in self.aspect_ratio_bucket_indices[bucket]:
                        self.aspect_ratio_bucket_indices[bucket].remove(image_path)
                    continue
                image = Image.open(image_path)
                if image.width < 1024 or image.height < 1024:
                    logging.warn(
                        f"Image too small: DELETING image and continuing search."
                    )
                    image.close()
                    os.remove(image_path)
                    continue
                aspect_ratio = round(image.width / image.height, 3)
                actual_bucket = str(aspect_ratio)
                if actual_bucket != bucket:
                    logging.warn(
                        f"Found an image in a bucket {bucket} it doesn't belong in, when actually it is: {actual_bucket}"
                    )
                    self.aspect_ratio_bucket_indices[bucket].remove(image_path)
                    if actual_bucket in self.aspect_ratio_bucket_indices:
                        logging.warn(f"Moved image to bucket, it already existed.")
                        self.aspect_ratio_bucket_indices[actual_bucket].append(
                            image_path
                        )
                    else:
                        # Create a new bucket if it doesn't exist
                        logging.warn(f"Created new bucket for that pesky image.")
                        self.aspect_ratio_bucket_indices[actual_bucket] = [image_path]
                    # Get a new sample from the same bucket
                    if self.aspect_ratio_bucket_indices[bucket]:
                        logging.debug(f"Yielding sample from bucket: {bucket}")
                        new_sample = self.aspect_ratio_bucket_indices[bucket].pop()
                        new_image = Image.open(new_sample)
                        if new_image.width < 1024 or new_image.height < 1024:
                            logging.warn(
                                f"Image too small: discarding and continuing search."
                            )
                            new_image.close()
                            os.remove(new_sample)
                        to_yield.append(new_sample)
                else:
                    logging.debug(
                        f"Yielding {image.width}x{image.height} sample from bucket: {bucket} with aspect {actual_bucket}"
                    )
                    to_yield.append(image_path)
                    if StateTracker.status_training():
                        self.seen_images[image_path] = actual_bucket
            if len(to_yield) == self.batch_size:
                logging.debug(f"Done yielding.")
                self.log_state()
                if StateTracker.status_training():
                    self.save_seen_images()  # Save seen images
                self.current_bucket = (self.current_bucket + 1) % len(self.buckets)
                for image_to_yield in to_yield:
                    yield image_to_yield

    def __len__(self):
        return sum(
            len(indices) for indices in self.aspect_ratio_bucket_indices.values()
        )

    def change_bucket(self):
        if self.buckets:
            self.current_bucket %= len(self.buckets)
            logging.info(f"Changing bucket to {self.buckets[self.current_bucket]}.")

    def move_to_exhausted(self):
        bucket = self.buckets[self.current_bucket]
        self.exhausted_buckets.append(bucket)
        self.buckets.remove(bucket)
        logging.info(
            f"Bucket {bucket} is empty or doesn't have enough samples for a full batch. Moving to the next bucket."
        )
        self.log_state()

    def log_state(self):
        logging.debug(
            f'Active Buckets: {", ".join(self.convert_to_human_readable(float(b)) for b in self.buckets)}'
        )
        logging.debug(
            f'Exhausted Buckets: {", ".join(self.convert_to_human_readable(float(b)) for b in self.exhausted_buckets)}'
        )

    @staticmethod
    def convert_to_human_readable(aspect_ratio_float: float):
        from math import gcd

        # The smallest side is always 1024. It could be portrait or landscape (eg. under or over 1)
        if aspect_ratio_float < 1:
            ratio_width = 1024
            ratio_height = int(1024 / aspect_ratio_float)
        else:
            ratio_width = int(1024 * aspect_ratio_float)
            ratio_height = 1024

        # Return the aspect ratio as a string in the format "width:height"
        return f"{aspect_ratio_float}"
        return f"{ratio_width}:{ratio_height}"
