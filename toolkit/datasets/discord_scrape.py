import requests
import re
import pandas as pd
from tqdm import tqdm
import signal
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor
import argparse


# Define a signal handler function to handle Ctrl+C
def signal_handler(sig, frame):
    print(f"Ctrl+C detected. Saving data to mj-{current_channel}.parquet...")
    df = pd.DataFrame(all_data)
    df.to_parquet(f"mj-{current_channel}.parquet", engine="pyarrow")
    print(f"Data saved to mj-{current_channel}.parquet. Exiting...")
    sys.exit(0)


def process_channel(channel_id, position, headers):
    """Processes a Discord channel to collect and save message data.

    Args:
        channel_id (int): The ID of the Discord channel to process.
    """
    global retry_after
    url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=50"
    all_data = []  # List to collect all the processed data for this channel
    last_message_id = None  # To keep track of the last message ID for pagination

    for page in tqdm(
        range(2000), position=position.index(channel_id), desc=f"Channel {channel_id}"
    ):
        page_url = url
        if last_message_id:
            page_url += f"&before={last_message_id}"

        time.sleep(retry_after)
        response = requests.get(page_url, headers=headers)
        if response.status_code == 429:
            print(
                f"Rate limited. Waiting for {response.headers['Retry-After']} seconds..."
            )
            time.sleep(retry_after)
            continue
        elif not response.status_code == 200:
            print(f"Failed to retrieve data: {response.status_code}")
            break

        messages = response.json()
        if not messages:
            break  # No more messages to fetch

        last_message_id = messages[-1][
            "id"
        ]  # Update the last_message_id for the next page

        target_source = "Midjourney Bot"
        for entry in messages:
            if entry["author"]["username"] != target_source:
                continue
            if "Variations" in entry["content"] or "Image #" not in entry["content"]:
                continue

            # print(f"Entry id: {entry['id']}, author: {entry['author']['username']}, attachments: {entry['attachments']}")
            # Capture first text group between "**"s using regex
            search = re.search(r"\*\*(.*?)\*\*", entry["content"])
            stripped_content = ""
            if hasattr(search, "group"):
                stripped_content = search.group(1)
            # Remove any <http(s)://..> url with surrounding brackets:
            stripped_content = re.sub(r"<(http[s]?://.*?)>", "", stripped_content)
            # Split the prompt into two pieces, and use only the first piece before --
            pieces = stripped_content.split("--")
            stripped_content = pieces[0].strip()
            version = 5.2
            arguments = pieces[1].strip() if len(pieces) > 1 else ""
            # If we have --v <float> inside the arguments, that is our version:
            version_search = re.search(r"v\s(\d+.\d+)", arguments)
            if hasattr(version_search, "group"):
                version = version_search.group(1)
            # Sometimes, people put "ar x:y" as in, "ar 16 9" or "ar 16:9" without the --, but we want to remove any mention of aspect ratio:
            stripped_content = re.sub(r"ar\s\d+:\d+", "", stripped_content)

            # We likely need to shorten the output so that it can work as a Linux filename:
            stripped_content = stripped_content[:225]

            # print(f"unfilteredcontent: {entry['content']}")
            # print(f"-> content: {stripped_content}\n")
            # print(f"-> attachments: {entry['attachments']}\n")
            # Collecting data
            if len(entry["attachments"]) == 0:
                continue
            processed_data = {
                "id": entry["id"],
                "version": str(version),
                "arguments": arguments,
                "original_text": entry["content"],
                "caption": stripped_content,
                "url": entry["attachments"][0]["url"].split("?")[0],
                "width": entry["attachments"][0]["width"],
                "height": entry["attachments"][0]["height"],
            }
            all_data.append(processed_data)

    # Save the data to a Parquet file at the end of processing this channel
    df = pd.DataFrame(all_data)
    df.to_parquet(f"mj-general-{channel_id}.parquet", engine="pyarrow")


def load_config(config_file):
    """Loads configuration from a JSON file.

    Args:
        config_file (str): Path to the JSON configuration file.

    Returns:
        dict: The configuration data.
    """
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Collect and save message data from Discord channels."
    )
    parser.add_argument("config_file", help="Path to the JSON configuration file.")
    args = parser.parse_args()

    config = load_config(args.config_file)

    # Set the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Discord credentials and settings from configuration
    headers = config["headers"]
    channel_list = config["channel_list"]

    # Use ThreadPoolExecutor to process each channel id in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(process_channel, channel_list)

    print("All data saved.")


if __name__ == "__main__":
    main()
