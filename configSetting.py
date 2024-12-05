#This code checks which system is running the code (work or home desktop) and uses the respective paths setup in "config.jason" so all paths donsnt have to be changed all the time.

import json
import os
import socket

def load_settings():
    # Load the configuration file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.json")  # Absolute path to config.json
    with open(config_path, "r") as file:
        config = json.load(file)

    # Get the hostname of the current machine
    hostname = socket.gethostname()

    # Define mappings between hostnames and environments
    hostname_to_env = {
        "Work-PC-Name": "work",
        "DESKTOP-5D7QE86": "home"
    }

    # Determine the environment
    environment = hostname_to_env.get(hostname)
    if environment is None:
        raise ValueError(f"Hostname '{hostname}' not recognized. Update the configuration file or hostname_to_env mapping.")

    # Retrieve the relevant configuration
    settings = config[environment]

    # Access variables from the selected environment
    originalFullSizeImages_path = settings["originalFullSizeImages_path"]
    full_image_annotations_path = settings["full_image_annotations_path"]
    originalPatchedImages_path = settings["originalPatchedImages_path"]
    savePatches_path = settings["savePatches_path"]
    cocoFormat_patched_labels_path = settings["cocoFormat_patched_labels_path"]

    # Example usage
    print("originalFullSizeImages_path", originalFullSizeImages_path)
    print("Full Image Annotations Path:", full_image_annotations_path)
    print("originalPatchedImages_path:", originalPatchedImages_path)
    print("Output Annotations Directory:", savePatches_path)
    print("COCO Labels Path:", cocoFormat_patched_labels_path)

    return originalFullSizeImages_path, savePatches_path, originalPatchedImages_path, full_image_annotations_path, cocoFormat_patched_labels_path