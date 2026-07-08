# Simple script to grab the flower dataset from Kaggle.
# Just make sure your kaggle.json is set up.

import kagglehub
import os
import shutil  # Import the shutil library for directory operations
import logging

# --- Setup & Config ---

# The slug for the dataset on Kaggle Hub.
KAGGLE_DATASET_SLUG = "imsparsh/flowers-dataset"

# Where we'll put the final image files.
FINAL_DATASET_PATH = "./dataset" 

# The local path to our kaggle.json file.
KAGGLE_API_KEY_PATH = "./.kaggle" 

# This tells the Kaggle library where our API key is.
os.environ['KAGGLE_CONFIG_DIR'] = KAGGLE_API_KEY_PATH

# Set up some basic logging to see what's happening.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_the_dataset():
    """
    Downloads the flower dataset using the Kaggle API and moves the
    contents to the local 'dataset' folder.
    """
    print("Alright, let's get this dataset...")
    logging.info(f"Using Kaggle key from: '{os.path.abspath(KAGGLE_API_KEY_PATH)}'")
    logging.info(f"About to download: '{KAGGLE_DATASET_SLUG}'")

    try:
        # kagglehub.dataset_download now returns the path to the DIRECTORY
        # where the dataset files are cached.
        downloaded_cache_path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
        
        logging.info(f"Success! Kaggle Hub cached the dataset at: {downloaded_cache_path}")

        # The actual images are often in a subdirectory. Let's find it.
        # For this dataset, it's a folder named 'flowers'.
        source_data_path = os.path.join(downloaded_cache_path, 'flowers')

        if not os.path.exists(source_data_path):
            # As a fallback, check if the files are in the root of the cache path
            source_data_path = downloaded_cache_path
            if not any(os.scandir(source_data_path)):
                 raise FileNotFoundError("Downloaded cache path is empty.")


        # Make sure our target directory exists and is empty to avoid errors
        if os.path.exists(FINAL_DATASET_PATH):
            logging.warning(f"'{FINAL_DATASET_PATH}' already exists. Removing it for a clean copy.")
            shutil.rmtree(FINAL_DATASET_PATH)
        
        os.makedirs(FINAL_DATASET_PATH, exist_ok=True)
        
        logging.info(f"Copying files from '{source_data_path}' to '{os.path.abspath(FINAL_DATASET_PATH)}'...")
        
        # Use shutil.copytree to copy the entire directory content
        shutil.copytree(source_data_path, FINAL_DATASET_PATH, dirs_exist_ok=True)
        
        logging.info("Copying complete.")
        
        print("\n" + "="*50)
        print("All done!")
        print(f"   You can find the flower images here:")
        print(f"   --> {os.path.abspath(FINAL_DATASET_PATH)}")
        print("="*50)

    except Exception as e:
        # If something went wrong, provide a clear message.
        logging.error(f"Something went wrong. Error: {e}")
        logging.error(
            "Check if your folder setup is correct and that the dataset downloaded properly.\n"
            "Expected setup:\n"
            "Flowers/\n"
            "└── .kaggle/\n"
            "    └── kaggle.json"
        )

# This just makes sure the script runs when called from the command line.
if __name__ == "__main__":
    get_the_dataset()

