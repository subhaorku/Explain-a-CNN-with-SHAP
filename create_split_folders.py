import os
import shutil
import random

def split_data(original_train_dir, base_output_dir, split_ratio=0.8):
    """
    Splits the data from the original train directory into new train_split
    and test_split folders.

    Args:
        original_train_dir (str): Path to the folder containing class subdirectories (e.g., './dataset/train').
        base_output_dir (str): Path to the folder where 'train_split' and 'test_split' will be created (e.g., './dataset').
        split_ratio (float): The ratio of training data.
    """
    print("="*50)
    print("Starting to split data into new train/test folders...")

    # Define paths for the new split directories
    train_split_dir = os.path.join(base_output_dir, 'train_split')
    test_split_dir = os.path.join(base_output_dir, 'test_split')

    # If the directories already exist, remove them for a clean start
    if os.path.exists(train_split_dir):
        print(f"Removing existing directory: {train_split_dir}")
        shutil.rmtree(train_split_dir)
    if os.path.exists(test_split_dir):
        print(f"Removing existing directory: {test_split_dir}")
        shutil.rmtree(test_split_dir)

    # Create the new empty directories
    os.makedirs(train_split_dir, exist_ok=True)
    os.makedirs(test_split_dir, exist_ok=True)
    print(f"Created new directories:\n  - {train_split_dir}\n  - {test_split_dir}")

    # Get the list of class names (e.g., 'daisy', 'rose')
    try:
        class_names = [d for d in os.listdir(original_train_dir) if os.path.isdir(os.path.join(original_train_dir, d))]
        if not class_names:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"Error: No class subdirectories found in '{original_train_dir}'.")
        print("Please make sure your structure is 'dataset/train/daisy', 'dataset/train/rose', etc.")
        return

    print(f"Found classes: {class_names}")

    # Process each class directory
    for class_name in class_names:
        original_class_dir = os.path.join(original_train_dir, class_name)
        
        # Create corresponding subdirectories in train_split and test_split
        new_train_class_dir = os.path.join(train_split_dir, class_name)
        new_test_class_dir = os.path.join(test_split_dir, class_name)
        os.makedirs(new_train_class_dir, exist_ok=True)
        os.makedirs(new_test_class_dir, exist_ok=True)

        # Get all image files for the current class
        all_files = [f for f in os.listdir(original_class_dir) if os.path.isfile(os.path.join(original_class_dir, f))]
        random.shuffle(all_files)

        # Determine the split index
        split_index = int(len(all_files) * split_ratio)

        # Split the files into training and testing lists
        train_files = all_files[:split_index]
        test_files = all_files[split_index:]

        # Copy files to the new directories
        print(f"  - Processing {class_name}: Copying {len(train_files)} to train_split and {len(test_files)} to test_split.")
        for file_name in train_files:
            shutil.copy(os.path.join(original_class_dir, file_name), new_train_class_dir)
        
        for file_name in test_files:
            shutil.copy(os.path.join(original_class_dir, file_name), new_test_class_dir)
            
    print("\n" + "="*50)
    print("Data splitting complete")
    print("You can now run 'finetune_vgg16.py'.")


if __name__ == '__main__':
    # Define the source and destination directories
    # Assumes the script is run from the 'Flowers' directory
    original_data_source = './dataset/train'
    output_destination = './dataset'
    
    split_data(original_data_source, output_destination)
