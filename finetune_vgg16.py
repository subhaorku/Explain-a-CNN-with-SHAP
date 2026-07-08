# This script fine-tunes a pre-trained VGG-16 model on a flower dataset.
# It expects the data to be pre-split into 'train_split' and 'test_split' folders.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import argparse
from tqdm import tqdm

def train_and_evaluate(data_dir, num_epochs, batch_size, learning_rate):
    """
    Main function to handle data loading, model setup, training, and evaluation.
    """
    
    # --- 1. Data Loading and Preprocessing ---
    print("="*50)
    print("Step 1: Preparing DataLoaders from split folders...")

    # Define paths for the new split train and test directories
    train_dir = os.path.join(data_dir, 'train_split')
    test_dir = os.path.join(data_dir, 'test_split') 

    # Check if the required split directories exist
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print(f"Error: 'train_split' or 'test_split' directory not found in '{data_dir}'.")
        print("Please run the 'create_split_folders.py' script first to generate these folders.")
        return
    
    # VGG-16 expects 224x224 images and specific normalization.
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load the datasets from the split folders using ImageFolder
    train_image_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    test_image_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

    train_loader = torch.utils.data.DataLoader(train_image_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_image_dataset, batch_size=batch_size, num_workers=4)
    
    class_names = train_image_dataset.classes
    num_classes = len(class_names)
    
    print(f"Dataset contains {num_classes} classes: {class_names}")
    print(f"Training set has {len(train_image_dataset)} images.")
    print(f"Test set has {len(test_image_dataset)} images.")

    # --- 2. Model Setup ---
    print("\n" + "="*50)
    print("Step 2: Setting up VGG-16 for fine-tuning...")

    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False
    
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    # --- 3. Training Preparation ---
    print("\n" + "="*50)
    print("Step 3: Starting the training process...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 4. Training Loop ---
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # -- Training Phase --
        model.train()
        running_loss = 0.0
        correct_train = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': running_loss / ((train_pbar.n + 1) * batch_size)})

        train_loss = running_loss / len(train_image_dataset)
        train_acc = 100 * correct_train / len(train_image_dataset)

        # -- Testing Phase --
        model.eval()
        running_loss = 0.0
        correct_test = 0

        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="Testing ")
            for inputs, labels in test_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_test += (predicted == labels).sum().item()
                
                test_pbar.set_postfix({'loss': running_loss / ((test_pbar.n + 1) * batch_size)})

        test_loss = running_loss / len(test_image_dataset)
        test_acc = 100 * correct_test / len(test_image_dataset)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

        # -- Save Checkpoint --
        checkpoint_path = os.path.join(checkpoint_dir, f'vgg16_flowers_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  --> Checkpoint saved to '{checkpoint_path}'")

    print("\n" + "="*50)
    print("Training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune VGG-16 for flower classification.")
    
    parser.add_argument('--data_dir', type=str, default='./dataset', 
                        help="Path to the parent directory containing 'train_split' and 'test_split'. Default: ./dataset")
                        
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs (default: 10)')
                        
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training and testing (default: 32)')
                        
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate for the optimizer (default: 0.001)')

    args = parser.parse_args()

    train_and_evaluate(args.data_dir, args.epochs, args.batch_size, args.learning_rate)

