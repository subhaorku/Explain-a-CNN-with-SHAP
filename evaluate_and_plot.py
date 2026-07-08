# This script loads the saved model checkpoints from each epoch,
# evaluates their performance on the train and test sets,
# and plots the loss and accuracy curves.

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import argparse
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on a given dataset.

    Returns:
        Tuple: (average loss, accuracy percentage)
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # No need to calculate gradients
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = 100 * correct_predictions / total_samples
    return epoch_loss, epoch_acc

def main(data_dir, checkpoints_dir, total_epochs):
    """
    Main function to load checkpoints, evaluate, and plot results.
    """
    
    # --- 1. Data Loading (using test transforms for consistency) ---
    print("="*50)
    print("Step 1: Preparing DataLoaders for evaluation...")

    # For evaluation, we don't need random crops or flips.
    # We use the 'test' transform for both sets.
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(data_dir, 'train_split')
    test_dir = os.path.join(data_dir, 'test_split')

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print(f"Error: 'train_split' or 'test_split' not found in '{data_dir}'.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    # Use a larger batch size for faster evaluation
    eval_batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size, num_workers=4)
    
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes.")

    # --- 2. Model and Evaluation Setup ---
    print("\n" + "="*50)
    print("Step 2: Setting up VGG-16 model...")

    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # --- 3. Loop Through Checkpoints and Evaluate ---
    print("\n" + "="*50)
    print("Step 3: Evaluating all checkpoints...")
    
    history = []

    # Use tqdm for a progress bar over the epochs
    epoch_pbar = tqdm(range(1, total_epochs + 1), desc="Evaluating Epochs")
    for epoch in epoch_pbar:
        checkpoint_path = os.path.join(checkpoints_dir, f'vgg16_flowers_epoch_{epoch}.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found for epoch {epoch}. Skipping.")
            continue
            
        # Load the state dict from the file
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # Evaluate on the training set
        train_loss, train_acc = evaluate_model(model, train_loader, criterion, device)
        
        # Evaluate on the test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        # Store the results
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })
        
        # Update progress bar description
        epoch_pbar.set_postfix({'Test Acc': f'{test_acc:.2f}%'})

    if not history:
        print("Error: No checkpoints were found or evaluated. Cannot create plots.")
        return
        
    # Convert history to a pandas DataFrame for easy plotting
    history_df = pd.DataFrame(history)

    # --- 4. Plotting Results ---
    print("\n" + "="*50)
    print("Step 4: Generating and saving plots...")

    sns.set_style("whitegrid")

    # Plot 1: Train and Test Loss vs. Epochs
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=history_df, x='epoch', y='train_loss', label='Train Loss', color='blue')
    sns.lineplot(data=history_df, x='epoch', y='test_loss', label='Test Loss', color='orange')
    plt.title('Training and Test Loss vs. Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.savefig('training_loss_plot.png', dpi=300)
    print("Saved 'training_loss_plot.png'")
    plt.show()

    # Plot 2: Train and Test Accuracy vs. Epochs
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=history_df, x='epoch', y='train_acc', label='Train Accuracy', color='blue')
    sns.lineplot(data=history_df, x='epoch', y='test_acc', label='Test Accuracy', color='orange')
    plt.title('Training and Test Accuracy vs. Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.savefig('training_accuracy_plot.png', dpi=300)
    print("Saved 'training_accuracy_plot.png'")
    plt.show()
    
    print("\n" + "="*50)
    print("Evaluation and plotting complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate VGG-16 checkpoints and plot learning curves.")
    
    parser.add_argument('--data_dir', type=str, default='./dataset', 
                        help="Path to the parent directory containing 'train_split' and 'test_split'.")
                        
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
                        help="Directory where model checkpoints are saved.")
                        
    parser.add_argument('--total_epochs', type=int, default=100, 
                        help='The total number of epochs that were trained.')

    args = parser.parse_args()

    main(args.data_dir, args.checkpoints_dir, args.total_epochs)
