import torch
import torch.nn as nn
from torchvision import datasets,models, transforms
from PIL import Image


#Preprocess
# Standard transforms for VGG16 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#getting the number of classes
train_dir = './dataset/train_split'
train_image_dataset = datasets.ImageFolder(train_dir, transform=transforms)
class_names = train_image_dataset.classes
num_classes = len(class_names)

# setting up the model with the correct number of output classes
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, num_classes)

# Load the finetuned weights
state_dict=torch.load("./checkpoints/vgg16_flowers_epoch_100.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

#Preprocess image
image = Image.open("./dataset/test_split/dandelion/3998275481_651205e02d.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

#Inference
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

class_names = ['setosa', 'versicolor', 'virginica']
print(f"Predicted class: {class_names[predicted.item()]}")