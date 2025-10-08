import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Load pretrained model   model = model.to_empty(device=desired_device)

model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()
