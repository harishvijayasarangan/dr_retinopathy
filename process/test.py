import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import densenet121
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")
model = densenet121(weights=None)  
model.classifier = torch.nn.Linear(1024, 5)  
model_path = r"C:\Users\STIC-11\Desktop\sk1\checkpoints\best_model1.ckpt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f" Model file not found: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval() 
classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def predict(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)  
        probabilities = torch.nn.functional.softmax(outputs, dim=1) 
        predicted_class = outputs.argmax(dim=1).item()

    print("\n🔹 **Predictions:**")
    for i, prob in enumerate(probabilities[0]):
        print(f"{classes[i]}: {prob.item():.4f}")

    print(f"\n**Predicted Class:** {classes[predicted_class]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(" Usage: python test.py <image_path>")
    else:
        predict(sys.argv[1])
