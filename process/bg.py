import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
transform = transforms.Compose([
    transforms.ToTensor(), 
])
def crop_fundus_image(image_path):
    """
    Crop a fundus image to remove as much black background as possible using GPU acceleration.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        cropped_img (numpy array): Cropped image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading {image_path}")
        return None
    img_tensor = transform(img).to(device)  
    gray = torch.mean(img_tensor, dim=0, keepdim=True) 
    binary_mask = (gray > 0.04).float() 
    binary_np = binary_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f" No contours found in {image_path}")
        return img  
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img
def process_folder(input_folder, output_folder):
    """
    Process all images in the input folder, cropping each and saving in the output folder.
    
    Args:
        input_folder (str): Path to the folder containing images.
        output_folder (str): Path to the folder where cropped images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f" Processing {len(image_files)} images on {device}...\n")
    for image_file in tqdm(image_files, desc="⏳ Cropping images", unit="img"):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        cropped_img = crop_fundus_image(input_path)

        if cropped_img is not None:
            cv2.imwrite(output_path, cropped_img)

    print(f"\nProcessing complete. Cropped images saved to: {output_folder}")

input_folder = r"D:\retina train\diabetic-retinopathy-detection\test\test"  
output_folder = r"C:\Users\STIC-11\Desktop\sk1\processed\test" 

process_folder(input_folder, output_folder)
