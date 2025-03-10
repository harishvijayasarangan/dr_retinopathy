import os
import shutil
import pandas as pd
from tqdm import tqdm
labels_df = pd.read_csv(r'C:\Users\STIC-11\Desktop\sk1\csv\test_proliferate.csv')
src_dir = r"C:\Users\STIC-11\Desktop\sk1\data\square\test"
dst_dir = r"C:\Users\STIC-11\Desktop\sk1\data\square\train"
os.makedirs(dst_dir, exist_ok=True)
image_names = labels_df['image'].values
successful_copies = []
failed_copies = []
missing_in_source = []
for image_name in tqdm(image_names, desc="Copying images", unit="image"):
    src_path = os.path.join(src_dir, f"{image_name}.jpeg")
    dst_path = os.path.join(dst_dir, f"{image_name}.jpeg")
    
    try:
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            successful_copies.append(image_name)
        else:
            failed_copies.append(image_name)
            missing_in_source.append(image_name)
            tqdm.write(f"Warning: Image not found - {image_name}.jpeg")
    except Exception as e:
        failed_copies.append(image_name)
        tqdm.write(f"Error copying {image_name}.jpeg: {str(e)}")

print("\nCopy Summary:")
print(f"Total images in CSV: {len(image_names)}")
print(f"Successfully copied: {len(successful_copies)}")
print(f"Failed to copy: {len(failed_copies)}")
print(f"\nMissing Images Summary:")
print(f"Images in CSV but missing from source: {len(missing_in_source)}")

if missing_in_source:
    print("\nList of missing images:")
    for img in missing_in_source:
        print(f"{img}.jpeg")

