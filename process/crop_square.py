import cv2
import numpy as np
import os
src_folder = r"D:\retina train\diabetic-retinopathy-detection\test\test"
dst_folder = r"C:\Users\STIC-11\Desktop\sk1\test"
os.makedirs(dst_folder, exist_ok=True)
failed_images = []
successful_images = []
def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return False

    print(f"Original image shape: {img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in: {img_path}")
        return False
    largest_contour = max(contours, key=cv2.contourArea)
    (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
    center_x, center_y, radius = int(center_x), int(center_y), int(radius)
    half_side = int(radius / np.sqrt(2))
    print(f"Center: ({center_x}, {center_y}), Radius: {radius}, Half side: {half_side}")
    x1 = center_x - half_side
    y1 = center_y - half_side
    x2 = center_x + half_side
    y2 = center_y + half_side
    height, width = img.shape[:2]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, width)
    y2 = min(y2, height)
    print(f"Crop coordinates: ({x1}, {y1}) to ({x2}, {y2})")
    if x2 <= x1 or y2 <= y1:
        print(f"Invalid crop coordinates for: {img_path}")
        return False

    inscribed_square = img[y1:y2, x1:x2]
    
    if inscribed_square.size == 0:
        print(f"Resulting crop is empty for: {img_path}")
        return False

    print(f"Cropped image shape: {inscribed_square.shape}")
    return inscribed_square
total_images = 0
for filename in os.listdir(src_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        total_images += 1
        input_path = os.path.join(src_folder, filename)
        output_path = os.path.join(dst_folder, filename)
        
        print(f"\nProcessing: {filename}")
        try:
            result = process_image(input_path)
            
            if result is not False and result.size > 0:
                try:
                    cv2.imwrite(output_path, result)
                    print(f"Successfully saved: {filename}")
                    successful_images.append(filename)
                except Exception as e:
                    print(f"Error saving {filename}: {str(e)}")
                    failed_images.append((filename, f"Save error: {str(e)}"))
            else:
                print(f"Failed to process: {filename}")
                failed_images.append((filename, "Processing failed"))
        except Exception as e:
            print(f"Unexpected error with {filename}: {str(e)}")
            failed_images.append((filename, f"Unexpected error: {str(e)}"))

print("\n" + "="*50)
print("PROCESSING SUMMARY")
print("="*50)
print(f"Total images processed: {total_images}")
print(f"Successfully processed: {len(successful_images)}")
print(f"Failed images: {len(failed_images)}")

if failed_images:
    print("\nFAILED IMAGES:")
    print("-"*50)
    for img, error in failed_images:
        print(f"• {img}: {error}")

with open(os.path.join(dst_folder, 'failed_images.txt'), 'w') as f:
    f.write("Failed Images List:\n")
    f.write("-"*50 + "\n")
    for img, error in failed_images:
        f.write(f"{img}: {error}\n")

print("\nProcessing complete!")
print(f"Failed images list saved to: {os.path.join(dst_folder, 'failed_images.txt')}")