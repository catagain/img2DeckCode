import os
import json
import imagehash
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def generate_card_hashes():
    """
    Scans raw images and generates an enhanced database 
    containing both 16x16 pHash and HSV Color Histograms.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    image_folder = os.path.join(project_root, 'data', 'images')
    output_path = os.path.join(project_root, 'data', 'card_hashes.json')

    if not os.path.exists(image_folder):
        print(f"Error: Could not find image folder at: {image_folder}")
        return

    hash_db = {}
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"Warning: No images found in {image_folder}")
        return

    print(f"Processing {len(image_files)} images (Enhanced Feature Extraction)...")

    for filename in tqdm(image_files, desc="Hashing Cards"):
        try:
            card_id = os.path.splitext(filename)[0]
            img_path = os.path.join(image_folder, filename)
            
            # --- 1. Calculate Enhanced pHash (hash_size=16 for 256-bit precision) ---
            with Image.open(img_path) as img:
                img_rgb_pil = img.convert('RGB')
                # High-res hash captures more detail than the default 8x8
                h = imagehash.phash(img_rgb_pil, hash_size=16)
            
            # --- 2. Calculate Color Histogram (HSV Space) ---
            # HSV is more robust to lighting changes than RGB
            cv_img = cv2.imread(img_path)
            hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            
            # Use H (Hue) and S (Saturation) for color distribution
            # Channels: [0, 1], Bins: [12, 8], Ranges: H[0-180], S[0-256]
            hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist) # Normalize to 0-1 range
            
            hash_db[card_id] = {
                "phash": str(h),
                "hist": hist.flatten().tolist() # Store as flat list for JSON compatibility
            }
                
        except Exception as e:
            print(f"\n[Error] Failed to process {filename}: {e}")

    # Save results
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            # indent=None to keep the file size smaller
            json.dump(hash_db, f) 
        print(f"\nSuccess! Enhanced database saved to: {output_path}")
    except IOError as e:
        print(f"\n[Error] Failed to save database: {e}")

if __name__ == "__main__":
    generate_card_hashes()