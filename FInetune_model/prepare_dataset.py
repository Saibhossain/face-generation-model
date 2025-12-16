import os
from PIL import Image
from tqdm import tqdm
import shutil

INPUT_DIR = "/content/drive/MyDrive/Datasets/Celebrity faces"

# Output: Where the clean, captioned training data will go
OUTPUT_DIR = "./training_data"

# SDXL native resolution
TARGET_SIZE = 1024

def prepare_data():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"Processing dataset from {INPUT_DIR}...")

    total_images = 0

    # Walk through actor folders
    for actor_name in os.listdir(INPUT_DIR):
        actor_path = os.path.join(INPUT_DIR, actor_name)
        if not os.path.isdir(actor_path):
            continue

        print(f"Processing Actor: {actor_name}")

        # Clean actor name for the prompt (remove underscores)
        # This becomes the "Trigger Word" for the model
        # e.g., "Brad_Pitt" -> "Brad Pitt"
        clean_name = actor_name.replace("_", " ")

        # Define the caption/trigger prompt
        # We use a rare token 'sks' + the name to help the model learn specific features
        caption = f"a photo of sks {clean_name} person, high quality, 8k uhd"

        for filename in os.listdir(actor_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue

            try:
                # 1. Load Image
                img_path = os.path.join(actor_path, filename)
                img = Image.open(img_path).convert("RGB")

                # 2. Smart Resize (Aspect Ratio Preserving or Center Crop)
                # For simplicity here, we center crop to square then resize
                # SDXL handles different aspect ratios, but square is safest for beginners
                w, h = img.size
                min_dim = min(w, h)
                left = (w - min_dim)/2
                top = (h - min_dim)/2
                right = (w + min_dim)/2
                bottom = (h + min_dim)/2

                img = img.crop((left, top, right, bottom))
                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)

                # 3. Save Image
                # We flatten the folder structure: "Brad_Pitt_001.jpg"
                new_filename = f"{actor_name}_{total_images:04d}.jpg"
                save_path = os.path.join(OUTPUT_DIR, new_filename)
                img.save(save_path, quality=95)

                # 4. Save Caption Text File
                # SDXL training scripts look for a .txt file with the same name as the image
                caption_path = os.path.join(OUTPUT_DIR, new_filename.replace(".jpg", ".txt"))
                with open(caption_path, "w") as f:
                    f.write(caption)

                total_images += 1

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nDataset Preparation Complete.")
    print(f"Total Images: {total_images}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("Each image has a matching .txt file with the prompt.")

if __name__ == "__main__":
    prepare_data()