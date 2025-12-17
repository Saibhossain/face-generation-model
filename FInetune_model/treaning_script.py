import os
import subprocess
import urllib.request
import torch

# Prefer MPS on macOS if available
if torch.backends.mps.is_available():
    torch.set_default_device("mps")
elif torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

# --- 2. CONFIGURATION ---
# Path to the dataset prepared by prepare_dataset.py
DATA_DIR = "./training_data"
OUTPUT_DIR = "./sdxl-lora-faces"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"

# --- 3. DOWNLOAD TRAINING SCRIPT ---
# We use the official script from Hugging Face Diffusers examples
if not os.path.exists("train_text_to_image_lora_sdxl.py"):
    print("Downloading official training script...")
    subprocess.run([
        "curl", "-o", "train_text_to_image_lora_sdxl.py",
        "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora_sdxl.py"
    ])
# --- 4. LAUNCH TRAINING ---
print(" Starting Training... check logs below.")

# These arguments are tuned for a T4/V100 GPU environment with ~500 images
cmd = [
    "accelerate", "launch", "train_text_to_image_lora_sdxl.py",
    f"--pretrained_model_name_or_path={MODEL_NAME}",
    f"--train_data_dir={DATA_DIR}",
    f"--output_dir={OUTPUT_DIR}",
    "--caption_column=text",
    "--resolution=51 2",          # ⬅️ Reduced!
    "--random_flip",
    "--train_batch_size=1",
    "--num_train_epochs=1",       # ⬅️ Test with 1 epoch first!
    "--checkpointing_steps=100",  # ⬅️ More frequent saves (optional)
    "--learning_rate=1e-4",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",
    "--gradient_checkpointing",
    "--seed=42",
    "--dataloader_num_workers=0", # ⬅️ macOS doesn’t handle workers well
    "--validation_prompt=",       # ⬅️ Disable validation
]

# Run the command
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Stream output so you can see progress
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())

# Check for errors
rc = process.poll()
if rc != 0:
    print(f" Training failed with exit code {rc}")
    print(process.stderr.read())
else:
    print(f" Training Complete! Model saved to {OUTPUT_DIR}")