import torch
from diffusers import StableDiffusionXLPipeline
from IPython.display import display
import os

# --- CONFIGURATION ---
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH = "./sdxl-lora-faces"  # The output folder from training

# --- AUTOMATIC DEVICE DETECTION ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
    print(" NVIDIA GPU Detected. Running in Fast Mode.")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
    print(" No NVIDIA GPU detected (You are likely on TPU or CPU runtime).")
    print("   Running on CPU. This will be slow (approx 5-10 mins per image).")
    print("   RECOMMENDATION: Go to Runtime > Change runtime type > Select T4 GPU.")


def load_pipeline():
    print(f"Loading SDXL Base on {DEVICE}...")

    # Load pipeline
    # Note: We use use_safetensors=True for faster loading
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        use_safetensors=True
    ).to(DEVICE)

    print(f"Loading LoRA from {LORA_PATH}...")
    if os.path.exists(LORA_PATH):
        try:
            pipe.load_lora_weights(LORA_PATH)
            pipe.fuse_lora()  # Merge weights for faster inference
            print("LoRA Loaded and Fused.")
        except Exception as e:
            print(f" Error loading LoRA weights: {e}")
            print("   (Continuing with base model only...)")
    else:
        print(f" LoRA path '{LORA_PATH}' not found. Generating with base model only.")

    return pipe


def generate_actor(pipe, actor_name, prompt_suffix):
    # Construct prompt using the trigger words
    prompt = f"a photo of sks {actor_name} person, {prompt_suffix}, 8k, highly detailed, photorealistic"
    negative_prompt = "cartoon, anime, 3d, painting, disfigured, bad anatomy, blurry"

    print(f"\nGenerating: '{prompt}'")

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    display(image)
    filename = f"generated_{actor_name.replace(' ', '_')}.png"
    image.save(filename)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    pipe = load_pipeline()
    if pipe:
        # Example: Generate one of your actors
        # Replace 'Brad Pitt' with one of the folder names from your original dataset
        actor = "Brad Pitt"
        scenario = "wearing a tuxedo, red carpet event, flashing cameras"

        generate_actor(pipe, actor, scenario)