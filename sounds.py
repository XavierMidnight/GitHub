import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import os
import re
import time  # Add this import

if not torch.cuda.is_available():
    raise RuntimeError("GPU device not available")

device = "cuda"

# Replace 'your_token' with the actual token from Hugging Face
os.environ['HF_TOKEN'] = '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'

def generate_audio(prompt, seconds_total, model, model_config, device, output_dir="./"):
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": seconds_total
    }]

    base_filename = re.sub(r'[\\/*?:"<>|]', "_", prompt.replace(' ', '_'))
    counter = 1
    filename = os.path.join(output_dir, f"{base_filename}.wav")
    while os.path.exists(filename):
        filename = os.path.join(output_dir, f"{base_filename}_{counter}.wav")
        counter += 1

    output = generate_diffusion_cond(
        model,
        steps=50,
        cfg_scale=6,
        conditioning=conditioning,
        sample_size=model_config["sample_size"],
        sigma_min=0.3,
        sigma_max=50,
        sampler_type="dpmpp-2m-sde",
        device=device
    )

    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save(filename, output, model_config["sample_rate"])

    print(f"Audio file generated: {filename}")
    return filename

if __name__ == "__main__":
    start_time = time.time()  # Record the start time

    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Array of prompts
    prompts = [
        "88 BPM fireworks explode overhead",
        "88 BPM fireworks explode overhead"
    ]

    seconds_total = 47
    generated_files = []

    # Loop through prompts and generate audio for each
    for prompt in prompts:
        generated_file = generate_audio(prompt, seconds_total, model, model_config, device)
        generated_files.append(generated_file)

    print("\nAll audio files generated:")
    for file in generated_files:
        print(file)

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time  # Calculate the total time

    print(f"\nTotal execution time: {total_time:.2f} seconds")  # Print the total time