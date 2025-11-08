# pip install diffusers transformers accelerate safetensors torch torchvision pillow opencv-python

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# === 1. Load Stable Diffusion (base model) ===
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# === 2. Load input floorplan ===
input_image = Image.open("walls.jpg").convert("RGB")

# === 3. Clean-only prompt (structure preservation) ===
prompt = (
    "clean architectural floorplan, same structure, thin black walls on white background, "
    "no text, no annotations, no added geometry"
)

# === 4. Run diffusion in low-strength mode ===
result = pipe(
    prompt=prompt,
    image=input_image,
    strength=0.25,       # low = preserves geometry, high = hallucinates
    guidance_scale=7.0,  # prompt adherence
    num_inference_steps=20
).images[0]

# === 5. Save output ===
result.save("walls_diffusion_refined.png")
print("✅ Saved: walls_diffusion_refined.png")


# # pip install diffusers transformers accelerate safetensors opencv-python pillow torch torchvision

# import torch
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
# from diffusers.utils import load_image
# import cv2

# # === 1. Load ControlNet (pretrained for line-art extraction) ===
# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16
# )
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
# ).to("cuda")

# # === 2. Input image ===
# input_path = "walls.jpg"
# img = load_image(input_path)

# # === 3. Prompt to guide denoising (keep lines, remove text) ===
# prompt = (
#     "clean architectural floorplan, black walls on white background, "
#     "no text, no annotations, no labels, no symbols"
# )

# # === 4. Denoise using diffusion model ===
# out = pipe(
#     prompt,
#     image=img,
#     num_inference_steps=20,
#     guidance_scale=7.0,
#     controlnet_conditioning_scale=0.8
# )

# # === 5. Save result ===
# out_image = out.images[0]
# out_image.save("walls_diffusion_clean.png")
# print("✅ Saved: walls_diffusion_clean.png")
