#Install dependecies in the terminal
#pip install git+https://github.com/huggingface/diffusers
#pip install transformers accelerate safetensors

from diffusers import StableDiffusionXLPipeline
import torch
pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()
prompt = "Blue tuxedo Cat." # Your prompt here
neg_prompt = "ugly, blurry, poor quality" # Negative prompt here
image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]

image.save("output/image3.jpg")