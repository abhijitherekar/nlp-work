# text to image using stable diffusion hugging face
# pip install accelerate
# pip install safetensors
# pip install diffusers
# pip install xformers
import torch
import diffusers
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
#DPMSolverMultistepScheduler

def text_to_image(prompt):
    torch.cuda.empty_cache()
    print(" in text to image for prompt ", prompt)
    repo_id = "runwayml/stable-diffusion-v1-5"
   
    scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
    stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler, local_files_only=True)
    print("To CUDA")
    stable_diffusion.to("cuda")
    image = stable_diffusion(prompt, height=496, width=496, num_inference_steps=30).images[0]
    image.save('output.png')
    print(image)
    return image





    




