from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import torch

def text_to_video(prompt):
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    print("generating video for prompt --> ", prompt)
    #prompt = "Spiderman is surfing"
    video_frames = pipe(prompt, num_inference_steps=20).frames
    video_path = export_to_video(video_frames, "/home/abhi/nlp-project/tldr_video.mp4")
    return video_path