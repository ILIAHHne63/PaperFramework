import torch
from diffusers import FluxFillPipeline


class FluxModel:
    def __init__(
        self,
        model_name="black-forest-labs/FLUX.1-Fill-dev",
        device="cuda",
        dtype=torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.pipe = FluxFillPipeline.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(self.device)

    def __call__(
        self,
        prompt,
        image,
        mask,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        seed=0,
    ):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        generated_image = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
        ).images[0]

        return generated_image
