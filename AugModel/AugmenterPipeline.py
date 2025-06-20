import torch
import numpy as np
import random
from PIL import Image
from typing import List, Optional, Tuple
from accelerate import Accelerator

from .models import FluxModel
from .models import YoloModel
from .models import AlphaCLIPModel
from .models import MultiModalModel


class Augmenter:
    """
    A class for replacing objects in images using a pipeline of:
    1. Object detection (YOLO)
    2. Caption generation (MultiModalModel)
    3. Object replacement (FluxModel)
    4. Quality control (AlphaCLIP)
    """


    def __init__(self, device: str = "cuda"):
        """
        Initializes the Augmenter class.

        Args:
        device (str): The device to use for computations. Defaults to "cuda".
        """
        self.device = device
        self.accelerator = Accelerator()

        self._models = {
            "Flux": FluxModel(device=self.device),
            "Yolo": YoloModel(),
            "AlphaCLIP": AlphaCLIPModel(device=self.device),
            "MultiModal": MultiModalModel(device=self.device),
        }

    def _set_seed(self, seed: int) -> None:
        """
        Sets the seed for the random number generators.

        Args:
        seed (int): The seed to use.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def to(self, device):
        """
        Moves the model to the specified device.

        Args:
        device (torch.device): The device on which the model will run.
        """
        self._models["Flux"].to(device)
        self._models["AlphaCLIP"].to(device)
        self._models["Yolo"].to(device)
        self._models["MultiModal"].to(device)
        self.device = device

    def __call__(
        self,
        image: Image.Image,
        current_object: Optional[str] = None,
        new_object: Optional[str] = None,
        mask: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        candidates: Optional[List[str]] = None,
        alpha_clip_threshold: float = 0.2,
        ddim_steps: int = 50,
        guidance_scale: int = 5,
        seed: int = 1,
    ) -> Tuple[Image.Image, Optional[Tuple[str, str]]]:
        """
        Replaces an object in an image and returns the augmented image with metadata.
        
        Pipeline:
        1. Detect object and get mask (if not provided)
        2. Generate caption and replacement prompt
        3. Perform inpainting with diffusion model
        4. Verify result quality with AlphaCLIP
        
        Args:
            image: Input PIL image (RGB recommended)
            current_object: Object to replace (None for auto-detection)
            new_object: Object to insert (None for auto-selection)
            mask: Binary mask of object to replace (None for auto-detection)
            prompt: Custom inpainting prompt (None for auto-generation)
            candidates: Candidate objects for replacement
            alpha_clip_threshold: Minimum AlphaCLIP score threshold
            ddim_steps: Diffusion denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed
            
        Returns:
            augmented_image: Resulting PIL image
            prompt_used: Final prompt used for inpainting
            clip_score: AlphaCLIP quality score
            bbox: Detected bounding box (xmin, ymin, xmax, ymax) or None
            new_object_used: Object used for replacement
        """
        self._set_seed(seed)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if mask is None:

            if current_object is None:
                mask, current_object, bbox = self._models["Yolo"](image)

            else:
                mask, _, bbox = self._models["Yolo"](image)

        if mask.mode != "L":
            mask = mask.convert("L")

        if prompt is None:
            image_description = self._models["MultiModal"].generate_image_caption(
                image, current_object
            )
            if candidates is None:
                candidates = ["pizza", "apple", "cigarettes"]
            new_object, image_description_filtred = self._models[
                "MultiModal"
            ].select_object(image_description, candidates, current_object)
            prompt = self._models["MultiModal"].generate_expanded_prompt(
                new_object, image_description_filtred
            )

        result = self._models["Flux"](
            prompt=prompt,
            image=image,
            mask=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=ddim_steps,
            max_sequence_length=512,
            seed=seed,
        )

        threshold = self._models["AlphaCLIP"](result, mask, prompt)

        if threshold < alpha_clip_threshold:
            print("This generation does not meet the specified threshold")

        return result, prompt, threshold, bbox
