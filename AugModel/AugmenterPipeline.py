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
    A class that performs image augmentation by replacing objects in images.
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
        current_object: str = None,
        new_object: str = None,
        mask: Image.Image = None,
        prompt: str = None,
        candidates: List[str] = None,
        alpha_clip_threshold: float = 0.2,
        ddim_steps: int = 50,
        guidance_scale: int = 5,
        seed: int = 1,
    ) -> Tuple[Image.Image, Optional[Tuple[str, str]]]:
        """
        Replaces the specified object in the given image with a new one.

        Args:
        image (Image.Image): The input image.
        mask (Image.Image): The mask of the object to replace.
        current_object (str): The name of the object to be replaced.
        new_objects_list (Optional[List[str]]): A list of potential new objects. If None, the method will generate a new object.
        ddim_steps (int): The number of denoising steps. More steps mean a slower but potentially higher quality result.
        guidance_scale (int): The scale for classifier-free guidance. Higher values lead to results that are more closely linked to the text prompt.
        seed (int): Integer value that initializes the random number generator for reproducibility.
        return_prompt (bool): If True, the method also returns the prompt used for generation and the new object.

        Returns:
        Tuple[Image.Image, Optional[Tuple[str, str]]]: The modified image and, optionally, the prompt used for generation and the new object.
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
