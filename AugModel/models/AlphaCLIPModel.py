import torch
from .AlphaCLIP import alpha_clip
from PIL import Image
import numpy as np
import os
from torchvision import transforms


class AlphaCLIPModel:
    """
    A class to compute the CLIPScore metric for evaluating the alignment between A PART OF THE generated image and a text prompt.
    """

    def __init__(
        self,
        weights="AugModel/models/checkpoints/clip_b16_grit1m_fultune_8xe.pth",
        device="cuda",
    ) -> None:
        """
        Initialize a MetricAlphaCLIPScore object with the specified device.

        Args:
            weights (str): The path to the model weights.
            device (str, optional): The device on which the model will run. Defaults to "cuda".
        """
        self.model, self.preprocess = alpha_clip.load(
            "ViT-B/16", alpha_vision_ckpt_pth=weights, device=device
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                # change to (336,336) when using ViT-L/14@336px
                transforms.Normalize(0.5, 0.26),
            ]
        )
        self.device = device

    def __call__(self, generated_image: Image.Image, mask: Image.Image, prompt: str):
        """
        Evaluate the alignment between the provided image and text prompt using the CLIPScore metric.

        Args:
            generated_image (Image.Image): The generated image for evaluation.
            mask (Image.Image): The binary mask on the generated image.
            prompt (str): The text prompt associated with the generated image.

        Returns:
            float: The computed CLIPScore.
        """
        image = generated_image.convert("RGB")
        mask = np.array(mask)
        # get `binary_mask` array (2-dimensional bool matrix)
        if len(mask.shape) == 2:
            binary_mask = mask == 255
        if len(mask.shape) == 3:
            binary_mask = mask[:, :, 0] == 255
        alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
        alpha = alpha.half().cuda().unsqueeze(dim=0)
        # calculate image and text features
        image = self.preprocess(image).unsqueeze(0).half().to(self.device)
        text = alpha_clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            image_features = self.model.visual(image, alpha)
            text_features = self.model.encode_text(text)
        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        clip_score = torch.matmul(image_features, text_features.T).item()
        return clip_score
