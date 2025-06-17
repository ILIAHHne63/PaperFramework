import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class YoloModel:
    def __init__(self, model_path="yolo11n.pt", device="cuda"):
        """Инициализация модели YOLO для сегментации"""
        self.model = YOLO(model_path).to(device)

    def __call__(self, image):
        """Обработка изображения и возврат маски самого большого объекта в формате PIL.Image

        Args:
            image_path (str): Путь к входному изображению

        Returns:
            PIL.Image.Image or None: Маска объекта в формате изображения или None
        """
        results = self.model(image)
        result = results[0]

        if len(result.boxes) == 0:
            raise ValueError("Yolo can't find bbox")

        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        max_idx = np.argmax(areas)

        class_id = int(result.boxes.cls[max_idx].item())
        class_name = self.model.names[int(class_id)]

        bbox = boxes[max_idx]
        x1, y1, x2, y2 = bbox.astype(int)

        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            width, height = image.size
            image = np.array(image)

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        mask_pil = Image.fromarray(mask)

        return mask_pil, class_name, bbox.astype(int)
