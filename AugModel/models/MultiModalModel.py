import torch
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from PIL import Image
import random


class MultiModalModel:
    def __init__(self, device="cuda"):
        self.device = device
        self._init_classifier()
        self._init_caption_model()
        self._init_text_model()

    def _init_classifier(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/bart-large-mnli-yahoo-answers",
            device=self.device,
        )

    def _init_caption_model(self):
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large",
        )
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)

    def _init_text_model(self):
        model_name = "Qwen/Qwen3-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map=self.device
        )

    def select_object(self, image_desc: str, candidates: list, base_class: str) -> str:
        objects_list = [
            "human",
            "land animal",  # наземные животные
            "water animal",  # водные животные
            "air animal",  # воздушные животные
            "plant",
            "food",
            "tool",
            "device",
            "building",
            "land vehicle",  # наземный транспорт
            "water vehicle",  # водный транспорт
            "air vehicle",  # воздушный транспорт
            "nature",
            "abstraction",
        ]

        new_type_class = self.classifier(
            base_class, objects_list, hypothesis_template="This is a {}."
        )
        modified_prompt = image_desc.replace(base_class, "object")
        result = self.classifier(
            f"What {new_type_class} is similar to a {base_class}?",
            candidates,
            multi_label=True,
            hypothesis_template="This is a {}.",
        )
        labels = result["labels"]
        scores = result["scores"]
        high_conf_labels = [
            label for label, score in zip(labels, scores) if score > 0.4
        ]
        if not high_conf_labels:
            return labels[scores.index(max(scores))], modified_prompt
        return random.choice(high_conf_labels), modified_prompt

    def generate_image_caption(self, image, context: str = None) -> str:
        if context:
            inputs = self.processor(image, context, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        out = self.caption_model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def generate_expanded_prompt(self, new_object: str, image_desc: str) -> str:
        prompt = (
            f"USER: Write a concise, realistic visual description of a {new_object} in less than 20 words in that scene: {image_desc}."
            f" Don't include background or other objects. Use descriptive terms only about the {new_object}."
            f" Then append style comments like: '4k, ultra HD, highly detailed, realistic lighting'."
            f" ASSISTANT:"
        )

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.text_model.generate(**model_inputs, max_new_tokens=100)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return content
