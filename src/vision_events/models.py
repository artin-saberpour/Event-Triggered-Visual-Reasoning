"""Optional CLIP, MediaPipe and LLaVA adapters, imported only when used."""

from __future__ import annotations

import numpy as np
from PIL import Image

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
# Pinned Hugging Face conversion-bot revision with a safetensors checkpoint.
# The checkpoint on the model's default branch is in the legacy .bin format.
CLIP_SAFE_REVISION = "aa8b329a670ccfc2a5170d7391ac3fca2e88595f"


class VisionModels:
    def __init__(self, device: str = "auto", use_vlm: bool = False) -> None:
        try:
            import cv2
            import mediapipe as mp
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU")
        self.cv2, self.torch = cv2, torch
        self.device = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, revision=CLIP_SAFE_REVISION)
        self.clip = CLIPModel.from_pretrained(
            CLIP_MODEL_ID, revision=CLIP_SAFE_REVISION, use_safetensors=True
        ).to(self.device).eval()
        self.vlm = self.vlm_processor = None
        if use_vlm:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            model_id = "llava-hf/llava-1.5-7b-hf"
            self.vlm_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.vlm_processor = AutoProcessor.from_pretrained(model_id)
            self.vlm = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=self.vlm_dtype).to(self.device).eval()

    def encode_frame(self, bgr: np.ndarray, mask_threshold: float = 0.5):
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        segmentation = self.segmenter.process(rgb).segmentation_mask
        if segmentation is None:
            raise RuntimeError("MediaPipe did not return a segmentation mask")
        mask = segmentation >= mask_threshold
        images = {
            "full": Image.fromarray(rgb),
            "person": Image.fromarray(np.where(mask[:, :, None], rgb, 0).astype(np.uint8)),
            "background": Image.fromarray(np.where(mask[:, :, None], 0, rgb).astype(np.uint8)),
        }
        inputs = self.clip_processor(images=list(images.values()), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            features = self.clip.get_image_features(**inputs).float().cpu().numpy()
        return images, dict(zip(images, features))

    def describe(self, image: Image.Image, question: str, max_new_tokens: int = 80) -> str:
        if self.vlm is None or self.vlm_processor is None:
            raise RuntimeError("The VLM was not enabled")
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.vlm_processor.apply_chat_template(conversation, add_generation_prompt=True)
        raw_inputs = self.vlm_processor(images=image, text=prompt, return_tensors="pt")
        inputs = {
            key: value.to(self.device, dtype=self.vlm_dtype) if value.is_floating_point() else value.to(self.device)
            for key, value in raw_inputs.items()
        }
        with self.torch.inference_mode():
            output = self.vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        return self.vlm_processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def close(self) -> None:
        self.segmenter.close()
