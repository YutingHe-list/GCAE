import os
import numpy as np
import cv2
from typing import Optional

try:
	from diffusers import StableDiffusionControlNetPipeline  # type: ignore
	diffusers_available = True
except Exception:
	diffusers_available = False

try:
	from PIL import Image
except Exception:
	Image = None  # type: ignore

try:
	import torch
except Exception:
	torch = None  # type: ignore


class VascularDataEngine:
	def __init__(self, use_diffusers: bool = False, device: str = "cpu"):
		self.use_diffusers = use_diffusers and diffusers_available and (Image is not None) and (torch is not None)
		self.device = device
		self.pipe = None
		if self.use_diffusers:
			self._init_diffusers()

	def _init_diffusers(self):
		try:
			self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
				"lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
			)
			self.pipe = self.pipe.to(self.device)
		except Exception:
			self.pipe = None
			self.use_diffusers = False

	def generate(self, mask: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
		mask = (mask > 0).astype(np.uint8) * 255
		if self.use_diffusers and self.pipe is not None:
			try:
				mh, mw = mask.shape
				ctrl = np.stack([mask, mask, mask], axis=-1)
				ctrl_img = Image.fromarray(ctrl)
				prompt = (
					"high-fidelity X-ray coronary angiography image, clear vessels, clinical style"
				)
				neg_prompt = "text, watermark, artifacts, color banding"
				generator = None
				if seed is not None and torch is not None:
					generator = torch.Generator(device=self.device)
					generator = generator.manual_seed(int(seed))
				result = self.pipe(
					prompt=prompt,
					image=ctrl_img,
					negative_prompt=neg_prompt,
					guidance_scale=7.5,
					num_inference_steps=20,
					generator=generator,
				)
				img = result.images[0]
				img = img.resize((mw, mh))
				img_np = np.array(img)
				if img_np.ndim == 3:
					img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
				return img_np.astype(np.uint8)
			except Exception:
				pass
		return self._classical(mask, seed)

	def _classical(self, mask: np.ndarray, seed: Optional[int]) -> np.ndarray:
		rng = np.random.RandomState(seed if seed is not None else 0)
		h, w = mask.shape
		bg = np.zeros((h, w), np.float32)
		for freq, amp in [(64, 0.6), (32, 0.3), (16, 0.1)]:
			n = rng.rand(max(1, h // freq) + 2, max(1, w // freq) + 2).astype(np.float32)
			n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
			bg += amp * n
		bg = cv2.GaussianBlur(bg, (0, 0), 3)
		bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-6)
		bg = (bg * 180 + 20).astype(np.uint8)
		vessel = cv2.GaussianBlur(mask, (0, 0), 1.0)
		img = bg.copy().astype(np.int16)
		img += (vessel > 0).astype(np.int16) * rng.randint(40, 70)
		img = np.clip(img, 0, 255).astype(np.uint8)
		y = np.linspace(-1, 1, h)[:, None]
		x = np.linspace(-1, 1, w)[None, :]
		r = np.sqrt(x * x + y * y)
		vignette = 1.0 - 0.2 * (r ** 2)
		img = (img.astype(np.float32) * vignette).astype(np.uint8)
		noise = (rng.randn(h, w) * 4).astype(np.int16)
		img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
		return img
