import os
import argparse
import yaml
import cv2
import numpy as np
from tqdm import tqdm

from src.ssa import generate_stenotic_mask
from src.engine import VascularDataEngine


def ensure_dir(p):
	os.makedirs(p, exist_ok=True)


def main(cfg):
	seg_img_dir = cfg["paths"]["seg_images"]
	seg_mask_dir = cfg["paths"]["seg_masks"]
	out_dir = cfg["paths"]["virtual_out"]
	use_diffusers = cfg["engine"].get("use_diffusers", False)
	num_per_mask = int(cfg["engine"].get("num_virtual_per_mask", 2))
	ssa_len = int(cfg["ssa"].get("segment_length_px", 40))
	ssa_sev = float(cfg["ssa"].get("severity", 0.5))

	img_names = [f for f in os.listdir(seg_mask_dir) if f.lower().endswith(".png")]
	engine = VascularDataEngine(use_diffusers=use_diffusers)

	out_img_dir = os.path.join(out_dir, "images")
	out_mask_dir = os.path.join(out_dir, "masks")
	ensure_dir(out_img_dir)
	ensure_dir(out_mask_dir)

	for name in tqdm(img_names, desc="Generating virtual data"):
		mask = cv2.imread(os.path.join(seg_mask_dir, name), cv2.IMREAD_GRAYSCALE)
		if mask is None:
			continue
		for k in range(num_per_mask):
			seed = (hash(name) + k) % (2**31 - 1)
			sten_mask = generate_stenotic_mask(mask, length_px=ssa_len, severity=ssa_sev, seed=seed)
			img = engine.generate(sten_mask, seed=seed)
			base = os.path.splitext(name)[0]
			cv2.imwrite(os.path.join(out_img_dir, f"{base}_v{k}.png"), img)
			cv2.imwrite(os.path.join(out_mask_dir, f"{base}_v{k}.png"), sten_mask)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True)
	args = parser.parse_args()
	with open(args.config, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	main(cfg)
