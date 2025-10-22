import os
import argparse
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.model.yolounet import YOLOUNet
from src.data.datasets import SegDataset, DetDataset, seg_collate, det_collate
from src.utils.losses import SegCriterion, DetCriterion


def main(cfg):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	img_size = int(cfg["training"]["img_size"])
	batch_size = int(cfg["training"]["batch_size"])
	epochs = int(cfg["training"]["epochs"])
	lr = float(cfg["training"]["lr"])
	wd = float(cfg["training"]["weight_decay"])
	nw = int(cfg["training"]["num_workers"])
	bce_w = float(cfg["training"]["seg_loss_weights"]["bce"]) 
	dice_w = float(cfg["training"]["seg_loss_weights"]["dice"]) 
	obj_w = float(cfg["training"]["det_loss_weights"]["obj"]) 
	box_w = float(cfg["training"]["det_loss_weights"]["box"]) 

	seg_ds = SegDataset(cfg["paths"]["seg_images"], cfg["paths"]["seg_masks"], img_size)
	det_ds = DetDataset(cfg["paths"]["det_images"], cfg["paths"]["det_labels"], img_size)
	seg_dl = DataLoader(seg_ds, batch_size=batch_size, shuffle=True, num_workers=nw, collate_fn=seg_collate)
	det_dl = DataLoader(det_ds, batch_size=batch_size, shuffle=True, num_workers=nw, collate_fn=det_collate)
	seg_iter = iter(seg_dl)
	det_iter = iter(det_dl)

	model = YOLOUNet().to(device)
	opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
	seg_crit = SegCriterion(bce_w=bce_w, dice_w=dice_w)
	det_crit = DetCriterion(obj_w=obj_w, box_w=box_w)

	for epoch in range(epochs):
		model.train()
		tbar = tqdm(range(max(len(seg_dl), len(det_dl))), desc=f"Epoch {epoch+1}/{epochs}")
		for _ in tbar:
			try:
				imgs_s, masks_s = next(seg_iter)
			except StopIteration:
				seg_iter = iter(seg_dl)
				imgs_s, masks_s = next(seg_iter)
			try:
				imgs_d, tgts_d = next(det_iter)
			except StopIteration:
				det_iter = iter(det_dl)
				imgs_d, tgts_d = next(det_iter)

			imgs_s = imgs_s.to(device)
			masks_s = masks_s.to(device)
			imgs_d = imgs_d.to(device)
			for k in tgts_d:
				tgts_d[k] = tgts_d[k].to(device)

			opt.zero_grad(set_to_none=True)
			out_s = model(imgs_s)
			loss_seg = seg_crit(out_s["seg_logit"], masks_s)
			out_d = model(imgs_d)
			loss_det = det_crit(out_d["det_pred"], tgts_d)
			loss = loss_seg + loss_det
			loss.backward()
			opt.step()
			tbar.set_postfix({"loss": float(loss.item()), "seg": float(loss_seg.item()), "det": float(loss_det.item())})


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True)
	args = parser.parse_args()
	with open(args.config, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	main(cfg)
