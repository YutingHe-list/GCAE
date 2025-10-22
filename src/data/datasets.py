import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SegDataset(Dataset):
	def __init__(self, img_dir: str, mask_dir: str, img_size: int = 512):
		self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
		self.mask_dir = mask_dir
		self.img_size = img_size

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, idx):
		p = self.img_paths[idx]
		name = os.path.basename(p)
		img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
		mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (self.img_size, self.img_size))
		mask = cv2.resize(mask, (self.img_size, self.img_size))
		img = (img.astype(np.float32) / 255.0)[None]
		mask = (mask.astype(np.float32) / 255.0)[None]
		return torch.from_numpy(img), torch.from_numpy(mask)


class DetDataset(Dataset):
	def __init__(self, img_dir: str, label_dir: str, img_size: int = 512, stride: int = 8):
		self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
		self.label_dir = label_dir
		self.img_size = img_size
		self.stride = stride

	def __len__(self):
		return len(self.img_paths)

	def _read_yolo_labels(self, txt_path: str):
		boxes = []
		if not os.path.exists(txt_path):
			return boxes
		with open(txt_path, "r") as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) != 5:
					continue
				_, cx, cy, w, h = map(float, parts)
				boxes.append([cx, cy, w, h])
		return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

	def __getitem__(self, idx):
		p = self.img_paths[idx]
		name = os.path.basename(p)
		lbl = os.path.splitext(name)[0] + ".txt"
		img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (self.img_size, self.img_size))
		H, W = self.img_size, self.img_size
		Hs, Ws = H // self.stride, W // self.stride
		obj = np.zeros((1, Hs, Ws), dtype=np.float32)
		box = np.zeros((4, Hs, Ws), dtype=np.float32)
		boxes = self._read_yolo_labels(os.path.join(self.label_dir, lbl))
		for (cx, cy, w, h) in boxes:
			xg = min(Ws - 1, max(0, int(cx * Ws)))
			yg = min(Hs - 1, max(0, int(cy * Hs)))
			obj[0, yg, xg] = 1.0
			box[:, yg, xg] = np.array([cx, cy, w, h], dtype=np.float32)
		img = (img.astype(np.float32) / 255.0)[None]
		return torch.from_numpy(img), {"obj": torch.from_numpy(obj), "box": torch.from_numpy(box)}


def seg_collate(batch):
	imgs, masks = zip(*batch)
	return torch.stack(imgs, 0), torch.stack(masks, 0)


def det_collate(batch):
	imgs, tgts = zip(*batch)
	imgs = torch.stack(imgs, 0)
	obj = torch.stack([t["obj"] for t in tgts], 0)
	box = torch.stack([t["box"] for t in tgts], 0)
	return imgs, {"obj": obj, "box": box}
