import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	probs = torch.sigmoid(logits)
	probs = probs.view(probs.size(0), -1)
	targets = targets.view(targets.size(0), -1)
	inter = (probs * targets).sum(dim=1)
	union = probs.sum(dim=1) + targets.sum(dim=1)
	dice = (2 * inter + eps) / (union + eps)
	return 1 - dice.mean()


class SegCriterion(nn.Module):
	def __init__(self, bce_w: float = 0.5, dice_w: float = 0.5):
		super().__init__()
		self.bce = nn.BCEWithLogitsLoss()
		self.bce_w = bce_w
		self.dice_w = dice_w

	def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
		return self.bce_w * self.bce(logits, masks) + self.dice_w * dice_loss(logits, masks)


def iou_loss(pred_boxes: torch.Tensor, tgt_boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	def to_xyxy(b):
		cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
		x1 = cx - w / 2
		y1 = cy - h / 2
		x2 = cx + w / 2
		y2 = cy + h / 2
		return x1, y1, x2, y2

	x1, y1, x2, y2 = to_xyxy(pred_boxes)
	x1g, y1g, x2g, y2g = to_xyxy(tgt_boxes)
	ix1 = torch.max(x1, x1g)
	iy1 = torch.max(y1, y1g)
	ix2 = torch.min(x2, x2g)
	iy2 = torch.min(y2, y2g)
	inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
	area_p = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
	area_g = torch.clamp(x2g - x1g, min=0) * torch.clamp(y2g - y1g, min=0)
	iou = inter / (area_p + area_g - inter + eps)
	return 1 - iou.mean()


class DetCriterion(nn.Module):
	def __init__(self, obj_w: float = 1.0, box_w: float = 2.0):
		super().__init__()
		self.obj_w = obj_w
		self.box_w = box_w
		self.bce = nn.BCEWithLogitsLoss()

	def forward(self, det_pred: torch.Tensor, tgt: dict) -> torch.Tensor:
		obj_logit = det_pred[:, 0:1]
		box = torch.sigmoid(det_pred[:, 1:5])
		obj_tgt = tgt["obj"].to(det_pred.device)
		box_tgt = tgt["box"].to(det_pred.device)
		obj_loss = self.bce(obj_logit, obj_tgt)
		mask = (obj_tgt > 0.5).float()
		if mask.sum() < 1:
			box_loss = det_pred.new_tensor(0.0)
		else:
			box_loss = iou_loss(
				(box * mask).permute(0, 2, 3, 1).reshape(-1, 4)[mask.permute(0, 2, 3, 1).reshape(-1, 1).squeeze() > 0.5],
				box_tgt.permute(0, 2, 3, 1).reshape(-1, 4)[mask.permute(0, 2, 3, 1).reshape(-1, 1).squeeze() > 0.5],
			)
		return self.obj_w * obj_loss + self.box_w * box_loss
