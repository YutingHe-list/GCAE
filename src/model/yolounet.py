import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
	def __init__(self, c_in, c_out, k=3, s=1, p=1):
		super().__init__()
		self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
		self.bn = nn.BatchNorm2d(c_out)
		self.act = nn.SiLU(inplace=True)

	def forward(self, x):
		return self.act(self.bn(self.conv(x)))


class Down(nn.Module):
	def __init__(self, c_in, c_out):
		super().__init__()
		self.block = nn.Sequential(
			ConvBlock(c_in, c_out, 3, 2, 1),
			ConvBlock(c_out, c_out),
		)

	def forward(self, x):
		return self.block(x)


class Up(nn.Module):
	def __init__(self, c_in, c_out):
		super().__init__()
		self.conv1x1 = ConvBlock(c_in, c_out)
		self.conv = ConvBlock(c_out, c_out)

	def forward(self, x, skip):
		x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
		x = torch.cat([x, skip], dim=1)
		x = self.conv1x1(x)
		return self.conv(x)


class YOLOUNet(nn.Module):
	def __init__(self, num_classes_det: int = 1):
		super().__init__()
		self.stem = ConvBlock(1, 32)
		self.d1 = Down(32, 64)
		self.d2 = Down(64, 128)
		self.d3 = Down(128, 256)
		self.u2 = Up(256 + 128, 128)
		self.u1 = Up(128 + 64, 64)
		self.seg_head = nn.Conv2d(64, 1, 1)
		self.det_head = nn.Sequential(
			ConvBlock(256, 256),
			nn.Conv2d(256, 1 + 4, 1),
		)

	def forward(self, x):
		e0 = self.stem(x)
		e1 = self.d1(e0)
		e2 = self.d2(e1)
		e3 = self.d3(e2)
		s2 = self.u2(torch.cat([e3], dim=1), e2)
		s1 = self.u1(torch.cat([s2], dim=1), e1)
		seg_logit = self.seg_head(s1)
		det = self.det_head(e3)
		return {"seg_logit": seg_logit, "det_pred": det}
