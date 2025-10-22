import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


def binarize(mask: np.ndarray) -> np.ndarray:
	if mask.dtype != np.uint8:
		mask = (mask > 0).astype(np.uint8) * 255
	return mask


def extract_centerline(mask: np.ndarray) -> np.ndarray:
	mask01 = (mask > 0).astype(np.uint8)
	skel = skeletonize(mask01 > 0)
	return (skel.astype(np.uint8) * 255)


def _centerline_points(centerline: np.ndarray) -> np.ndarray:
	pts = np.column_stack(np.where(centerline > 0))
	return pts


def _pick_segment_points(pts: np.ndarray, length_px: int, rng: np.random.RandomState) -> np.ndarray:
	if len(pts) == 0:
		return np.empty((0, 2), dtype=int)
	idx = rng.randint(0, len(pts))
	cy, cx = pts[idx]
	dists = ((pts[:, 0] - cy) ** 2 + (pts[:, 1] - cx) ** 2) ** 0.5
	order = np.argsort(dists)
	seg_idx = order[: max(3, length_px)]
	return pts[seg_idx]


def apply_stenosis(mask: np.ndarray, seg_pts: np.ndarray, severity: float) -> np.ndarray:
	if seg_pts.size == 0:
		return mask
	mask01 = (mask > 0).astype(np.uint8)
	dist = distance_transform_edt(mask01)
	H, W = mask01.shape
	sten_mask = np.zeros_like(mask01, dtype=bool)
	for (y, x) in seg_pts:
		r = int(max(3, severity * 12))
		y0, y1 = max(0, y - r), min(H, y + r + 1)
		x0, x1 = max(0, x - r), min(W, x + r + 1)
		patch_y, patch_x = np.ogrid[y0:y1, x0:x1]
		sten_mask[y0:y1, x0:x1] |= (patch_y - y) ** 2 + (patch_x - x) ** 2 <= r * r
	reduction = np.clip(severity, 0.05, 0.95)
	new_mask = mask01.copy()
	threshold = dist * (1.0 - reduction)
	remove = (dist <= threshold.max() * reduction) & sten_mask
	new_mask[remove] = 0
	new_mask = cv2.morphologyEx(new_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
	return (new_mask.astype(np.uint8) * 255)


def generate_stenotic_mask(mask: np.ndarray, length_px: int = 40, severity: float = 0.5, seed: int = 42) -> np.ndarray:
	mask = binarize(mask)
	centerline = extract_centerline(mask)
	pts = _centerline_points(centerline)
	rng = np.random.RandomState(seed)
	seg_pts = _pick_segment_points(pts, length_px, rng)
	return apply_stenosis(mask, seg_pts, severity)
