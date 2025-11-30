import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


def cl_score(v: np.ndarray, s: np.ndarray) -> float:
    """Compute skeleton coverage of a volume."""
    denom = np.sum(s)
    if denom == 0:
        return 0.0
    return float(np.sum(v * s) / denom)


def compute_cldice(v_p: np.ndarray, v_l: np.ndarray) -> float:
    """Compute clDice between prediction and label volumes.

    Args:
        v_p: Predicted mask (binary numpy array).
        v_l: Ground-truth mask (binary numpy array).
    """
    v_p = (v_p > 0).astype(np.uint8)
    v_l = (v_l > 0).astype(np.uint8)

    if np.sum(v_p) == 0 or np.sum(v_l) == 0:
        return 0.0

    s_p = skeletonize(v_p)
    s_l = skeletonize(v_l)

    tprec = cl_score(v_p, s_l)
    tsens = cl_score(v_l, s_p)

    if (tprec + tsens) == 0:
        return 0.0

    return float(2.0 * tprec * tsens / (tprec + tsens))


def compute_cbdice(v_p: np.ndarray, v_l: np.ndarray) -> float:
    """Compute cbDice (radius-aware clDice variant)."""
    v_p = (v_p > 0).astype(np.uint8)
    v_l = (v_l > 0).astype(np.uint8)

    if np.sum(v_p) == 0 or np.sum(v_l) == 0:
        return 0.0

    s_p = skeletonize(v_p)
    s_l = skeletonize(v_l)

    dist_p = distance_transform_edt(v_p)
    dist_l = distance_transform_edt(v_l)

    smooth = 1.0

    radius_l = dist_l * s_l
    max_r_l = np.max(radius_l) if np.max(radius_l) > 0 else 1.0
    dist_l_norm = dist_l / max_r_l

    radius_p = dist_p * s_p
    max_r_p = np.max(radius_p) if np.max(radius_p) > 0 else 1.0
    dist_p_norm = dist_p / max_r_p

    tprec = (np.sum(s_p * dist_l_norm) + smooth) / (np.sum(s_p) + smooth)
    tsens = (np.sum(s_l * dist_p_norm) + smooth) / (np.sum(s_l) + smooth)

    return float(2.0 * tprec * tsens / (tprec + tsens))


# --- PyTorch Soft Skeleton Utils ---

def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of soft erosion."""
    if len(img.shape) == 4:  # 2D: (B, C, H, W)
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:  # 3D: (B, C, D, H, W)
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    return img


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of soft dilation."""
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
    return img


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iter_: int = 3) -> torch.Tensor:
    """Compute soft skeleton for 2D/3D probability maps.
    Optimization: Ensure this runs on small batches or single channels.
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def compute_clce(y_pred_logits: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute centerline cross-entropy (lower is better).

    Memory Optimized Version:
    Iterates over classes one by one to avoid OOM on GPUs when handling
    large 3D volumes with many classes (e.g., 87 classes).
    """
    if y_pred_logits.ndim != 5:
        return 0.0

    num_classes = y_pred_logits.shape[1]

    y_pred_prob = torch.softmax(y_pred_logits, dim=1)
    log_prob = F.log_softmax(y_pred_logits, dim=1)

    is_indices = y_true.shape[1] == 1

    total_loss = 0.0
    valid_classes = 0
    smooth = 1e-5
    iters = 3

    for c in range(1, num_classes):
        if is_indices:
            y_true_c = (y_true == c).float()
        else:
            y_true_c = y_true[:, c:c + 1].float()

        y_pred_c = y_pred_prob[:, c:c + 1]

        skel_true = soft_skel(y_true_c, iter_=iters)
        skel_pred = soft_skel(y_pred_c, iter_=iters)

        log_prob_c = log_prob[:, c:c + 1]
        ce_map_c = -(y_true_c * log_prob_c)

        tprec = torch.sum(ce_map_c * skel_true) / (torch.sum(skel_true) + smooth)
        tsens = torch.sum(ce_map_c * skel_pred) / (torch.sum(skel_pred) + smooth)

        total_loss += (tprec + tsens).item()
        valid_classes += 1

        del y_true_c, y_pred_c, skel_true, skel_pred, ce_map_c, log_prob_c

    if valid_classes == 0:
        return 0.0

    return total_loss / valid_classes
