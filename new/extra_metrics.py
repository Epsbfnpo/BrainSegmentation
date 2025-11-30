import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


def cl_score(v: np.ndarray, s: np.ndarray) -> float:
    """Compute skeleton coverage of a volume."""
    return float(np.sum(v * s) / np.sum(s))


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


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """PyTorch soft erosion for 2D/3D tensors."""
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    if len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    raise ValueError("Unsupported tensor shape for soft_erode")


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """PyTorch soft dilation for 2D/3D tensors."""
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    if len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
    raise ValueError("Unsupported tensor shape for soft_dilate")


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iter_: int = 3) -> torch.Tensor:
    """Compute soft skeleton for 2D/3D probability maps."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def compute_clce(y_pred_logits: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute centerline cross-entropy (lower is better)."""
    y_pred_prob = torch.softmax(y_pred_logits, dim=1)

    if y_true.shape[1] == 1:
        num_classes = y_pred_logits.shape[1]
        y_true_oh = (
            F.one_hot(y_true.squeeze(1).long(), num_classes=num_classes)
            .permute(0, 4, 1, 2, 3)
            .float()
        )
    else:
        y_true_oh = y_true.float()

    log_prob = F.log_softmax(y_pred_logits, dim=1)
    ce_map = -(y_true_oh * log_prob)

    iters = 3
    skel_pred = soft_skel(y_pred_prob, iters)
    skel_true = soft_skel(y_true_oh, iters)

    num_classes = y_pred_logits.shape[1]
    loss_val = 0.0
    cnt = 0

    for c in range(1, num_classes):
        tprec = torch.sum(ce_map[:, c] * skel_true[:, c]) / (torch.sum(skel_true[:, c]) + 1e-5)
        tsens = torch.sum(ce_map[:, c] * skel_pred[:, c]) / (torch.sum(skel_pred[:, c]) + 1e-5)
        loss_val += (tprec + tsens)
        cnt += 1

    if cnt == 0:
        return 0.0
    return float((loss_val / cnt).item())
