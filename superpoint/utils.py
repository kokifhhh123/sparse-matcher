import numpy as np
import torch
import math
from types import SimpleNamespace
import kornia
from typing import Tuple

def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError

def from_homogeneous(points, eps=0.0):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
        eps: Epsilon value to prevent zero division.
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)





def sample_homography_corners(
    ori_shape,
    patch_shape,

    difficulty=0.8,
    translation=1.0,
    max_angle=60,
    n_angles=10,
    min_convexity=0.05,
    
    rng=np.random,
):
    max_angle = max_angle / 180.0 * math.pi
    width, height = ori_shape
    pwidth, pheight = width * (1 - difficulty), \
                        height * (1 - difficulty)
    
    
    min_pts1 = create_center_patch(ori_shape, (pwidth, pheight))
    full     = create_center_patch(ori_shape)
    pts2     = create_center_patch(patch_shape)
    
    scale = min_pts1 - full
    found_valid = False
    while not found_valid:
        offsets = rng.uniform(0.0, 1.0, size=(4, 2)) * scale
        pts1 = full + offsets
        found_valid = check_convex(pts1 / np.array(ori_shape), min_convexity)

    # re-center
    pts1 = pts1 - np.mean(pts1    , axis=0, keepdims=True)
    pts1 = pts1 + np.mean(min_pts1, axis=0, keepdims=True)

    # Rotation
    if n_angles > 0 and difficulty > 0:
        angles = np.linspace(-max_angle * difficulty, max_angle * difficulty, n_angles)
        rng.shuffle(angles)
        rng.shuffle(angles)
        angles = np.concatenate([[0.0], angles], axis=0)

        center = np.mean(pts1, axis=0, keepdims=True)
        rot_mat = np.reshape(
            np.stack(
                [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)],axis=1,),
            [-1, 2, 2],)
        rotated = (np.matmul(
                np.tile(np.expand_dims(pts1 - center, axis=0), [n_angles + 1, 1, 1]),
                rot_mat,
            )+ center)

        for idx in range(1, n_angles):
            warped_points = rotated[idx] / np.array(ori_shape)
            if np.all((warped_points >= 0.0) & (warped_points < 1.0)):
                pts1 = rotated[idx]
                break
    # Translation
    if translation > 0:
        min_trans = -np.min(pts1, axis=0)
        max_trans = ori_shape - np.max(pts1, axis=0)
        trans = rng.uniform(min_trans, max_trans)[None]
        pts1 += trans * translation * difficulty

    H      = compute_homography(pts1, pts2, [1.0, 1.0])

    # apply coord = H * full
    coords = warp_points(full, H, inverse=False)

    return H, coords
# Homography creation
def create_center_patch(shape, patch_shape=None):
    if patch_shape is None:
        patch_shape = shape
    
    width,  height  = shape
    pwidth, pheight = patch_shape
    left   =  int((width - pwidth) / 2)
    bottom =  int((height - pheight) / 2)
    right  =  int((width + pwidth) / 2)
    top    =  int((height + pheight) / 2)
    
    return np.array([[left, bottom], [left, top], [right, top], [right, bottom]])

def check_convex(patch, min_convexity=0.05):
    """Checks if given polygon vertices [N,2] form a convex shape"""
    for i in range(patch.shape[0]):
        x1, y1 = patch[(i - 1) % patch.shape[0]]
        x2, y2 = patch[i]
        x3, y3 = patch[(i + 1) % patch.shape[0]]
        if (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1) > -min_convexity:
            return False
    return True

def flat2mat(H):
    return np.reshape(np.concatenate([H, np.ones_like(H[:, :1])], axis=1), [3, 3])

def compute_homography(pts1_, pts2_, shape):
    """Compute the homography matrix from 4 point correspondences"""
    # Rescale to actual size
    shape = np.array(shape[::-1], dtype=np.float32)  # different convention [y, x]
    pts1 = pts1_ * np.expand_dims(shape, axis=0)
    pts2 = pts2_ * np.expand_dims(shape, axis=0)

    def ax(p, q):
        return [p[0], p[1], 1, 
                0, 0, 0, 
                -p[0] * q[0], 
                -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, 
                p[0], p[1], 1, 
                -p[0] * q[1], 
                -p[1] * q[1]]
    
    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(
        np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))

    homography = np.transpose(np.linalg.solve(a_mat, p_mat))

    return flat2mat(homography)






# Given H,P, compute A = H*P
def warp_points(points, homography, inverse=True):
    """
    Warp a list of points with the INVERSE of the given homography.
    The inverse is used to be coherent with tf.contrib.image.transform
    
    Arguments:
        points: list of N points, shape (N, 2).
        homography: batched or not (shapes (B, 3, 3) and (3, 3) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2) 
    (depending on whether the homography is batched) containing the new coordinates of the warped points.
    """
    H = homography[None] if len(homography.shape) == 2 else homography

    # Get the points to the homogeneous format
    num_points = points.shape[0]
    points = np.concatenate([points, np.ones([num_points, 1], dtype=np.float32)], -1)

    H_inv = np.transpose(np.linalg.inv(H) if inverse else H)

    warped_points = np.tensordot(points, H_inv, axes=[[1], [0]])
    warped_points = np.transpose(warped_points, [2, 0, 1])


    warped_points[np.abs(warped_points[:, :, 2]) < 1e-8, 2] = 1e-8
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]

    return warped_points[0] if len(homography.shape) == 2 \
                            else warped_points

def warp_points_torch(points, H, inverse=True):
    """
    Warp a list of points with the INVERSE of the given homography.
    The inverse is used to be coherent with tf.contrib.image.transform
    Arguments:
        points: batched list of N points, shape (B, N, 2).
        H: batched or not (shapes (B, 3, 3) and (3, 3) respectively).
        inverse: Whether to multiply the points by H or the inverse of H
    Returns: a Tensor of shape (B, N, 2) containing the new coordinates of the warps.
    """

    # Get the points to the homogeneous format
    points = to_homogeneous(points)

    # Apply the homography
    H_mat = (torch.inverse(H) if inverse else H).transpose(-2, -1)
    warped_points = torch.einsum("...nj,...ji->...ni", points, H_mat)

    warped_points = from_homogeneous(warped_points, eps=1e-5)
    return warped_points






class ImagePreprocessor:
    default_conf = {
        "resize": None,  # target edge length, None for no resizing
        "side": "long",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
    }

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        if self.conf.resize is not None:
            img = kornia.geometry.transform.resize(
                img,
                self.conf.resize,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        return img, scale

class Extractor(torch.nn.Module):
    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})

    @torch.no_grad()
    def extract(self, img: torch.Tensor, **conf) -> dict:
        """Perform extraction with online resizing"""
        if img.dim() == 3:
            img = img[None]  # add batch dim
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1]
        img, scales = ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        feats = self.forward({"image": img})
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5
        return feats
    

IGNORE_FEATURE = -2
UNMATCHED_FEATURE = -1

@torch.no_grad()
def gt_matches_from_homography(kp0, kp1, H, pos_th=3, neg_th=6, **kw):
    if kp0.shape[1] == 0 or kp1.shape[1] == 0:
        b_size, n_kp0 = kp0.shape[:2]
        n_kp1 = kp1.shape[1]
        assignment = torch.zeros(
            b_size, n_kp0, n_kp1, dtype=torch.bool, device=kp0.device
        )
        m0 = -torch.ones_like(kp0[:, :, 0]).long()
        m1 = -torch.ones_like(kp1[:, :, 0]).long()
        return assignment, m0, m1
    
    
    kp0_1 = warp_points_torch(kp0, H, inverse=False)
    kp1_0 = warp_points_torch(kp1, H, inverse=True)

    # build a distance matrix of size [... x M x N]
    dist0 = torch.sum((kp0_1.unsqueeze(-2) - kp1.unsqueeze(-3)) ** 2, -1)
    dist1 = torch.sum((kp0.unsqueeze(-2) - kp1_0.unsqueeze(-3)) ** 2, -1)
    dist = torch.max(dist0, dist1)
    # print('distance:\n',dist,dist.shape)



    reward = (dist < pos_th**2).float() - (dist > neg_th**2).float()

    min0 = dist.min(-1).indices
    min1 = dist.min(-2).indices

    ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    ismin1 = ismin0.clone()
    ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    positive = ismin0 & ismin1 & (dist < pos_th**2)
    # print('positive\n',positive,positive.shape)
    # positive_true = torch.sum(positive)
    # print('positive true\n',positive_true.item())



    negative0 = dist0.min(-1).values > neg_th**2
    negative1 = dist1.min(-2).values > neg_th**2

    # pack the indices of positive matches
    # if -1: unmatched point
    # if -2: ignore point
    
    unmatched = min0.new_tensor(UNMATCHED_FEATURE)
    ignore = min0.new_tensor(IGNORE_FEATURE)
    # print('unmatched\n',unmatched,unmatched.shape)
    # print('ignore\n',ignore,ignore.shape)
    # print('min0\n',min0,min0.shape)

    m0 = torch.where(positive.any(-1), min0, ignore)
    # print('m0\n',m0,m0.shape)
    m1 = torch.where(positive.any(-2), min1, ignore)
    m0 = torch.where(negative0, unmatched, m0)
    m1 = torch.where(negative1, unmatched, m1)

    return {
        "assignment": positive,
        "reward": reward,
        "matches0": m0,
        "matches1": m1,
        "matching_scores0": (m0 > -1).float(),
        "matching_scores1": (m1 > -1).float(),
        "proj_0to1": kp0_1,
        "proj_1to0": kp1_0,
    }



from pathlib import Path
import cv2
from typing import Callable, List, Optional, Tuple, Union


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str = "max",
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale

def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image

def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    image = read_image(path)
    # print(image.shape)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }