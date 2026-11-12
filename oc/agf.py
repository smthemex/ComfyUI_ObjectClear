"""attention_guided_fusion 的纯 torch 移植（官方实现用 numpy + cv2 + scipy）。

官方链路（objectclear/utils/attention_guided_fusion.py）：
    attn(PIL L, uint8) -> threshold 128 -> 二值
    -> NEAREST 放大到像素分辨率
    -> cv2.dilate(21x21 椭圆) -> cv2.GaussianBlur(9x9, sigma=2)
    -> max(原始二值, 膨胀模糊) -> am_merged
    ori_out = ori*(1-am_bin) ; rem_out = removed*(1-am_bin)
    -> pad 到 8 的倍数 -> 小波分解：content 取高频 + style 取低频 -> 合成
    -> fused = wave*(1-am_merged) + removed*am_merged

这里逐条复刻，包括官方在 uint8 上的**截断**量化点，便于数值对拍。
"""

import math

import torch
import torch.nn.functional as F

from .common import (
    AGF_BLUR_KERNEL,
    AGF_BLUR_SIGMA,
    AGF_DILATE_KERNEL,
    AGF_PAD_MULTIPLE,
    AGF_THRESHOLD,
    AGF_WAVELET_LEVELS,
)

_WAVELET_KERNEL = torch.tensor(
    [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ],
    dtype=torch.float32,
)


# cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)) 的精确栅格
# （cv2 是 polygon 填充的椭圆，比理想圆盘"胖"，实测 333 vs 317 像素，差 16 个边界点，
#   所以这里直接写死每行的 [lo, hi] 闭区间，保证与官方逐像素一致）
_CV2_ELLIPSE_21_ROWS = (
    (10, 10),
    (6, 14),
    (4, 16),
    (3, 17),
    (2, 18),
    (1, 19),
    (1, 19),
    (0, 20),
    (0, 20),
    (0, 20),
    (0, 20),
    (0, 20),
    (0, 20),
    (0, 20),
    (1, 19),
    (1, 19),
    (2, 18),
    (3, 17),
    (4, 16),
    (6, 14),
    (10, 10),
)


def _ellipse_footprint(size: int, device) -> torch.Tensor:
    if size == 21:
        fp = torch.zeros((21, 21), dtype=torch.float32, device=device)
        for dy, (lo, hi) in enumerate(_CV2_ELLIPSE_21_ROWS):
            fp[dy, lo : hi + 1] = 1.0
        return fp

    r = (size - 1) / 2.0
    ax = torch.arange(size, dtype=torch.float32, device=device) - (size // 2)
    dy, dx = torch.meshgrid(ax, ax, indexing="ij")
    return (((dx / r) ** 2 + (dy / r) ** 2) <= 1.0).to(torch.float32)


def _dilate(x: torch.Tensor, footprint: torch.Tensor) -> torch.Tensor:
    """x [N,1,H,W]，footprint [k,k] bool；边界按 0 处理（cv2 dilate 默认 -inf 等价）。"""
    kh, kw = footprint.shape
    ph, pw = kh // 2, kw // 2
    n, c, h, w = x.shape
    xp = F.pad(x, (pw, pw, ph, ph), mode="constant", value=-1.0)
    out = torch.full_like(x, -1.0)
    for dy, dx in footprint.nonzero().tolist():
        out = torch.maximum(out, xp[:, :, dy : dy + h, dx : dx + w])
    return out


def _gaussian_kernel1d(ksize: int, sigma: float, device) -> torch.Tensor:
    half = (ksize - 1) / 2.0
    xs = torch.arange(ksize, dtype=torch.float32, device=device) - half
    k = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def _gaussian_blur(x: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
    """cv2.GaussianBlur(x, (k,k), sigmaX=sigma) —— 可分离 + BORDER_REFLECT_101。"""
    k = _gaussian_kernel1d(ksize, sigma, x.device)
    p = ksize // 2
    n, c, h, w = x.shape
    x = F.pad(x, (p, p, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, (0, 0, p, p), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x


def _blur3x3(x: torch.Tensor) -> torch.Tensor:
    """scipy.ndimage.convolve(x, kernel, mode='nearest')，kernel 对称 -> 视为相关。"""
    n, c, h, w = x.shape
    k = _WAVELET_KERNEL.to(x.dtype).to(x.device).view(1, 1, 3, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(xp, k.expand(c, 1, 3, 3), groups=c)


def _zoom_pair(x: torch.Tensor, radius: int) -> torch.Tensor:
    """scipy.ndimage.zoom(zoom(x, 1/radius, order=1), radius, order=1)。

    scipy grid_mode=False 的坐标映射是半像素中心：src = (i+0.5)/scale - 0.5，
    实测等价于 torch 的 align_corners=True（scipy 1.17 的 _zoom_shift 是角点对齐）。
    """
    if radius <= 1:
        return x
    h, w = x.shape[-2:]
    h2 = max(1, int(round(h / radius)))
    w2 = max(1, int(round(w / radius)))
    x = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=True)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
    return x


def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    x = _blur3x3(x)
    if radius > 1:
        x = _zoom_pair(x, radius)
    return x


def _wavelet_decompose(x: torch.Tensor, levels: int = AGF_WAVELET_LEVELS):
    high = torch.zeros_like(x)
    for i in range(levels):
        low = _wavelet_blur(x, 2 ** i)
        high = high + (x - low)
        x = low
    return high, x


def _quant_u8(x: torch.Tensor) -> torch.Tensor:
    """numpy .astype(uint8)：截断（floor）后夹到 [0,255]，再回到 [0,1]。"""
    return torch.floor(x.clamp(0.0, 1.0) * 255.0).clamp(0.0, 255.0) / 255.0


def _quant_round(x: torch.Tensor) -> torch.Tensor:
    """官方 ori/removed 来自 PIL(uint8)，等价于 round(x*255)。"""
    return torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0


def attention_guided_fusion(ori: torch.Tensor, removed: torch.Tensor, attn: torch.Tensor):
    """ori / removed: [B,H,W,3] in [0,1]；attn: [B,h,w] in [0,1]（latent 分辨率）。"""
    device = ori.device
    out = []
    for b in range(ori.shape[0]):
        o = _quant_round(ori[b].permute(2, 0, 1).unsqueeze(0))  # [1,3,H,W]
        r = _quant_round(removed[b].permute(2, 0, 1).unsqueeze(0))
        a = attn[b].unsqueeze(0).unsqueeze(0)

        # 官方：attn_map*255 -> uint8 -> threshold(128)
        a_u8 = torch.floor(a.clamp(0, 1) * 255.0)
        a_bin = (a_u8 > AGF_THRESHOLD).to(o.dtype)

        h, w = o.shape[-2:]
        am_up = F.interpolate(a_bin, size=(h, w), mode="nearest")
        fp = _ellipse_footprint(AGF_DILATE_KERNEL, device).to(o.dtype)
        am_d = _dilate(am_up, fp > 0.5)
        am_d = _gaussian_blur(am_d, AGF_BLUR_KERNEL, AGF_BLUR_SIGMA)
        am_merged = torch.clamp(torch.maximum(am_up, am_d), 0, 1)

        ori_out = o * (1 - am_up)
        rem_out = r * (1 - am_up)

        # pad_to_multiple(reflect)
        ph = (AGF_PAD_MULTIPLE - h % AGF_PAD_MULTIPLE) % AGF_PAD_MULTIPLE
        pw = (AGF_PAD_MULTIPLE - w % AGF_PAD_MULTIPLE) % AGF_PAD_MULTIPLE
        if ph or pw:
            o_pad = F.pad(ori_out, (0, pw, 0, ph), mode="reflect")
            r_pad = F.pad(rem_out, (0, pw, 0, ph), mode="reflect")
        else:
            o_pad, r_pad = ori_out, rem_out

        c_high, _ = _wavelet_decompose(o_pad)
        _, s_low = _wavelet_decompose(r_pad)
        wave = _quant_u8(c_high + s_low)[:, :, :h, :w]

        fused = wave * (1 - am_merged) + r * am_merged
        fused = torch.floor(fused.clamp(0, 1) * 255.0).clamp(0, 255.0) / 255.0
        out.append(fused[0].permute(1, 2, 0))

    return torch.stack(out, dim=0)
