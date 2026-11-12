"""物体嵌入：官方 CLIPImageEncoder 的等价实现（用原生 CLIP_VISION 模型跑前向）。

官方语义（objectclear/models/clip_image_encoder.py）：
    obj_only = init_image * (mask > 0.5)      # init_image ∈ [-1,1]，不是 [0,1]！
    if h != 224: F.interpolate(obj_only, (224,224), mode="bilinear", antialias=True)
    x = Normalize(CLIP mean/std)(x)           # 直接对 [-1,1] 的数做归一化
    emb = visual_projection(vision_model(x).pooler_output)   # [B,768]

注意：comfy 的 clip_preprocess 是「[0,1] -> 缩放+中心裁剪 -> 归一化」，与上面**不同**，
所以这里不用它，自己实现预处理（坑点之一）。
"""

import torch
import torch.nn.functional as F

from .common import CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD

__all__ = ["object_pixel_values", "encode_object_embed"]


def object_pixel_values(image, mask, size=CLIP_IMAGE_SIZE):
    """image [B,H,W,3] in [0,1]；mask [B,H,W] in [0,1] -> [B,3,size,size] 归一化后的像素。"""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4:  # [B,H,W,1]
        mask = mask[..., 0]
    if mask.shape[0] != image.shape[0]:
        if mask.shape[0] == 1:
            mask = mask.expand(image.shape[0], -1, -1)
        else:
            raise ValueError("ObjectClear: image/mask batch 不一致 %s vs %s" % (image.shape, mask.shape))

    h, w = image.shape[1], image.shape[2]
    if mask.shape[1] != h or mask.shape[2] != w:
        mask = F.interpolate(mask.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)

    x = image.permute(0, 3, 1, 2).to(torch.float32)  # -> [-1,1]，与 VaeImageProcessor 一致
    x = x * 2.0 - 1.0
    x = x * (mask > 0.5).to(x.dtype).unsqueeze(1)

    if x.shape[-2] != size or x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True)

    mean = torch.tensor(CLIP_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def encode_object_embed(clip_vision, pixel_values):
    """跑原生 CLIPVision 模型，取 image_embeds（= visual_projection(pooler_output)）。"""
    import comfy.model_management as mm

    mm.load_model_gpu(clip_vision.patcher)
    pv = pixel_values.to(device=clip_vision.load_device, dtype=torch.float32)
    out = clip_vision.model(pixel_values=pv, intermediate_output=-2)
    emb = out[2]  # image_embeds
    return emb.float()
