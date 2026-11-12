"""ObjectClear 原生节点。

链路：
    CheckpointLoaderSimple(ObjectClear_fp16) ─┐
    VAELoader(sdxl.vae) ─────────────────────┤
    DualCLIPLoader(clip_l + clip_g, sdxl) ───┤-> CLIPTextEncode -> positive/negative
    CLIPVisionLoader(clip-vit-large-patch14) ┤
                                             └-> ObjectClearConditioning -> ObjectClearSampler
                                                 -> VAEDecode -> ObjectClearAGF
"""

import torch
import torch.nn.functional as F
import os

import folder_paths
import latent_preview
import comfy.model_management as mm
import comfy.sample
import comfy.samplers
import comfy.utils
node_oc_cr_path = os.path.dirname(os.path.abspath(__file__))

from .agf import attention_guided_fusion
from .attn import build_attn_mask, install, uninstall
from .common import FUSE_INDEX
from .postfuse import load_postfuse
from .vision import encode_object_embed, object_pixel_values



# --------------------------------------------------------------------------- utils
def _as_bhw(image, mask):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4:
        mask = mask[..., 0]
    if mask.shape[0] != image.shape[0]:
        if mask.shape[0] == 1:
            mask = mask.expand(image.shape[0], -1, -1)
        else:
            raise ValueError(
                "ObjectClear: image batch=%d 与 mask batch=%d 不一致"
                % (image.shape[0], mask.shape[0])
            )
    return image, mask


def resize_by_short_side(h, w, target):
    """官方 objectclear/utils/image_utils.resize_by_short_side 的等价实现（返回 w,h）。"""
    if min(w, h) < target:
        new_w, new_h = (w + 15) // 16 * 16, (h + 15) // 16 * 16
    else:
        if w < h:
            new_w, new_h = target, int(h * target / w)
        else:
            new_h, new_w = target, int(w * target / h)
        new_w, new_h = (new_w + 15) // 16 * 16, (new_h + 15) // 16 * 16
    return max(16, new_w), max(16, new_h)


def _resize_image(image, w, h):
    if image.shape[1] == h and image.shape[2] == w:
        return image
    x = image.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(h, w), mode="bicubic", antialias=True)
    return x.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def _resize_mask(mask, w, h):
    if mask.shape[1] == h and mask.shape[2] == w:
        return mask
    return F.interpolate(mask.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)


# --------------------------------------------------------------------------- 1. postfuse
class ObjectClearPostfuseLoader:
    """加载 postfuse_module（28MB，纯 torch 结构，strict=True 自校验）。"""

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("clip")
        default = next((f for f in files if "postfuse" in f.lower()), files[0] if files else "")
        return {"required": {"postfuse": (files, {"default": default})}}

    RETURN_TYPES = ("OBJECTCLEAR",)
    RETURN_NAMES = ("postfuse",)
    FUNCTION = "load"
    CATEGORY = "ObjectClear"
    DESCRIPTION = (
        "加载 ObjectClear 的 postfuse_module（把 CLIP-Vision 的物体嵌入写进文本 embedding 第 5 个 token）。\n"
        "文件放 models/clip/，官方名字 postfuse_module_fp16.safetensors。"
    )

    def load(self, postfuse):
        path = folder_paths.get_full_path_or_raise("clip", postfuse)
        device = mm.get_torch_device()
        dtype = torch.float16 if mm.should_use_fp16(device) else torch.float32
        mod = load_postfuse(path, dtype=dtype, device=device)
        return ({"module": mod, "path": path, "dtype": dtype},)


# --------------------------------------------------------------------------- 2. conditioning
def _fuse_conditioning(cond, module, object_embeds):
    """把 object_embeds 写进每条 conditioning 的第 FUSE_INDEX 个 token（只动 positive）。"""
    out = []
    for item in cond:
        t = item[0]
        d = dict(item[1]) if len(item) > 1 and isinstance(item[1], dict) else {}
        b = object_embeds.shape[0]
        if t.shape[0] == 1 and b > 1:
            t = t.repeat(b, 1, 1)
        elif t.shape[0] != b:
            raise ValueError(
                "ObjectClear: conditioning batch=%d 与 image batch=%d 不匹配"
                % (t.shape[0], b)
            )

        # 以模块自身的 dtype/device 为准（postfuse 常驻 fp16），算完再转回原 dtype
        mp = next(module.parameters())
        dev, dt = mp.device, mp.dtype
        fused = module(t.to(dev, dt), object_embeds.to(dev, dt).unsqueeze(1), FUSE_INDEX)
        fused = fused.to(t.device, t.dtype)

        pooled = d.get("pooled_output", None)
        if pooled is not None and pooled.shape[0] == 1 and b > 1:
            d["pooled_output"] = pooled.repeat(b, 1)

        mconds = d.get("model_conds", None)
        if isinstance(mconds, dict) and "c_crossattn" in mconds:
            try:
                import comfy.conds

                mconds["c_crossattn"] = comfy.conds.CONDCrossAttn(fused)
            except Exception:
                pass

        out.append([fused, d])
    return out


class ObjectClearConditioning:
    """物体感知条件：CLIP-Vision 物体嵌入 -> postfuse -> 替换 positive 的第 5 个 token。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "postfuse": ("OBJECTCLEAR",),
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "ObjectClear"
    DESCRIPTION = (
        "ObjectClear 的核心条件：只把 mask 内物体喂给 CLIP-Vision（并保留官方 [-1,1] 预处理语义），\n"
        "经 postfuse_module 融合成 2048 维 token，替换 positive embedding 的第 5 个 token。\n"
        "negative 原样返回（官方 negative 不融合）。"
    )
    def _move_cond_to_cuda(self, data):
        """Recursively move all tensors in the conditioning data to CUDA."""
        import comfy.model_management
        target_device = comfy.model_management.get_torch_device()
        if isinstance(data, torch.Tensor):
            return data.to(device=target_device)
        elif isinstance(data, dict):
            return {k: self._move_cond_to_cuda(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move_cond_to_cuda(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(self._move_cond_to_cuda(v) for v in data)
        return data

    def apply(self, postfuse, clip_vision, image, mask,positive=None, negative=None):
        if positive is None or negative is None:
            positive = torch.load(
                os.path.join(node_oc_cr_path, "sdxl-positive_saved_cond.pt"),
                map_location="cpu",
            )
            negative = torch.load(
                os.path.join(node_oc_cr_path, "sdxl-negative_saved_cond.pt"),
                map_location="cpu",
            )
            positive = self._move_cond_to_cuda(positive)
            negative = self._move_cond_to_cuda(negative)
        image, mask = _as_bhw(image, mask)
        pv = object_pixel_values(image, mask)
        emb = encode_object_embed(clip_vision, pv)  # [B,768] fp32
        module = postfuse["module"]
        return (_fuse_conditioning(positive, module, emb), negative)


# --------------------------------------------------------------------------- 3. sampler
def _inject_inpaint_conds(cond, mask_lat, lat_unscaled):
    """把 9 通道 inpaint 需要的 mask / masked_image 通过 cond 传给 concat_cond。

    这样既拿到正确的 concat 通道，又不会触发 sampler 层的 out/latent 混合
    （官方 9 通道路线不做逐 step latent 混合）。
    """
    out = []
    for item in cond:
        t = item[0]
        d = dict(item[1]) if len(item) > 1 and isinstance(item[1], dict) else {}
        d["concat_mask"] = mask_lat
        d["concat_latent_image"] = lat_unscaled
        out.append([t, d])
    return out


class ObjectClearSampler:
    """原生 KSampler 内核 + 9 通道 inpaint concat + 末步注意力捕获。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.01}),
                "short_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32}),
            }
        }

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("latent", "attn_map")
    FUNCTION = "sample"
    CATEGORY = "ObjectClear"
    DESCRIPTION = (
        "ObjectClear 采样：短边缩放到 short_size（16 的倍数）-> VAE 编码 -> 原生 KSampler。\n"
        "UNet 必须是 9 通道 inpaint 模型（ObjectClear_fp16）。\n"
        "attn_map 是末步 cross-attention 的物体图（latent 分辨率），供 ObjectClearAGF 使用。\n"
        "strength=1.0 时从纯噪声起跑（官方默认）；<1 时从加噪原图起跑。"
    )

    def sample(
        self,
        model,
        vae,
        positive,
        negative,
        image,
        mask,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        strength,
        short_size,
    ):
        device = mm.get_torch_device()
        image, mask = _as_bhw(image, mask)
        b, h, w, _ = image.shape

        tw, th = resize_by_short_side(h, w, short_size)
        image = _resize_image(image, tw, th)
        mask = _resize_mask(mask, tw, th)
        lh, lw = th // 8, tw // 8

        lat = vae.encode(image[:, :, :, :3])  # 未缩放 latent [B,4,lh,lw]
        mask_lat = F.interpolate((mask > 0.5).float().unsqueeze(1), size=(lh, lw), mode="nearest")

        if strength >= 0.999:
            samples = torch.zeros_like(lat)  # 官方 strength=1.0：latents=noise*init_noise_sigma
        else:
            samples = lat

        pos = _inject_inpaint_conds(positive, mask_lat, lat)
        neg = _inject_inpaint_conds(negative, mask_lat, lat)

        m = model.clone()
        state = install(m, FUSE_INDEX, expect_inner=640)
        try:
            latent_image = samples.to(device).float()
            noise = comfy.sample.prepare_noise(latent_image, seed)
            ks = comfy.samplers.KSampler(
                m,
                steps=steps,
                device=device,
                sampler=sampler_name,
                scheduler=scheduler,
                denoise=strength,
            )
            callback = latent_preview.prepare_callback(m, ks.steps)
            out = ks.sample(
                noise,
                pos,
                neg,
                cfg,
                latent_image=latent_image,
                seed=seed,
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            )
        finally:
            uninstall(m, state)

        amap = build_attn_mask(state, lh, lw, b)
        state["pieces"].clear()
        if amap is None:
            raise RuntimeError(
                "ObjectClear: 没有捕获到注意力图（patch key=%s，被调用 %d 次）。"
                "常见原因：采样器未提供 sample_sigmas，或该 MODEL 已被别的补丁覆盖。"
                % (state.get("key"), state.get("calls", 0))
            )

        return ({"samples": out.float()}, amap.float().cpu().clamp(0, 1))


# --------------------------------------------------------------------------- 4. AGF
class ObjectClearAGF:
    """Attention Guided Fusion：用注意力图把生成结果贴回原图（含小波低频校色）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "decoded": ("IMAGE",),
                "attn_map": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "fuse"
    CATEGORY = "ObjectClear"
    DESCRIPTION = (
        "注意力引导融合（官方 attention_guided_fusion 的纯 torch 移植）：\n"
        "attn_map 二值化 -> 21x21 膨胀 -> 9x9 高斯 -> 取 max，然后把 decoded 贴回 image，\n"
        "并用小波高频(原图)+低频(生成图) 校正颜色，保证非目标区域完全保留原图。"
    )

    def fuse(self, image, decoded, attn_map):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        h, w = decoded.shape[1], decoded.shape[2]
        image = _resize_image(image, w, h)

        if attn_map.ndim == 2:
            attn_map = attn_map.unsqueeze(0)
        if attn_map.shape[0] == 1 and image.shape[0] > 1:
            attn_map = attn_map.expand(image.shape[0], -1, -1)
        if decoded.shape[0] == 1 and image.shape[0] > 1:
            decoded = decoded.expand(image.shape[0], -1, -1, -1)

        fused = attention_guided_fusion(image, decoded.clamp(0, 1), attn_map)
        return (fused,)





NODE_CLASS_MAPPINGS = {
    "ObjectClearPostfuseLoader": ObjectClearPostfuseLoader,
    "ObjectClearConditioning": ObjectClearConditioning,
    "ObjectClearSampler": ObjectClearSampler,
    "ObjectClearAGF": ObjectClearAGF,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ObjectClearPostfuseLoader": "ObjectClear 后融合加载器",
    "ObjectClearConditioning": "ObjectClear 物体条件",
    "ObjectClearSampler": "ObjectClear 采样器",
    "ObjectClearAGF": "ObjectClear 注意力融合",
}
