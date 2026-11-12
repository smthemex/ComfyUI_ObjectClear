"""末步 cross-attention 捕获（ObjectClear 的 "Adaptive Target-Aware Attention"）。

官方做法：把 down_blocks.1.attentions.0.transformer_blocks.0.attn2 换成非 SDPA 的
AttnProcessor，在最后一个时间步取出 attention_probs[:, :, 5]（第 5 个文本 token =
物体 token），再按 head 求均值、双线性放大到 latent 分辨率、逐样本 min-max 归一化。

ComfyUI 原生通道：transformer_options["patches_replace"]["attn2"][(section, id, depth)]，
外层已做完 to_q/to_k/to_v/to_out，patch 只接管 attention 本身 —— 与 diffusers 的
AttnProcessor 上下文逐位对应。默认只挂 1 个 key，不影响其它层。
"""

import torch

import comfy.ldm.modules.attention as comfy_attention
from comfy.ldm.modules.attention import SpatialTransformer
from comfy.ldm.modules.diffusionmodules.openaimodel import Downsample

from .common import ATTN_DOWNSCALE

__all__ = ["find_capture_key", "install", "uninstall", "build_attn_mask"]


def _first_attn2(layer):
    """SpatialTransformer 本身没有 attn2，cross-attention 在 transformer_blocks[*] 里。"""
    direct = getattr(layer, "attn2", None)
    if direct is not None:
        return direct
    for blk in getattr(layer, "transformer_blocks", []):
        a = getattr(blk, "attn2", None)
        if a is not None:
            return a
    return None


def find_capture_key(diffusion_model):
    """定位官方 TARGET_LAYER = down_blocks.1.attentions.0.transformer_blocks.0.attn2。

    结构等价条件：第 1 次下采样之后出现的第一个 attention（即 1/2 latent 分辨率）。
    返回 ((section, id, depth), heads, inner_dim)。
    """
    downsamples = 0
    cand = []
    for i, block in enumerate(diffusion_model.input_blocks):
        for j, layer in enumerate(block):
            if isinstance(layer, Downsample):
                downsamples += 1
            elif isinstance(layer, SpatialTransformer):
                attn2 = _first_attn2(layer)
                if attn2 is None:
                    continue
                heads = getattr(attn2, "heads", None)
                inner = getattr(attn2.to_q, "out_features", None)
                # patch key = (section, input_blocks 索引, transformer_blocks 深度)
                cand.append((("input", i, 0), downsamples, heads, inner))

    for key, nd, heads, inner in cand:
        if nd == 1:
            if not heads or not inner:
                raise RuntimeError("ObjectClear: 无法从 %s 读取 heads/inner_dim" % (key,))
            return key, int(heads), int(inner)

    raise RuntimeError(
        "ObjectClear: 没有在第 1 次下采样后找到 cross-attention，该 UNet 结构不被支持 "
        "(candidates=%s)" % ([c[0] for c in cand],)
    )


def _is_last_step(extra_options):
    """用 sample_sigmas 反查当前步索引（无状态，CFG 分块 / 多次采样都安全）。

    注意：sigmas 表比"模型调用次数"多一项（末尾的 0），所以最后一次 UNet 调用的索引
    是 len(sigmas)-2，而不是 len(sigmas)-1。
    """
    ss = extra_options.get("sample_sigmas", None)
    cur = extra_options.get("sigmas", None)
    if ss is None or cur is None:
        return True  # 自定义采样器没给 sigma 表 -> 每步都捕获，保留最后一次
    idx = _step_index(extra_options)
    return idx >= len(torch.as_tensor(ss).flatten()) - 2


def _step_index(extra_options):
    ss = extra_options.get("sample_sigmas", None)
    cur = extra_options.get("sigmas", None)
    if ss is None or cur is None:
        return -1
    ss = torch.as_tensor(ss).flatten().to("cpu", torch.float32)
    cur = torch.as_tensor(cur).flatten()[0].to("cpu", torch.float32)
    return int((ss - cur).abs().argmin())


def _attention_probs(q, k, heads, fuse_index):
    """[B,N,C]/[B,S,C] -> probs[:, :, fuse_index]，返回 [B, heads, N]（fp32）。"""
    b, n, c = q.shape
    dim_head = c // heads
    qh = q.view(b, n, heads, dim_head).transpose(1, 2).float()
    kh = k.view(b, k.shape[1], heads, dim_head).transpose(1, 2).float()
    scale = dim_head ** -0.5
    probs = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) * scale, dim=-1)
    return probs[..., fuse_index]


def make_patch(state, heads, fuse_index):
    """state: {"pieces": [(step, [B,heads,N] probs, labels)], ...}"""

    def patch(q, k, v, extra_options):
        state["calls"] = state.get("calls", 0) + 1
        if _is_last_step(extra_options):
            with torch.no_grad():
                probs = _attention_probs(q, k, heads, fuse_index)
            labels = list(extra_options.get("cond_or_uncond", []))
            state["pieces"].append((_step_index(extra_options), probs, labels))
        # 输出仍走原生 attention，保证与未打补丁时逐位一致
        return comfy_attention.optimized_attention(q, k, v, heads)

    patch.__oc_owner__ = "objectclear"
    return patch


def install(model_patcher, fuse_index, expect_inner=None):
    """就地在 clone 出来的 model_options 上挂 attn2 patch（不调用 set_model_patch_replace）。"""
    dm = model_patcher.model.diffusion_model
    key, heads, inner = find_capture_key(dm)
    if expect_inner is not None and inner != expect_inner:
        raise RuntimeError(
            "ObjectClear: 捕获层 inner_dim=%d，期望 %d（可能不是 ObjectClear/SDXL-inpaint UNet）"
            % (inner, expect_inner)
        )
    state = {"pieces": [], "key": key, "heads": heads, "inner": inner}
    topts = model_patcher.model_options.setdefault("transformer_options", {})
    patches = topts.setdefault("patches_replace", {})
    attn2 = patches.setdefault("attn2", {})
    attn2[key] = make_patch(state, heads, fuse_index)
    return state


def uninstall(model_patcher, state):
    """摘掉本次挂的 patch（clone 出来的 model_options 私有，摘掉只是徒手清理）。"""
    try:
        pr = model_patcher.model_options.get("transformer_options", {}).get("patches_replace", {})
        attn2 = pr.get("attn2", {})
        key = state.get("key", None)
        if key is not None and key in attn2:
            del attn2[key]
    except Exception:
        pass


def build_attn_mask(state, latent_h, latent_w, batch):
    """复刻官方 resize_attn_map_divide2：取 cond 行 -> head 均值 -> 放大 -> 逐样本 min-max。

    返回 [batch, latent_h, latent_w] in [0,1]（与官方一致，双线性 align_corners=False）。
    """
    pieces = state.get("pieces") or []
    if not pieces:
        return None

    last = max(p[0] for p in pieces)
    chosen = [p for p in pieces if p[0] == last]

    rows = []
    for _, probs, labels in chosen:  # probs: [b_i, heads, N]
        if labels and len(labels) == probs.shape[0]:
            keep = [i for i, l in enumerate(labels) if int(l) == 0]  # 0 = cond
            if keep:
                rows.append(probs[keep])
                continue
        rows.append(probs)

    probs = torch.cat(rows, dim=0)  # [B, heads, N]
    heads = probs.shape[1]
    n = probs.shape[2]
    b = probs.shape[0]
    if b != batch:
        b = min(b, batch)
        probs = probs[:b]

    hh = latent_h // ATTN_DOWNSCALE
    ww = latent_w // ATTN_DOWNSCALE
    if n != hh * ww:
        raise RuntimeError(
            "ObjectClear: 捕获层 token 数 %d 与 latent//2 网格 %dx%d 不符（latent=%dx%d）"
            % (n, hh, ww, latent_h, latent_w)
        )

    amap = probs.view(b, heads, hh, ww).mean(dim=1, keepdim=True)
    amap = torch.nn.functional.interpolate(
        amap.float(), size=(latent_h, latent_w), mode="bilinear", align_corners=False
    )
    mn = amap.amin(dim=(2, 3), keepdim=True)
    mx = amap.amax(dim=(2, 3), keepdim=True)
    amap = (amap - mn) / (mx - mn + 1e-8)
    return amap.squeeze(1)
