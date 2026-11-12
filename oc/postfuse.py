"""postfuse_module 的原生（纯 torch）实现。

权重文件：models/clip/postfuse_module_fp16.safetensors
key 与官方 objectclear/models/postfuse_module.py 1:1：
    layer_norm.{weight,bias}
    mlp1.fc1.{weight,bias} mlp1.fc2.{weight,bias}
    mlp2.layernorm.{weight,bias} mlp2.fc1.{weight,bias} mlp2.fc2.{weight,bias}
共 14 张量，因此可以 load_state_dict(strict=True) 自校验。
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual

    def forward(self, x):
        residual = x
        x = self.fc1(self.layernorm(x))
        x = F.gelu(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class PostfuseModule(nn.Module):
    def __init__(self, embed_dim=2048, embed_dim_img=768):
        super().__init__()
        self.mlp1 = _MLP(embed_dim_img, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = _MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, object_embeds):
        x = self.mlp1(object_embeds)
        x = self.mlp2(x)
        return self.layer_norm(x)

    def forward(self, text_embeds, object_embeds, fuse_index):
        """text_embeds [B,L,2048] / object_embeds [B,1,768] -> 第 fuse_index 个 token 被替换。"""
        text_object_embed = self.fuse_fn(object_embeds)
        out = text_embeds.clone()
        out[:, fuse_index, :] = text_object_embed.squeeze(1).to(out.dtype)
        return out


_CACHE = {}


def load_postfuse(path, dtype=torch.float16, device=None):
    """按 (路径, dtype, device) 缓存。fp16 权重 28MB，常驻显存无压力。"""
    from safetensors.torch import load_file

    if device is None:
        device = torch.device("cpu")
    key = (os.path.abspath(path), str(dtype), str(device))
    mod = _CACHE.get(key)
    if mod is not None:
        return mod

    sd = load_file(path)
    mod = PostfuseModule(embed_dim=2048, embed_dim_img=768)
    mod.load_state_dict(sd, strict=True)  # key 1:1，不匹配直接抛错
    mod = mod.to(device=device, dtype=dtype)
    mod.eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    _CACHE[key] = mod
    return mod
