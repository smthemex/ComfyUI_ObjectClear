"""ObjectClear 原生实现（不依赖 diffusers / transformers）。

参考：ComfyUI模型封装通用方法论.md / ComfyUI_MS_Diffusion核心封装模式总结.md
目标链路：
    原生 CheckpointLoaderSimple(ObjectClear_fp16) + 原生 VAELoader + 原生 CLIP / CLIPVision
      -> ObjectClearPostfuseLoader   (只加载 28MB 的 postfuse_module)
      -> ObjectClearConditioning     (CLIP-Vision 物体嵌入 -> postfuse -> 替换第 5 个文本 token)
      -> ObjectClearSampler          (原生 KSampler 内核 + 9 通道 inpaint concat + 末步注意力捕获)
      -> 原生 VAEDecode
      -> ObjectClearAGF             (注意力引导融合，纯 torch 移植)
"""

FUSE_INDEX = 5           # 官方 postfuse / 注意力图都取第 5 个文本 token
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# 官方 resize_attn_map_divide2 里 attention 图相对 latent 的下采样倍率
ATTN_DOWNSCALE = 2

# 官方 AGF 超参（写死，保持与上游一致）
AGF_DILATE_KERNEL = 21
AGF_BLUR_KERNEL = 9
AGF_BLUR_SIGMA = 2.0
AGF_THRESHOLD = 128
AGF_WAVELET_LEVELS = 5
AGF_PAD_MULTIPLE = 8
