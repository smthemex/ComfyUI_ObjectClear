import os
import argparse
from safetensors.torch import load_file
import torch
from .objectclear.pipelines import ObjectClearPipeline
from .objectclear.utils import resize_by_short_side
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

import gc


def loader_objectclear(cf_model,cf_vae,cur_path,device,cache_dir,use_fp16,postfuse_path):
     # ------------------ set up ObjectClear pipeline -------------------
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    variant = "fp16" if use_fp16 else None
    UNET = load_singlefile(cf_model,cur_path,device)
    VAE = load_singlevae(cf_vae,cur_path,device)

    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        os.path.join(cur_path,"sdxl_repo"),
        unet=UNET,
        vae=VAE,
        torch_dtype=torch_dtype,
        apply_attention_guided_fusion=True,
        cache_dir=cache_dir,
        variant=variant,
        postfuse_path=postfuse_path,

    )
    pipe.to(device)
    #pipe.enable_model_cpu_offload()
    return pipe



def inference_objectclear(pipe,input_img_list, input_mask_list,device,positive,negative,steps=20,seed=42,strength=0.99,guidance_scale=2.5,object_embeds=None,short_size=512):
    # -------------------- start to processing ---------------------
    output_img_list = []
    for i, (image, mask) in enumerate(zip(input_img_list, input_mask_list)):
        print(f'[{i+1}/{len(input_img_list)}] Processing')

        image = resize_by_short_side(image, short_size, resample=Image.BICUBIC)
        mask = resize_by_short_side(mask, short_size, resample=Image.NEAREST)

        
        w, h = image.size
        result = pipe(
            prompt=None,#"remove the instance of object",
            image=image,
            mask_image=mask,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=guidance_scale,
            height=h,
            width=w,
            return_attn_map=False,
            prompt_embeds= positive[0][0].to(device,dtype=torch.float16),
            negative_prompt_embeds= negative[0][0].to(device,dtype=torch.float16),
            pooled_prompt_embeds= positive[0][1]["pooled_output"].to(device, dtype=torch.float16),
            negative_pooled_prompt_embeds= negative[0][1]["pooled_output"].to(device, dtype=torch.float16),
            object_embeds=object_embeds[i].to(device, dtype=torch.float16),
        )
        
        fused_img_pil = result.images[0]
        output_img_list.append(fused_img_pil)

        # save results
        # save_path = os.path.join(result_root, f'{basename}.png')
        # fused_img_pil = fused_img_pil.resize(image_or.size)
        # fused_img_pil.save(save_path)

    #print(f'\nAll results are saved in {result_root}')
    return output_img_list



def load_singlefile(model,cur_path,device):
    from diffusers import UNet2DConditionModel
    
    config_file = os.path.join(cur_path,"sdxl_repo/unet/config.json")
    unet_state_dict=load_file(model)
    unet_config = UNet2DConditionModel.load_config(config_file)
    Unet = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
    Unet.load_state_dict(unet_state_dict, strict=False)
    del unet_state_dict
    gc.collect()
    torch.cuda.empty_cache()
    return Unet


def load_singlevae(VAE,cur_path,device):
    from diffusers import AutoencoderKL

    vae_config = os.path.join(cur_path, "sdxl_repo/vae/config.json")
    vae_state_dict=load_file(VAE)
    ae_config = AutoencoderKL.load_config(vae_config)
    AE = AutoencoderKL.from_config(ae_config).to(device)
    AE.load_state_dict(vae_state_dict, strict=False)
    del vae_state_dict
    torch.cuda.empty_cache()
    return AE