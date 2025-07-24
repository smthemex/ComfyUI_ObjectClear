# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from omegaconf import OmegaConf
from pathlib import PureWindowsPath
import yaml
        
from .node_utils import gc_cleanup,tensor2pil_list,load_images,mask2pil_list,tensor_upscale
from .inference_objectclear import loader_objectclear,inference_objectclear
import folder_paths


########
MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")


######


class ObjectClearLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint":(["none"] + folder_paths.get_filename_list("checkpoints"),),
                "vae":(["none"] + folder_paths.get_filename_list("vae"),),
                "clip":(["none"] + folder_paths.get_filename_list("clip"),),
                "use_fp16":("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("MODEL_ObjectClear",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "ObjectClear"

    def loader_main(self,checkpoint,vae,clip,use_fp16,):


        # load model
        print("***********Load model ***********")

        if clip == "none" :
            raise Exception("Please select a clip model")

        else:
            weight_path=folder_paths.get_full_path("clip", clip)
            
        if vae == "none" :
            raise Exception("Please select a vae model")

        else:
            vae_path=folder_paths.get_full_path("vae", vae)

        
        if checkpoint == "none" :
            raise Exception("Please select a checkpoint model")

        else:
            model_path=folder_paths.get_full_path("checkpoints", checkpoint)
        pipe = loader_objectclear(model_path,vae_path,current_node_path,device,None,use_fp16,weight_path)

        print("***********Load model done ***********")

        gc_cleanup()

        return (pipe,)



class ObjectClearSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_ObjectClear",),
                "iamge": ("IMAGE",),
                "mask": ("MASK",),
                "positive": ("CONDITIONING", {
                "tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "vison_emb":("CLIP_VISION_OUTPUT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED,
                                 "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 20.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "strength": (
                    "FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "short_size": (
                    "INT", {"default": 512, "min": 512, "max": 2048, "step": 64,}),

            }}

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "sampler_main"
    CATEGORY = "ObjectClear"

    def sampler_main(self,model,iamge,mask,positive,negative,vison_emb,seed,steps,cfg,strength,short_size):

        #object_embeds=vison_emb["image_embeds"].view(1, 1, -1) # [1, 1, 768]
        B = vison_emb["image_embeds"].shape[0]
        obj_embeds = vison_emb["image_embeds"].view(B, 1, -1)
        if B==1:
            object_embeds=[obj_embeds]
        else:
            object_embeds=list(torch.chunk(obj_embeds, chunks=B))
        images_list=tensor2pil_list(iamge)
        masks_list=mask2pil_list(mask)
        images=inference_objectclear(model,images_list,masks_list,device,positive,negative,steps,seed,strength,cfg,object_embeds,short_size)
        gc.collect()
        torch.cuda.empty_cache()
        return (load_images(images), )


class ObjectClearVision:
    def __init__(self): 
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "iamge": ("IMAGE",),
                "mask": ("MASK",),
            }}

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "sampler_main"
    CATEGORY = "ObjectClear"

    def sampler_main(self,iamge,mask):
        B,_,_,_=iamge.size()
        B_,_,_=mask.size()
        if B!=B_:
            raise ValueError("input image and mask must have same batch size")
        else:
            if B==1:
                mask=mask.unsqueeze(-1) 
                img=iamge * (mask > 0.5)
                return (img, )
            else:
                mask_list=list(torch.chunk(mask, chunks=B))
                img_list=list(torch.chunk(iamge, chunks=B))
                masked_img_list=[]
                for mask_,img_ in zip(mask_list,img_list):
                    msk=mask_.unsqueeze(-1) 
                    image_=img_ * (msk > 0.5)
                    masked_img_list.append(image_)
                return (torch.cat(masked_img_list, dim=0), )

class ObjectClearBatch:
    def __init__(self): 
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",), # B,h,w
                "mask2": ("MASK",),
            }}

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask",)
    FUNCTION = "sampler_main"
    CATEGORY = "ObjectClear"

    def sampler_main(self,mask1,mask2):

        B1,height,width=mask1.size()
        B2,_,_=mask2.size()
        

        mask1=tensor_upscale(mask1.unsqueeze(-1) ,width,height).squeeze(-1)
        mask2=tensor_upscale(mask2.unsqueeze(-1) ,width,height).squeeze(-1)
        
        if B1==B2:  
            return (torch.cat((mask1,mask2), dim=0), )    
        else:
            out_list = []
            if B1==1:
                out_list.append(mask1)
            else:
                for i  in list(torch.chunk(mask1, chunks=B1)):
                    out_list.append(i)

            if B2==1:
                out_list.append(mask2)
            else:
                for i  in list(torch.chunk(mask2, chunks=B2)):
                    out_list.append(i)
            return (torch.cat(out_list, dim=0), )
NODE_CLASS_MAPPINGS = {

    "ObjectClearLoader": ObjectClearLoader,
    "ObjectClearSampler": ObjectClearSampler,
    "ObjectClearVision": ObjectClearVision,
    "ObjectClearBatch": ObjectClearBatch,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ObjectClearLoader": "ObjectClearLoader",
    "ObjectClearSampler": "ObjectClearSampler",
    "ObjectClearVision": "ObjectClearVision",
    "ObjectClearBatch": "ObjectClearBatch"

}
