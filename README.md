# ComfyUI_ObjectClear
[ObjectClear](https://github.com/zjx0101/ObjectClear):Complete Object Removal via Object-Effect Attention,you can try it in ComfyUI


# New
* 此方法优势在于内绘不仅限于主体，比如去除物体，物体的影子也会去掉，表现优异。

# 1 . Installation /安装

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ ComfyUI_ObjectClear.git
```
---

# 2 . Requirements  
* 通常不需要安装，因为没什么特别的库

```
pip install -r requirements.txt
```

# 3 . Model/模型
* [jixin0101/ObjectClear](https://huggingface.co/jixin0101/ObjectClear/tree/main)  download files as below /下载下方列出的模型:
```
├── your comfyUI path/models/checkpoints/
|   ├──diffusion_pytorch_model.safetensors     #  unet dir/ 目录，rename or not  重命名或者随你，可以下fp16的
├── your comfyUI path/models/vae/
|   ├──diffusion_pytorch_model.safetensors     #  vae dir/目录，rename or not  重命名或者随你
├── models/clip_vision/
|   ├── diffusion_pytorch_model.safetensors    #  image_prompt_encoder 目录，其实就是这个： clip-vit-large-patch14.safetensors
├── models/clip/
|   ├── model.safetensors                                      # postfuse_module dir /目录 rename or not  重命名或者随你 ，可以下fp16 
|   ├── clip_l.safetensors                                        # normal comfyUI clip 常规comfy的clip
|   ├── clip_g.safetensors                                       # normal comfyUI clip 常规comfy的clip

```
* seg /遮罩没有内置，随便用其他的吧。

# 4 . Example
![](https://github.com/smthemex/ComfyUI_ObjectClear/blob/main/example_mult.png)
![](https://github.com/smthemex/ComfyUI_ObjectClear/blob/main/example.png)


# 5 .Tips/使用说明
* 支持单图或多图，遮罩合并用插件带的batch节点，512是垫图最小短边尺寸（会自动裁切），模型用512*512训练的，按理此尺寸是最好的。
* Support single or multiple images, use batch nodes with plugins for mask merging, 512 is the input image ‘s minimum short edge size, and the model is trained with 512 * 512, which is theoretically the best size.

#  Citation
```
@InProceedings{zhao2025ObjectClear,
    title     = {{ObjectClear}: Complete Object Removal via Object-Effect Attention},
    author    = {Zhao, Jixin and Zhou, Shangchen and Wang, Zhouxia and Yang, Peiqing and Loy, Chen Change},
    booktitle = {arXiv preprint arXiv:2505.22636},
    year      = {2025}
    }
```
