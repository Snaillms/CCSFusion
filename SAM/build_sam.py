# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .image_encoder import ImageEncoderViT #, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from .sam import Sam

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_medsam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

sam_model_registry = {
    "default": build_sam,
    "vit_h": build_sam,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "medsam_vit_b": build_medsam_vit_b,
}


def _build_sam(
    encoder_embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=12,
    encoder_global_attn_indexes=[2, 5, 8, 11],
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
        ,
        # prompt_encoder=PromptEncoder(
        #     embed_dim=prompt_embed_dim,
        #     image_embedding_size=(image_embedding_size, image_embedding_size),
        #     input_image_size=(image_size, image_size),
        #     mask_in_chans=16,
        # ),
        # mask_decoder=MaskDecoder(
        #     num_multimask_outputs=3,
        #     transformer=TwoWayTransformer(
        #         depth=2,
        #         embedding_dim=prompt_embed_dim,
        #         mlp_dim=2048,
        #         num_heads=8,
        #     ),
        #     transformer_dim=prompt_embed_dim,
        #     iou_head_depth=3,
        #     iou_head_hidden_dim=256,
        # ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )


    #TODO

    sam.eval()
    # 从完整的SAM模型中提取图像编码器部分的过程
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

            print(type(state_dict))
            # for k, v in state_dict.items():  # k 是键名 v 对应的值
            #     print(k)
            
            new_dict = {key:val for key, val in state_dict.items() if key.split('.')[0]=='image_encoder'} 
            print('//////////////////////////')
            for k, v in new_dict.items():  # k 是键名 v 对应的值
                print(k)
            torch.save(new_dict, '/data2/lms/Model/SAM/sam_encoder_vit_b_01ec64.pth')
        print('done!!!!!!!!!!!!!!')

        # 加载图像编码器部分--老是报错  
        # 这是因为您创建的sam对象只有image_encoder部分，但权重文件包含了完整模型的参数
        # sam.load_state_dict(state_dict)
    return sam
