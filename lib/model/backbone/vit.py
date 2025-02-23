import torch
import torch.nn as nn
from functools import partial
import timm
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(
        self,
        global_pool="avg",
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super(VisionTransformer, self).__init__(
            global_pool=global_pool,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
        )

        pretrained = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
        weight_dict = pretrained.state_dict()
        weight_dict.update({'fc_norm.weight': weight_dict["norm.weight"]})
        weight_dict.update({'fc_norm.bias': weight_dict["norm.bias"]})
        # self.load_state_dict(weight_dict, strict=False)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_ls = []
        for blk in self.blocks:
            x = blk(x)
            x_ls.append(x)
        x = self.norm(x)
        return x, x_ls
