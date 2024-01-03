import logging
import torch
import torch.nn as nn
from mmcv.runner.checkpoint import load_state_dict
from mmdet.models.builder import BACKBONES
from .vit import ViT, SimpleFeaturePyramid, partial
from .fpn import LastLevelMaxPool


@BACKBONES.register_module()
class EVA02(nn.Module):
    def __init__(
        self,
        # args for ViT
        img_size=1024,
        real_img_size=(256, 704),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4*2/3,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_abs_pos=True,
        pt_hw_seq_len=16,
        intp_freq=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        xattn=False,
        frozen_blocks=-1,
        # args for simple FPN
        fpn_in_feature="last_feat",
        fpn_out_channels=256,
        fpn_scale_factors=(4.0, 2.0, 1.0, 0.5),
        fpn_top_block=False,
        fpn_norm="LN",
        fpn_square_pad=0,
        pretrained=None
    ):
        super().__init__()

        self.backbone = SimpleFeaturePyramid(
            ViT(
                img_size=img_size,
                real_img_size=real_img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                use_abs_pos=use_abs_pos,
                pt_hw_seq_len=pt_hw_seq_len,
                intp_freq=intp_freq,
                window_size=window_size,
                window_block_indexes=window_block_indexes,
                residual_block_indexes=residual_block_indexes,
                use_act_checkpoint=use_act_checkpoint,
                pretrain_img_size=pretrain_img_size,
                pretrain_use_cls_token=pretrain_use_cls_token,
                out_feature=out_feature,
                xattn=xattn,
                frozen_blocks=frozen_blocks,
            ),
            in_feature=fpn_in_feature,
            out_channels=fpn_out_channels,
            scale_factors=fpn_scale_factors,
            top_block=LastLevelMaxPool() if fpn_top_block else None,
            norm=fpn_norm,
            square_pad=fpn_square_pad,
        )
        self.init_weights(pretrained)
    
    def init_weights(self, pretrained=None):
        if pretrained is None:
            return
        logging.info('Loading pretrained weights from %s' % pretrained)
        state_dict = torch.load(pretrained)['model']
        load_state_dict(self, state_dict, strict=False)

    def forward(self, x):
        outs = self.backbone(x)
        return list(outs.values())
