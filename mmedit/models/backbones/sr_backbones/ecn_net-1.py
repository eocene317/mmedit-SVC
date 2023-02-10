# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint
from torch.nn.modules.utils import _pair

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmcv.ops import DeformConv2dPack, DeformConv2d
import math

@BACKBONES.register_module()
class ECNNet(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_frames (int): Number of input frames. Default: 5.
        deform_groups (int): Deformable groups. Defaults: 8.
        num_blocks_extraction (int): Number of blocks for feature extraction.
            Default: 5.
        num_blocks_reconstruction (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        with_tsa (bool): Whether to use TSA module. Default: True.
    """

    def __init__(self):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.fusion = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=9, padding=9 // 2),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=5 // 2),
            nn.PReLU(),
            nn.Conv2d(32, 3, kernel_size=5, padding=5 // 2)
        )

        # learn the offset
        self.conv1 = nn.Conv2d(6, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 18, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

        self.deform_conv1 = DeformConv2d(3, 3, kernel_size=3, padding=1, im2col_step=256)

    def forward(self, bl, el):
        """Forward function for ECNNET.
        x: (n, t, c, h, w) 表示输入的视频连续t帧来自基础层低分辨率图像(t特殊处理了这里是5)
        x1: (n, t, c, h, w) 表示输入的视频连续t帧来自基础层高分辨率图像(t特殊处理了这里是3)
        """
        n, t, c, h, w = bl.size()
        n1, t1, c1, h1, w1 = el.size()

        lr0_patch = bl[:, t // 2 - 1, ...]
        lr_patch = bl[:, t // 2, ...]
        hr0_patch = el[:, t1 // 2 - 1, ...]

        lr0_up = self.upsample(lr0_patch)
        lr0_patch_up = torch.cat([lr0_up, hr0_patch], dim=1)
        res = self.relu(self.conv1(lr0_patch_up))
        res = self.relu(self.conv2(res))
        res = self.conv3(res)

        lr_patch_up = self.upsample(lr_patch)
        x = self.deform_conv1(lr_patch_up, res)
        
        res0 = lr0_up - hr0_patch
        res = torch.cat([res0, x], dim=1)
        res = self.fusion(res)

        return x + res

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
