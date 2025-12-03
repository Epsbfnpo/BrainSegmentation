from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from .blocks import ConvHead, DownBlock, MedNeXtBlock, UpBlock, _get_conv


class MedNeXt(nn.Module):
    """A compact 3D MedNeXt-style UNet for segmentation."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        n_channels: int = 32,
        n_classes: int = 87,
        block_counts: Sequence[int] = (2, 2, 2, 2, 2),
        exp_r: int | Sequence[int] = 4,
        kernel_size: int = 7,
        drop_path_rate: float = 0.0,
        deep_supervision: bool = True,
        do_res: bool = True,
        do_res_up_down: bool = True,
        dim: str = "3d",
        grn: bool = True,
    ):
        super().__init__()
        assert dim in {"2d", "3d"}, "dim must be '2d' or '3d'"
        self.dim = 3 if dim == "3d" else 2
        self.deep_supervision = deep_supervision

        if isinstance(exp_r, int):
            exp_r = [exp_r] * max(len(block_counts), 5)
        exp_r_list = list(exp_r)

        if len(block_counts) < 5:
            raise ValueError("block_counts should provide at least 5 stages")
        if len(block_counts) > 5:
            enc_counts = block_counts[:5]
            dec_counts = block_counts[5:9]
            if len(dec_counts) < 4:
                dec_counts = dec_counts + [dec_counts[-1]] * (4 - len(dec_counts))
        else:
            enc_counts = list(block_counts)
            dec_counts = [block_counts[-2], block_counts[-3], block_counts[-4], block_counts[-5]]

        channels = [n_channels * (2 ** i) for i in range(5)]

        self.stem = _get_conv(self.dim, in_channels, channels[0], kernel_size=3)

        self.enc_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            channels[i],
                            dim=self.dim,
                            expansion=exp_r_list[min(i, len(exp_r_list) - 1)],
                            drop_path=drop_path_rate,
                            use_grn=grn,
                        )
                        for _ in range(enc_counts[i])
                    ]
                )
                for i in range(5)
            ]
        )
        self.downs = nn.ModuleList(
            [DownBlock(channels[i], channels[i + 1], dim=self.dim) for i in range(4)]
        )

        self.ups = nn.ModuleList(
            [UpBlock(channels[i + 1], channels[i], dim=self.dim) for i in reversed(range(4))]
        )
        self.fuse = nn.ModuleList(
            [
                _get_conv(self.dim, channels[i] * 2, channels[i], kernel_size=1, padding=0)
                for i in reversed(range(4))
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            channels[i],
                            dim=self.dim,
                            expansion=exp_r_list[min(i, len(exp_r_list) - 1)],
                            drop_path=drop_path_rate,
                            use_grn=grn,
                        )
                        for _ in range(dec_counts[::-1][idx])
                    ]
                )
                for idx, i in enumerate(reversed(range(4)))
            ]
        )

        self.head = ConvHead(channels[0], n_classes, dim=self.dim)
        if deep_supervision:
            self.aux_heads = nn.ModuleList(
                [ConvHead(channels[i], n_classes, dim=self.dim) for i in reversed(range(1, 4))]
            )
        else:
            self.aux_heads = None

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        enc_feats: List[torch.Tensor] = []
        for i in range(5):
            x = self.enc_blocks[i](x)
            enc_feats.append(x)
            if i < 4:
                x = self.downs[i](x)

        outputs: List[torch.Tensor] = []
        for idx, (up, fuse, dec) in enumerate(zip(self.ups, self.fuse, self.dec_blocks)):
            x = up(x)
            skip = enc_feats[3 - idx]
            x = torch.cat([x, skip], dim=1)
            x = fuse(x)
            x = dec(x)
            if self.deep_supervision and idx < 3:
                outputs.append(self.aux_heads[idx](x))

        out = self.head(x)
        if self.deep_supervision:
            outputs.append(out)
            return outputs
        return out
