"""Global/local degradation adapters (paper equations 6-11).

The reference must be a geometrically aligned pseudo-clear image. This module
does not synthesize a clear reference or infer semantic class names.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DegradationAdapter(nn.Module):
    def __init__(self, embed_dim, out_chans=256, stages=4):
        super().__init__()
        self.global_encoder = nn.Sequential(
            nn.Conv2d(6, 64, 7, 2, 3), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))
        self.local_encoder = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1, groups=3),
            nn.Conv2d(3, 64, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.Conv2d(64, 64, 1), nn.GELU())
        self.global_maps = nn.ModuleList([nn.Linear(256, embed_dim) for _ in range(stages)])
        self.local_maps = nn.ModuleList([nn.Conv2d(64, embed_dim, 1) for _ in range(stages)])
        self.projections = nn.ModuleList([nn.Conv2d(embed_dim, out_chans, 1) for _ in range(stages)])
        self.fusion = nn.Conv2d(stages * out_chans, out_chans, 1)
        # Preserve pretrained SAM exactly at initialization.
        for layer in list(self.global_maps) + list(self.local_maps) + [self.fusion]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def cues(self, image, reference, token_hw):
        if reference is None or reference.shape != image.shape:
            raise ValueError('Adapted SAM requires an aligned reference with the same BCHW shape.')
        # Bound the global-branch cost; the local branch stays token aligned.
        pair = F.interpolate(torch.cat([image, reference], 1), (256, 256), mode='bilinear', align_corners=False)
        difference = F.interpolate((reference - image).abs(), token_hw, mode='bilinear', align_corners=False)
        return self.global_encoder(pair), self.local_encoder(difference)

    def inject(self, tokens, cues, stage):
        global_cue, local_cue = cues
        delta = self.global_maps[stage](global_cue)[:, None, None, :]
        delta = delta + self.local_maps[stage](local_cue).permute(0, 2, 3, 1)
        return tokens + delta

    def aggregate(self, features):
        projected = [layer(x.permute(0, 3, 1, 2)) for layer, x in zip(self.projections, features)]
        return self.fusion(torch.cat(projected, 1))
