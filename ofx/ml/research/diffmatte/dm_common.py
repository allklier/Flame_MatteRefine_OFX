# DiffMatte model construction (mirrors configs/common/model.py + configs/ViT{S,B}.py of
# YihanHu-2022/DiffMatte @ 12082fd) and a tile wrapper with the DDIM loop unrolled, taking the
# same [1,4,T,T] input as ViTMatte in Matte Refine ML (RGB (x-0.5)/0.5, trimap 0/0.5/1).
import os, sys
from functools import partial
import torch, torch.nn as nn

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(B, "dm_src"))  # .py-only copy of the repo's modeling/
from modeling.meta_arch.difmodel import DifModel          # noqa: E402
from modeling.decoder.DUNet import DUNet                  # noqa: E402
from modeling.backbone import create_backbone             # noqa: E402
from modeling.diffusion.script_utils import create_uniform_gaussian_diffusion  # noqa: E402

VARIANTS = {  # name: (embed_dim, heads, img_size)
    "vits": (384, 6, 512), "vitb": (768, 12, 512), "vits1024": (384, 6, 1024)}
MEAN = [123.675 / 255., 116.280 / 255., 103.530 / 255.]
STD = [58.395 / 255., 57.120 / 255., 57.375 / 255.]
SCALE = 0.2  # configs/ViT*.py: diffusion.scale = 0.2


def build(variant, weights):
    dim, heads, img = VARIANTS[variant]
    backbone = create_backbone(name="ViT", kwargs=dict(
        in_chans=4, img_size=img, patch_size=16, embed_dim=dim, depth=12, num_heads=heads,
        drop_path_rate=0, window_size=14, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10], residual_block_indexes=[2, 5, 8, 11],
        use_rel_pos=True, out_feature="last_feat"))
    decoder = DUNet(model_channels=32, emb_channels=32, downsample_in=[7, 32, 64, 128],
                    upsample_in=[dim, 256, 128, 64, 32])
    m = DifModel(backbone=backbone, pixel_mean=MEAN, pixel_std=STD, decoder=decoder,
                 backbone_in_channel=4, decoder_in_channel=7)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = m.load_state_dict(sd, strict=True), None
    return m.eval()


def diffusion(steps):
    # configs/common/diffusion.py (uniform_timesteps=True branch) with the ViT* overrides and
    # run_one_image.py's "ddim<steps>" (use_ddim=True, diffusion.steps=<steps>).
    return create_uniform_gaussian_diffusion(schedule_kwargs={}, time_difference=0., scale=SCALE,
                                             steps=steps, uniform_timesteps=True,
                                             noise_schedule="linear", predict_xstart=True,
                                             criterion=None, jump_step=-1)


def reference(m, steps, rgb01, trimap, noise):
    """The authors' path: UniformGauss.ddim_sample + DifMatte.forward's post-processing.
    rgb01 [1,3,H,W] 0-1, trimap [1,1,H,W] 0/0.5/1, noise [1,1,H,W]. Returns [1,1,H,W] 0-1."""
    d = diffusion(steps)
    cond = {"image": rgb01, "trimap": trimap}
    with torch.no_grad():
        s = d.ddim_sample(m, trimap.shape, cond, noise=noise.clone())
    a = torch.clamp(s, 0., 1.)
    a[trimap == 0] = 0.0  # DifMatte.forward; its "trimap == 2 -> 1" never fires (trimap is 0/0.5/1)
    return a


class Tile(nn.Module):
    """[1,4,T,T] ViTMatte-style input -> [1,1,T,T] alpha, DDIM with `steps` unrolled, fixed noise."""
    def __init__(self, m, steps, noise, mask_clear=True):
        super().__init__()
        self.m = m
        self.register_buffer("noise", noise)  # [1,1,T,T]
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        times = torch.linspace(1., 0., steps + 1)  # UniformGauss.get_sampling_timesteps
        self.pairs = [(times[i:i + 1].clone(), times[i + 1:i + 2].clone()) for i in range(steps)]
        self.mask_clear = mask_clear

    def forward(self, x):
        rgb = x[:, 0:3] * 0.5 + 0.5
        tri = x[:, 3:4]
        img = (rgb - self.mean) / self.std
        feats = self.m.backbone(torch.cat([img, tri], 1))  # once
        onehot = torch.cat([(tri < 0.25), (tri >= 0.25) & (tri < 0.75), (tri >= 0.75)], 1).to(x.dtype)
        cond = torch.cat([img, onehot], 1) * 2 - 1
        xt = self.noise
        for t, tn in self.pairs:  # decoder only, `steps` times
            g, gn = (1 - t).clamp(min=1e-9), (1 - tn).clamp(min=1e-9)
            a, s = torch.sqrt(g) * SCALE, torch.sqrt(1 - g)
            an, sn = torch.sqrt(gn) * SCALE, torch.sqrt(1 - gn)
            std = torch.std(xt, dim=(1, 2, 3), keepdim=True, unbiased=False).clamp(min=1e-5)
            pred = self.m.decoder(feats, torch.cat([cond, xt / std], 1), t)["phas"].clamp(-1., 1.)
            eps = (xt - a.view(1, 1, 1, 1) * pred) / s.clamp(min=1e-10).view(1, 1, 1, 1)
            xt = pred * an.view(1, 1, 1, 1) + eps * sn.view(1, 1, 1, 1)
        out = ((xt / SCALE) + 1) * 0.5
        out = out.clamp(0., 1.)
        if self.mask_clear:
            out = out * (tri != 0).to(x.dtype)
        return out


def make_noise(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, T, T, generator=g)
