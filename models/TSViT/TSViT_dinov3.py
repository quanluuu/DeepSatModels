import os
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.TSViT.TSViTdense import Transformer
from data.PASTIS24.data_transforms import Normalize

# normalization of DINOv3 web (LVD-1689M) weights, from the facebookresearch/dinov3 README
DINOV3_MEAN = (0.485, 0.456, 0.406)
DINOV3_STD = (0.229, 0.224, 0.225)


def amp_dtype(device):
    """bf16 on Ampere+, fp16 on Volta/Turing, None (fp32) on CPU and Pascal where fp16 is slow"""
    if device.type != 'cuda':
        return None
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        return torch.bfloat16
    if major >= 7:
        return torch.float16
    return None


class TSViT_DINOv3(nn.Module):
    """
    TSViT with the linear patch embedding replaced by a frozen DINOv3 encoder applied to each RGB frame.
    Each frame is resized by 16/patch_size so the DINOv3 token grid matches the TSViT patch grid.
    """
    def __init__(self, model_config, backbone=None):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = model_config['num_classes']
        self.dim = model_config['dim']
        self.temporal_depth = model_config['temporal_depth']
        self.spatial_depth = model_config['spatial_depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.scale_dim = model_config['scale_dim']
        self.rgb_idx = list(model_config['rgb_idx'])
        self.rgb_scale = model_config['rgb_scale']
        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        if backbone is None:
            repo = model_config['dinov3_repo']
            backbone = torch.hub.load(repo, 'dinov3_vits16', source='local' if os.path.isdir(repo) else 'github',
                                      weights=model_config['dinov3_weights'])
        self.backbone = backbone
        self.backbone.requires_grad_(False).eval()
        assert self.backbone.patch_size % self.patch_size == 0, 'DINOv3 patch size must be a multiple of patch_size'
        self.backbone_res = self.image_size * self.backbone.patch_size // self.patch_size

        # inputs arrive normalized per band by the PASTIS transforms; these stats undo it for the RGB bands
        stats = Normalize()
        self.register_buffer('data_mean', torch.from_numpy(stats.mean_fold1[0, self.rgb_idx, 0, 0]), persistent=False)
        self.register_buffer('data_std', torch.from_numpy(stats.std_fold1[0, self.rgb_idx, 0, 0]), persistent=False)
        self.register_buffer('dinov3_mean', torch.tensor(DINOV3_MEAN), persistent=False)
        self.register_buffer('dinov3_std', torch.tensor(DINOV3_STD), persistent=False)

        num_patches = self.num_patches_1d ** 2
        self.to_patch_embedding = nn.Linear(self.backbone.embed_dim, self.dim)
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, model_config['dropout'])
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head,
                                             self.dim * self.scale_dim, model_config['dropout'])
        self.dropout = nn.Dropout(model_config['emb_dropout'])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # frozen backbone always in eval mode
        return self

    def rgb_to_backbone_input(self, rgb):
        """(B, T, 3, H, W) normalized RGB -> (B*T, 3, R, R) in DINOv3 input space"""
        rgb = rgb.flatten(0, 1)
        rgb = rgb * self.data_std.view(1, 3, 1, 1) + self.data_mean.view(1, 3, 1, 1)  # back to reflectance
        rgb = (rgb / self.rgb_scale).clamp(0, 1)
        rgb = (rgb - self.dinov3_mean.view(1, 3, 1, 1)) / self.dinov3_std.view(1, 3, 1, 1)
        return F.interpolate(rgb, size=(self.backbone_res, self.backbone_res), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)
        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)

        frames = self.rgb_to_backbone_input(x[:, :, self.rgb_idx])
        dtype = amp_dtype(frames.device)
        with torch.no_grad(), torch.autocast(device_type=frames.device.type, dtype=dtype, enabled=dtype is not None):
            tokens = self.backbone.forward_features(frames)['x_norm_patchtokens']  # (B*T, h*w, D), row-major grid
        x = self.to_patch_embedding(tokens.float())
        x = rearrange(x, '(b t) n d -> b n t d', b=B)

        x = x + temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2)

        x = rearrange(x, 'b c (h1 w1) (p1 p2) -> b c (h1 p1) (w1 p2)',
                      h1=self.num_patches_1d, w1=self.num_patches_1d,
                      p1=self.patch_size, p2=self.patch_size)
        return x


if __name__ == "__main__":
    # self-check with a stub backbone (no weights, no GPU): python -m models.TSViT.TSViT_dinov3
    class StubBackbone(nn.Module):
        embed_dim = 384
        patch_size = 16

        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, self.embed_dim, self.patch_size, self.patch_size)
            self.last_input_shape = None

        def forward_features(self, x):
            self.last_input_shape = tuple(x.shape)
            return {'x_norm_patchtokens': self.proj(x).flatten(2).transpose(1, 2)}

    config = {'img_res': 24, 'patch_size': 4, 'num_classes': 19, 'max_seq_len': 6, 'num_channels': 11,
              'dim': 128, 'temporal_depth': 2, 'spatial_depth': 2, 'heads': 4, 'dim_head': 32,
              'dropout': 0., 'emb_dropout': 0., 'scale_dim': 4, 'rgb_idx': [2, 1, 0], 'rgb_scale': 3000.}
    stub = StubBackbone()
    model = TSViT_DINOv3(config, backbone=stub)

    B, T = 2, 6
    x = torch.randn(B, T, 24, 24, 11)
    x[..., -1] = torch.randint(0, 365, (B, T, 1, 1)).float() / 365.0001  # day-of-year channel, as in PASTIS transforms
    out = model(x)
    assert out.shape == (B, 19, 24, 24), out.shape
    assert stub.last_input_shape == (B * T, 3, 96, 96), stub.last_input_shape

    # frozen backbone: no grads; projection gets grads
    out.mean().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert model.to_patch_embedding.weight.grad is not None

    # backbone stays in eval mode even when the model is put in train mode
    model.train()
    assert model.backbone.training is False and model.temporal_transformer.training is True

    # radiometry: a pixel with reflectance == rgb_scale maps to (1 - mean) / std of the DINOv3 normalization
    refl = torch.full((1, 1, 3, 1, 1), 3000.)
    normed = (refl - model.data_mean.view(1, 1, 3, 1, 1)) / model.data_std.view(1, 1, 3, 1, 1)
    got = model.rgb_to_backbone_input(normed)
    want = (1 - model.dinov3_mean) / model.dinov3_std
    assert torch.allclose(got.mean(dim=(2, 3)), want.view(1, 3), atol=1e-5), (got.mean(dim=(2, 3)), want)

    # warm-start: only temporal params come from the old TSViT checkpoint (skipped if the file is absent)
    ckpt_path = 'TSViT_fold1_full_a100_opt/best.pth'
    if os.path.isfile(ckpt_path):
        from utils.torch_utils import load_from_checkpoint
        prefixes = ["to_temporal_embedding_input.", "temporal_token", "temporal_transformer."]
        model = TSViT_DINOv3({**config, 'temporal_depth': 4, 'spatial_depth': 4}, backbone=StubBackbone())
        before = {k: v.clone() for k, v in model.state_dict().items()}
        load_from_checkpoint(model, ckpt_path, partial_restore=True, device='cpu', restore_prefixes=prefixes)
        saved = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        for k, v in model.state_dict().items():
            if k.startswith(tuple(prefixes)):
                assert torch.equal(v, saved[k]), k
            else:
                assert torch.equal(v, before[k]), k
        print("warm-start check passed")
    print("TSViT_DINOv3 self-check passed")
