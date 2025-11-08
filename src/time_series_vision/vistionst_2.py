# ============================================================
# 🌎 Multimodal Groundwater Predictor (GRACE + Satellite)
# ============================================================

import torch, torch.nn as nn, torch.optim as optim
from torchvision import models, transforms
import numpy as np

# ---------- 1. Mock multimodal data ----------
n_samples = 100
grace_series = np.random.randn(n_samples, 12).astype(np.float32)    # 12-month GRACE
sat_images   = np.random.randn(n_samples, 3, 224, 224).astype(np.float32)  # Sentinel patch
dem_values   = np.random.randn(n_samples, 1).astype(np.float32)     # DEM / rainfall
targets      = np.random.randn(n_samples, 1).astype(np.float32)     # groundwater

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 2. VisionTS encoder (frozen) ----------
# from visionts.model.visionts import VisionTS
import sys, os
sys.path.append("d:/5-Projects/5-Codes/multimodal_llms_applications/VisionTS")

from visionsTS.model import VisionTS

visionts = VisionTS(img_size=64, patch_size=8, in_chans=3,
                    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.)
visionts.load_state_dict(torch.load("VisionTS/checkpoints/visionts_base.pth",
                                    map_location=device)["model"], strict=False)
visionts.to(device).eval()
for p in visionts.parameters(): p.requires_grad = False

def ts_to_img(ts, h=3, w=4):
    img = ts.reshape(h, w)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return np.stack([img]*3, axis=0)

grace_imgs = np.stack([ts_to_img(s) for s in grace_series])
x_grace = torch.tensor(grace_imgs).float().to(device)

with torch.no_grad():
    grace_feat = visionts.forward_encoder(x_grace).mean(dim=(2,3))  # (N, 768)

# ---------- 3. Satellite image encoder (frozen ViT) ----------
vit = models.vit_b_16(weights="IMAGENET1K_V1").to(device).eval()
for p in vit.parameters(): p.requires_grad = False

x_sat = torch.tensor(sat_images).float().to(device)
with torch.no_grad():
    sat_feat = vit(x_sat)   # (N, 1000) classification tokens

# ---------- 4. Fuse features + DEM ----------
x = torch.cat([
    grace_feat,
    sat_feat,
    torch.tensor(dem_values).float().to(device)
], dim=1)

# ---------- 5. Train small head to predict groundwater ----------
class FusionHead(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

head = FusionHead(x.shape[1]).to(device)
opt = optim.Adam(head.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
y = torch.tensor(targets).float().to(device)

for epoch in range(50):
    pred = head(x)
    loss = loss_fn(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if (epoch+1)%10==0:
        print(f"Epoch {epoch+1:3d} | Loss={loss.item():.4f}")

print("✅ Training complete — fused GRACE + satellite embeddings -> groundwater.")
