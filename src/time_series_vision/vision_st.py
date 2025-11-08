# ============================================================
# 🌎 VisionTS Frozen Feature Extractor + Forecast Head Example
# (Fixed import: uses VisionTS GitHub repo structure)
# ============================================================

# --- Install dependencies (first run once) ---
# pip install torch torchvision timm einops matplotlib numpy pandas scikit-learn
# git clone https://github.com/Keytoyze/VisionTS.git
# cd VisionTS
# pip install -e .

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from visionts.model.visionts import VisionTS   # ✅ correct import

# ---- 1. Load pretrained VisionTS checkpoint ----
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "VisionTS/checkpoints/visionts_base.pth"  # update path if different

model = VisionTS(
    img_size=64, patch_size=8, in_chans=3,
    embed_dim=768, depth=12, num_heads=12,
    mlp_ratio=4., qkv_bias=True
)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state["model"], strict=False)
model.to(device)
model.eval()
for p in model.parameters(): p.requires_grad = False

print("✅ VisionTS loaded and frozen.")

# ---- 2. Create synthetic GRACE-like data (12 months × 10 regions) ----
n_regions, seq_len = 10, 12
np.random.seed(0)
grace_series = np.random.randn(n_regions, seq_len).astype(np.float32)

def series_to_image(series, h=3, w=4):
    img = series.reshape(h, w)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.stack([img, img, img], axis=0)
    return img

images = np.stack([series_to_image(s) for s in grace_series])  # (N, 3, H, W)
x = torch.tensor(images).float().to(device)

# ---- 3. Extract frozen embeddings ----
with torch.no_grad():
    latent = model.forward_encoder(x)   # encoder output before forecast head
img_emb = latent.mean(dim=(2,3)).cpu().numpy()  # global average pooling

print("Embedding shape:", img_emb.shape)

# ---- 4. Small forecasting head (trainable) ----
class ForecastHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

head = ForecastHead(img_emb.shape[1]).to(device)
opt = optim.Adam(head.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

y = torch.randn(n_regions, 1).to(device)
x_emb = torch.tensor(img_emb).to(device)

# ---- 5. Train forecasting head ----
for epoch in range(100):
    head.train()
    opt.zero_grad()
    pred = head(x_emb)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss = {loss.item():.4f}")

# ---- 6. Inference and visualization ----
head.eval()
with torch.no_grad():
    y_pred = head(x_emb).cpu().numpy()
plt.figure(figsize=(6,3))
plt.plot(y.cpu(), "o-", label="True")
plt.plot(y_pred, "x--", label="Predicted")
plt.legend(); plt.tight_layout(); plt.show()

np.save("grace_embeddings.npy", img_emb)
print("💾 Saved embeddings to grace_embeddings.npy")
