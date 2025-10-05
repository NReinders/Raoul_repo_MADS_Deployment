import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from settings import VAESettings
from mltrainer import vae

# ---- device selectie ----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ---- settings / data ----
presets = VAESettings(latent=2)  # latent=2 vereist dat je model ook zo is getraind
test_data = datasets.MNIST(
    root=presets.data_dir, train=False, download=True, transform=transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False)

# ---- model laden (robust) ----
modelpath = presets.modeldir / presets.modelname

def load_model(modelpath, device):
    try:
        m = torch.load(modelpath, map_location=device, weights_only=False)  # trusted
        m.to(device).eval()
        return m
    except Exception:
        m = vae.AutoEncoder(presets.model_dump()).to(device)
        state = torch.load(modelpath, map_location=device)
        m.load_state_dict(state)
        m.eval()
        return m

autoencoder = load_model(modelpath, device)

# ---- helper: NCHW -> NHWC (zoals je VAEstreamer doet) ----
def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    # [N,1,28,28] -> [N,28,28,1]
    return x.permute(0, 2, 3, 1).contiguous()

# ---- embeddings verzamelen ----
embeds, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = nchw_to_nhwc(x).to(device)     # maak NHWC voor jouw encoder
        z = autoencoder.encoder(x)         # z shape: [N, 2]
        embeds.append(z.detach().cpu())
        labels.append(y)

embeds = torch.cat(embeds).numpy()
labels = torch.cat(labels).numpy()

# ---- plot ----
plt.figure(figsize=(8, 6))
scatter = plt.scatter(embeds[:, 0], embeds[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title("2D latent space of MNIST (colored by digit label)")
plt.xlabel("z1"); plt.ylabel("z2")
plt.show()