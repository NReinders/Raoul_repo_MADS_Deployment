import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor

# ====== jouw modules uit de repo ======
from settings import VAESettings, VAEstreamer
from mltrainer import vae

# Allow torch.load to instantiate the VAE classes stored in the checkpoint
torch.serialization.add_safe_globals([
    vae.AutoEncoder,
    vae.Encoder,
    vae.Decoder,
])

# ---------- laad settings & model ----------
presets = VAESettings()
modelpath = presets.modeldir / presets.modelname

device = torch.device("cpu")  # of "mps"/"cuda" als je dat wilt en je model zo is opgeslagen
autoencoder = torch.load(modelpath, map_location=device, weights_only=False)
autoencoder.eval()

# ---------- haal een batch uit de testset ----------
test_data = datasets.MNIST(root=presets.data_dir, train=False, download=True, transform=ToTensor())
teststreamer = VAEstreamer(test_data, batchsize=16).stream()
X1, X2 = next(teststreamer)  # X2 == target == X1
# X1-shape is waarschijnlijk [N, 28, 28, 1] (channels-last)

# ---------- model forward ----------
with torch.no_grad():
    # Zorg dat shapes kloppen voor jouw model (veel code in repo gebruikt NHWC)
    x_in = X1.to(device)
    x_hat = autoencoder(x_in)

# ---------- naar numpy voor plot ----------
def to_np(img_batch):
    # img_batch: [N,H,W,1] of [N,1,H,W]
    if img_batch.ndim == 4 and img_batch.shape[-1] == 1:   # NHWC
        imgs = img_batch[..., 0].cpu().numpy()
    elif img_batch.ndim == 4 and img_batch.shape[1] == 1:  # NCHW
        imgs = img_batch[:, 0, ...].cpu().numpy()
    else:
        raise ValueError(f"Unexpected shape: {img_batch.shape}")
    return imgs

orig = to_np(X1)
reco = to_np(x_hat)

# ---------- plot ----------
n = min(8, orig.shape[0])
plt.figure(figsize=(2*n, 4))
for i in range(n):
    # originals
    ax = plt.subplot(2, n, i + 1)
    ax.imshow(orig[i], cmap="gray")
    ax.axis("off")
    if i == 0:
        ax.set_title("Original")
    # reconstructions
    ax = plt.subplot(2, n, n + i + 1)
    ax.imshow(reco[i], cmap="gray")
    ax.axis("off")
    if i == 0:
        ax.set_title("Reconstruction")
plt.tight_layout()
plt.show()
