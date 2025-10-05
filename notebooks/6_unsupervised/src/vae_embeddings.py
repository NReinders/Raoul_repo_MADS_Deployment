# save as notebooks/6_unsupervised/src/write_tb_embeddings.py
import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from settings import VAESettings
from mltrainer import vae

# device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

presets = VAESettings(latent=2)  # or 10 — must match your trained model
modelpath = presets.modeldir / presets.modelname

# robust load (PyTorch 2.6 safe)
def load_model(path):
    try:
        m = torch.load(path, map_location=device, weights_only=False)
    except Exception:
        m = vae.AutoEncoder(presets.model_dump()).to(device)
        state = torch.load(path, map_location=device)
        m.load_state_dict(state)
    m.eval()
    return m

model = load_model(modelpath)

# data — take a decent chunk so projector is interesting
test = datasets.MNIST(root=presets.data_dir, train=False, download=True,
                      transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(test, batch_size=1024, shuffle=False)

# collect one big batch (adjust if you want all)
imgs, ys = next(iter(loader))       # imgs: [N,1,28,28] (NCHW)
# your encoder expects NHWC:
imgs_nhwc = imgs.permute(0,2,3,1).contiguous().to(device)

with torch.no_grad():
    z = model.encoder(imgs_nhwc)   # [N,latent]

# TensorBoard wants CHW images; move back to CHW and to CPU
label_img = imgs                   # [N,1,28,28] on CPU
embs = z.detach().cpu()            # [N,latent]
labels = ys.cpu().tolist()         # metadata as Python list

writer = SummaryWriter(log_dir="models/embeddings")
# add a tiny scalar so TB immediately shows a dashboard
writer.add_scalar("heartbeat/ok", 1, 0)
# write embeddings to Projector (this creates projector files under models/embeddings)
writer.add_embedding(embs, metadata=[str(l) for l in labels], label_img=label_img, global_step=0)
writer.flush()
writer.close()

# also keep a .pt for your KD-tree demo
torch.save((imgs, embs.numpy()), "models/embeds.pt")
print("Wrote embeddings to TensorBoard at models/embeddings and to models/embeds.pt")