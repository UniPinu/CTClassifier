"""
Step 3 check - Grad-CAM: where in the patch does the model look?

Grad-CAM takes the last conv block's feature maps and weights each one by how much it
pushed the "cancer" score up. The result is a heat map over the patch.

Reads : models/best.pt, test patches
Writes: checks/gradcam.png, and prints a "centre score"
Run   : python step3_gradcam.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from config import CHECK_DIR, MODEL_DIR, PATCH
from step3_model import PatchCNN, PatchDataset


def grad_cam(model, x):
    """Heat map (same size as x) of where the model looked to call x 'cancer'."""
    store = {}

    def hook(module, inputs, output):          # runs during the forward pass
        store["maps"] = output                 # (n, 128, 8, 8, 8) feature maps of block 4
        output.register_hook(lambda grad: store.update(grads=grad))

    handle = model.encoder[3][2].register_forward_hook(hook)   # ReLU inside block 4
    model.zero_grad()
    model(x).sum().backward()
    handle.remove()

    weights = store["grads"].mean(dim=(2, 3, 4), keepdim=True)          # importance per map
    cam = F.relu((weights * store["maps"]).sum(dim=1, keepdim=True))    # (n, 1, 8, 8, 8)
    cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear")        # back to 64^3
    cam = cam / (cam.amax(dim=(2, 3, 4), keepdim=True) + 1e-8)          # scale to 0..1
    return cam[:, 0].detach()


def main():
    model = PatchCNN()
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location="cpu"))
    model.eval()

    ds = PatchDataset("test", augment=False)
    cancers = ds.df.index[ds.df["kind"] == "cancer"][:64]
    x = torch.stack([ds[i][0] for i in cancers])
    cam = grad_cam(model, x)
    prob = torch.sigmoid(model(x)).detach().ravel()

    # Centre score: share of the heat inside the central 24^3 mm, where the lesion is.
    # If the heat were spread evenly this would be (24/64)^3 = 0.05.
    lo, hi = PATCH // 2 - 12, PATCH // 2 + 12
    centre = (cam[:, lo:hi, lo:hi, lo:hi].sum(dim=(1, 2, 3)) / cam.sum(dim=(1, 2, 3))).mean()
    print(f"Centre score on {len(x)} test cancers: {centre:.2f} (0.05 = no focus on the centre)")

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    mid = PATCH // 2
    for ax, i in zip(axes.ravel(), range(len(x))):
        ax.imshow(x[i, 0, mid], cmap="gray", vmin=0, vmax=1)
        ax.imshow(cam[i, mid], cmap="jet", alpha=0.35, vmin=0, vmax=1)
        ax.set_title(f"p(cancer) = {prob[i]:.2f}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    CHECK_DIR.mkdir(exist_ok=True)
    fig.savefig(CHECK_DIR / "gradcam.png", dpi=80)
    print(f"Saved {CHECK_DIR / 'gradcam.png'}")


if __name__ == "__main__":
    main()
