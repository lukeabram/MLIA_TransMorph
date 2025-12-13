import os, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM

from data.npy_dataset import NPYBrainDataset
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph

class ResizeToFixed:
    def __init__(self, size):
        self.size = size
    def __call__(self, imgs):
        out = []
        for img in imgs:
            t = torch.from_numpy(img).float().unsqueeze(0)
            t = F.interpolate(t, self.size, mode="bilinear", align_corners=False)
            out.append(t.squeeze(0).numpy())
        return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = (32, 32)
    test_dir = "data/partB/quickdraw_test.npy"
    model_dir = "experiments/TransMorph_partB_mnist/"

    ckpt = sorted(os.listdir(model_dir))[-1]
    ckpt_path = os.path.join(model_dir, ckpt)

    config = CONFIGS_TM["TransMorph"]
    config.img_size = img_size
    config.in_chans = 2

    model = TransMorph.TransMorph(config).to(device)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model.eval()

    test_tf = ResizeToFixed(img_size)
    test_set = NPYBrainDataset(test_dir, transforms=test_tf)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    ssim = SSIM(data_range=1, size_average=True, channel=1).to(device)
    scores = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            warped, _ = model(torch.cat((y, x), dim=1))
            scores.append(ssim(warped, x).item())

    print("QuickDraw SSIM:", np.mean(scores))

if __name__ == "__main__":
    main()
