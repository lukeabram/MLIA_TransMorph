import os, torch, numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
import torch.nn.functional as F

from data.npy_pair_dataset import NPYPairDataset
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import losses, utils

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
    batch_size = 16
    max_epoch = 3   # 🔥 intentionally small
    lr = 1e-4

    train_dir = "data/partB/mnist_train.npy"
    save_dir = "experiments/TransMorph_partB_mnist/"

    os.makedirs(save_dir, exist_ok=True)

    config = CONFIGS_TM["TransMorph"]
    config.img_size = img_size
    config.in_chans = 2

    model = TransMorph.TransMorph(config).to(device)

    train_tf = transforms.Compose([ResizeToFixed(img_size)])
    train_set = NPYPairDataset(train_dir, transforms=train_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    sim_loss = losses.SSIM_loss(False)
    reg_loss = losses.Grad("l2")

    for epoch in range(max_epoch):
        model.train()
        loss_meter = utils.AverageMeter()

        for i, (x, y) in enumerate(train_loader):
            if i % 10 == 0:
                print(f"Epoch {epoch} | Iter {i}/{len(train_loader)}")

            x, y = x.to(device), y.to(device)
            out = model(torch.cat((x, y), dim=1))

            loss = sim_loss(out[0], y) + reg_loss(out[1], y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), y.numel())

        print(f"Epoch {epoch} loss: {loss_meter.avg:.4f}")

        torch.save(
            {"state_dict": model.state_dict()},
            os.path.join(save_dir, f"epoch_{epoch}.pth.tar")
        )

if __name__ == "__main__":
    main()
