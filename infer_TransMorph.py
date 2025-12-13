from torch.utils.tensorboard import SummaryWriter
import os
import utils
import glob
import losses
import sys
#
from torch.utils.data import DataLoader
from data.npy_dataset import NPYBrainDataset
from data.npy_pair_dataset import NPYPairDataset

import numpy as np
import torch
from torchvision import transforms
from torch import optim
from natsort import natsorted

from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from pytorch_msssim import SSIM
import torch.nn.functional as F


class ResizeToFixed:
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs):
        resized = []
        for img in imgs:
            img = np.ascontiguousarray(img)
            img_tensor = torch.from_numpy(img).float().unsqueeze(0)
            img_resized = F.interpolate(
                img_tensor,
                size=self.size,
                mode="bilinear",
                align_corners=False
            )
            resized.append(img_resized.squeeze(0).numpy())
        return resized


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def save_checkpoint(state, save_dir, filename, max_model_num=4):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))
    model_lists = natsorted(glob.glob(os.path.join(save_dir, "*")))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, "*")))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    MODE = "mnist"  # "brain" or "mnist"
    weights = [1, 1]
    lr = 1e-4
    cont_training = False

    if MODE == "brain":
        batch_size = 32
        max_epoch = 400
        img_size = (256, 256)
        train_dir = "/scratch/vfv5up/MLIA/FinalProject/TransMorph_Transformer_for_Medical_Image_Registration/RaFD/TransMorph2D/brain_train_image_final.npy"
        val_dir = "/scratch/vfv5up/MLIA/FinalProject/TransMorph_Transformer_for_Medical_Image_Registration/RaFD/TransMorph2D/brain_test_image_final.npy"
    else:
        batch_size = 64
        max_epoch = 30
        img_size = (32, 32)
        train_dir = "data/partB/mnist_train.npy"
        val_dir = "data/partB/mnist_test.npy"

    save_dir = f"TransMorph_{MODE}_ssim_{weights[0]}_diffusion_{weights[1]}/"
    exp_dir = os.path.join("experiments", save_dir)
    log_dir = os.path.join("logs", save_dir)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    sys.stdout = Logger(log_dir)
    epoch_start = 0

    config = CONFIGS_TM["TransMorph"]
    config.img_size = img_size
    config.in_chans = 2

    model = TransMorph.TransMorph(config).to(device)

    if cont_training:
        ckpts = natsorted(os.listdir(exp_dir))
        ckpt_path = os.path.join(exp_dir, ckpts[-1])
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print("Resumed from:", ckpt_path)

    train_tf = transforms.Compose([ResizeToFixed(img_size)])
    val_tf = transforms.Compose([ResizeToFixed(img_size)])

    if MODE == "brain":
        train_set = NPYBrainDataset(train_dir, transforms=train_tf)
        val_set = NPYBrainDataset(val_dir, transforms=val_tf)
    else:
        train_set = NPYPairDataset(train_dir, transforms=train_tf)
        val_set = NPYPairDataset(val_dir, transforms=val_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    sim_loss = losses.SSIM_loss(False)
    reg_loss = losses.Grad("l2")
    criterions = [sim_loss, reg_loss]

    ssim_metric = SSIM(data_range=1, size_average=True, channel=1).to(device)

    writer = SummaryWriter(log_dir=log_dir)
    best_score = -1e9

    for epoch in range(epoch_start, max_epoch):
        model.train()
        loss_meter = utils.AverageMeter()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)

            loss = criterions[0](output[0], y) + criterions[1](output[1], y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), y.numel())

        writer.add_scalar("Loss/train", loss_meter.avg, epoch)
        print(f"Epoch {epoch}: Train Loss {loss_meter.avg:.4f}")

        model.eval()
        eval_meter = utils.AverageMeter()

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                x_in = torch.cat((y, x), dim=1)
                output = model(x_in)
                score = ssim_metric(output[0], x)
                eval_meter.update(score.item(), x.numel())

        writer.add_scalar("SSIM/val", eval_meter.avg, epoch)
        print(f"Epoch {epoch}: Val SSIM {eval_meter.avg:.4f}")

        if eval_meter.avg > best_score:
            best_score = eval_meter.avg
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                },
                save_dir=exp_dir,
                filename=f"best_{best_score:.4f}.pth.tar"
            )

    writer.close()


if __name__ == "__main__":
    main()