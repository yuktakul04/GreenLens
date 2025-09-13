# -*- coding: utf-8 -*-
import os, time, json, csv, random, argparse
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CO2Dataset(Dataset):
    """
    CSV format (relative paths allowed):
        image_path,co2
        dataset_products/train/ProdA/img_001.jpg,12.7
        dataset_products/train/ProdB/img_123.png,5.3
    """
    def __init__(self, csv_path, transform=None, log_target=False):
        self.transform = transform
        self.log_target = log_target
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                self.rows.append((r["image_path"], float(r["co2"])))

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        p, val = self.rows[idx]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        y = np.log1p(val) if self.log_target else val
        return img, torch.tensor([y], dtype=torch.float32)

def build_model(dropout=0.1):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Identity()
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
    return nn.Sequential(model, head)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def epoch_loop(model, loader, criterion, device, train, optimizer=None):
    model.train(train)
    total_loss, n = 0.0, 0
    ys, ps = [], []
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        preds = model(imgs)
        loss = criterion(preds, targets)
        if train:
            loss.backward()
            optimizer.step()
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs
        ys.append(targets.detach().cpu().numpy().ravel())
        ps.append(preds.detach().cpu().numpy().ravel())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    log = {"loss": total_loss / max(n, 1)}
    if y_true.size > 0:
        log["MAE"] = float(mean_absolute_error(y_true, y_pred))
        log["RMSE"] = rmse(y_true, y_pred)
        log["R2"] = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    else:
        log.update({"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")})
    return log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, help="CSV with image_path,co2")
    parser.add_argument("--val_csv",   required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--use_huber", action="store_true")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--log_target", action="store_true", help="Train on log1p(CO2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_path", default="co2_regressor_model.pth")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # transforms = same style as your classifier
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = CO2Dataset(args.train_csv, transform=train_tfms, log_target=args.log_target)
    val_ds   = CO2Dataset(args.val_csv,   transform=val_tfms,   log_target=args.log_target)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(device)
    criterion = (nn.HuberLoss(delta=args.huber_delta) if args.use_huber else nn.MSELoss())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    print(f"\nStarting CO2 regression training for {args.epochs} epochs.")
    best_rmse, best_wts = float("inf"), None
    history = {"train": [], "val": []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}\n" + "-"*40)
        tr = epoch_loop(model, train_loader, criterion, device, True, optimizer)
        va = epoch_loop(model, val_loader,   criterion, device, False)

        history["train"].append(tr); history["val"].append(va)
        print(f"Train  Loss:{tr['loss']:.4f}  MAE:{tr['MAE']:.4f}  RMSE:{tr['RMSE']:.4f}  R2:{tr['R2']:.4f}")
        print(f"Val    Loss:{va['loss']:.4f}  MAE:{va['MAE']:.4f}  RMSE:{va['RMSE']:.4f}  R2:{va['R2']:.4f}")

        if va["RMSE"] < best_rmse:
            best_rmse, best_wts = va["RMSE"], {k: v.cpu() for k, v in model.state_dict().items()}
            print("New best regressor (lower RMSE).")

        scheduler.step()
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    if best_wts is not None:
        model.load_state_dict(best_wts)

    torch.save({
        "model_state_dict": model.state_dict(),
        "normalize_mean": IMAGENET_MEAN,
        "normalize_std": IMAGENET_STD,
        "best_rmse": best_rmse,
        "history": history,
        "log_target": args.log_target,
        "criterion": "Huber" if args.use_huber else "MSE",
        "backbone": "mobilenet_v3_small",
    }, args.out_path)

    print("\n✅ Saved best model to:", args.out_path)
    print(f"Best validation RMSE: {best_rmse:.4f}")

if __name__ == "__main__":
    main()
