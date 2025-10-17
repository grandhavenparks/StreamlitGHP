import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import timm  # pip install timm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import multiprocessing

# ── CONFIG ──────────────────────────────────────────────────────────────────────
DATA_DIR     = r"D:\DNR\DATA\Data5_2025"
BATCH_SIZE   = 32
IMG_SIZE     = 256  
NUM_CLASSES  = 2
LR           = 3e-4
WEIGHT_DECAY = 1e-2
EPOCHS       = 55
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose your Swin V2 variant:
MODEL_NAME = "swinv2_tiny_window8_256"   # smaller, faster
# MODEL_NAME = "swinv2_small_window8_256"  # higher capacity

SAVE_PATH = "swinv2_tiny_oakwilt25.pth"

# ── TRANSFORMS ──────────────────────────────────────────────────────────────────
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
val_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE+32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

def main():
    # ── LOAD & SPLIT DATA ────────────────────────────────────────────────────────
    full_ds = datasets.ImageFolder(DATA_DIR)
    class_names = full_ds.classes  # ['now', 'ow']

    # Manually map 'now' to 0 and 'ow' to 1
    class_map = {'now': 0, 'ow': 1}
    
    # Update samples with the correct label using the class_map
    updated_samples = []
    for sample in full_ds.samples:
        class_name = sample[1]  # This is the class index, 0 for 'now' and 1 for 'ow'
        class_name = class_names[class_name]  # Get the actual class name ('now' or 'ow')
        updated_samples.append((sample[0], class_map[class_name]))

    full_ds.samples = updated_samples

    n = len(full_ds)
    idx = list(range(n))
    random.seed(42)
    random.shuffle(idx)

    # 70% train, 20% test, 10% validation
    split_train = int(0.7 * n)
    split_val = int(0.9 * n)

    train_idx, val_idx, test_idx = idx[:split_train], idx[split_train:split_val], idx[split_val:]

    train_ds = Subset(
        datasets.ImageFolder(DATA_DIR, transform=train_tfms),
        train_idx
    )
    val_ds = Subset(
        datasets.ImageFolder(DATA_DIR, transform=val_tfms),
        val_idx
    )
    test_ds = Subset(
        datasets.ImageFolder(DATA_DIR, transform=val_tfms),
        test_idx
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ── MODEL ─────────────────────────────────────────────────────────────────────
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ── TRAIN & VALIDATE ─────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, EPOCHS+1):
        # — Training — 
        model.train()
        run_loss = run_corr = run_tot = 0
        for imgs, labs in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            run_corr += (preds == labs).sum().item()
            run_tot  += labs.size(0)

        tloss = run_loss/run_tot
        tacc  = run_corr/run_tot
        train_losses.append(tloss)
        train_accs.append(tacc)

        # — Validation — 
        model.eval()
        v_loss = v_corr = v_tot = 0
        with torch.no_grad():
            for imgs, labs in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labs)
                v_loss += loss.item() * imgs.size(0)
                preds = out.argmax(dim=1)
                v_corr += (preds == labs).sum().item()
                v_tot  += labs.size(0)

        vloss = v_loss/v_tot
        vacc  = v_corr/v_tot
        val_losses.append(vloss)
        val_accs.append(vacc)

        print(
            f"Epoch {epoch}/{EPOCHS} → "
            f"Train Loss: {tloss:.4f}, Train Acc: {tacc:.4f} │ "
            f"Val Loss: {vloss:.4f}, Val Acc: {vacc:.4f}"
        )

    # ── SAVE MODEL WEIGHTS ─────────────────────────────────────────────────────────
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nModel weights saved to {SAVE_PATH}")

    # ── PLOT LOSS & ACCURACY ───────────────────────────────────────────────────────
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1,EPOCHS+1), train_losses, label="Train Loss")
    plt.plot(range(1,EPOCHS+1), val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epochs"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1,EPOCHS+1), train_accs,  label="Train Acc")
    plt.plot(range(1,EPOCHS+1), val_accs,    label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epochs"); plt.legend()

    plt.tight_layout()
    plt.show()

    # ── CONFUSION MATRIX & FINAL METRICS ──────────────────────────────────────────
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs = imgs.to(DEVICE)
            out  = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labs.numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.colorbar(); plt.tight_layout(); plt.show()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(
        y_true, y_pred, average="binary",
        pos_label=class_names.index("ow")
    )
    rec  = recall_score(
        y_true, y_pred, average="binary",
        pos_label=class_names.index("ow")
    )
    f1   = f1_score(
        y_true, y_pred, average="binary",
        pos_label=class_names.index("ow")
    )

    print(f"\nFinal Accuracy : {acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1-Score       : {f1:.4f}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
