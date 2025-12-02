# src/train_baseline.py

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import make_dataloaders
from models import LabelOnlyModel


def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(logits, labels) * batch_size
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_acc / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Val", leave=False):
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy_from_logits(logits, labels) * batch_size
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_acc / total
    return epoch_loss, epoch_acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, num_concepts, num_classes = make_dataloaders(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        args.img_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    print(f"num_concepts={num_concepts}, num_classes={num_classes}")

    model = LabelOnlyModel(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.out_dir, "label_only_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_path)
            print(f"Saved new best model to {ckpt_path}")

    # final test eval with best weights
    ckpt = torch.load(os.path.join(args.out_dir, "label_only_best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    print(f"\nFinal TEST: loss={test_loss:.4f}, acc={test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    main(args)
