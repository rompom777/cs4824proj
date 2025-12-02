# src/train_label_from_concepts.py

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets import make_dataloaders
from src.models import LabelFromConcepts


def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        concepts = batch["concepts"].to(device)      # [B, K]
        labels   = batch["label"].to(device)         # [B]

        optimizer.zero_grad()
        logits = model(concepts)                     # [B, num_classes]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(logits, labels) * batch_size
        total += batch_size

    return running_loss / total, running_acc / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Val", leave=False):
            concepts = batch["concepts"].to(device)
            labels   = batch["label"].to(device)

            logits = model(concepts)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy_from_logits(logits, labels) * batch_size
            total += batch_size

    return running_loss / total, running_acc / total


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

    model = LabelFromConcepts(num_concepts=num_concepts, num_classes=num_classes, hidden_dim=args.hidden_dim).to(device)

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.out_dir, "label_from_concepts_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Saved new best label-from-concepts model to {ckpt_path}")

    ckpt = torch.load(os.path.join(args.out_dir, "label_from_concepts_best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    print(f"\nFinal TEST (label from TRUE concepts): loss={test_loss:.4f}, acc={test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="checkpoints_cub_c2y")

    args = parser.parse_args()
    main(args)