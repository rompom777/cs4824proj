# src/train_concept_predictor.py

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets import make_dataloaders
from src.models import ConceptPredictor


def concept_accuracy_from_logits(logits, targets):
    """
    logits: [B, K]
    targets: [B, K] (float 0/1)
    Returns mean concept-wise accuracy over batch.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    correct = (preds == targets).float().mean().item()
    return correct


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        imgs = batch["image"].to(device)
        concepts = batch["concepts"].to(device)  # [B, K] floats

        optimizer.zero_grad()
        logits = model(imgs)  # [B, K]
        loss = criterion(logits, concepts)
        loss.backward()
        optimizer.step()

        batch_size = concepts.size(0)
        running_loss += loss.item() * batch_size
        running_acc += concept_accuracy_from_logits(logits, concepts) * batch_size
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
            concepts = batch["concepts"].to(device)

            logits = model(imgs)
            loss = criterion(logits, concepts)

            batch_size = concepts.size(0)
            running_loss += loss.item() * batch_size
            running_acc += concept_accuracy_from_logits(logits, concepts) * batch_size
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

    model = ConceptPredictor(num_concepts=num_concepts, pretrained=True).to(device)

    # BCEWithLogitsLoss expects float targets in [0,1]
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f}, concept-acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, concept-acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.out_dir, "concept_predictor_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Saved new best concept predictor to {ckpt_path}")

    # final test eval
    ckpt = torch.load(os.path.join(args.out_dir, "concept_predictor_best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    print(f"\nFinal TEST (concept predictor): loss={test_loss:.4f}, concept-acc={test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="checkpoints_cub_concepts")

    args = parser.parse_args()
    main(args)