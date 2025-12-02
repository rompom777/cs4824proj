# src/eval_cbm.py

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from src.datasets import make_dataloaders
from src.models import ConceptPredictor, LabelFromConcepts, CBMSequential


def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def eval_true_concepts(label_from_concepts, loader, device):
    label_from_concepts.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Eval (true concepts)", leave=False):
            concepts = batch["concepts"].to(device)
            labels   = batch["label"].to(device)

            logits = label_from_concepts(concepts)
            loss   = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss    += loss.item() * batch_size
            total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total         += batch_size

    return total_loss / total, total_correct / total


def eval_predicted_concepts(cbm, loader, device):
    cbm.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Eval (predicted concepts)", leave=False):
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits_y, c_hat, c_probs = cbm(imgs)
            loss = criterion(logits_y, labels)

            batch_size = labels.size(0)
            total_loss    += loss.item() * batch_size
            total_correct += (torch.argmax(logits_y, dim=1) == labels).sum().item()
            total         += batch_size

    return total_loss / total, total_correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # get loaders and dimensions
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

    # ----- Load concept predictor -----
    concept_model = ConceptPredictor(num_concepts=num_concepts, pretrained=False).to(device)
    cp_ckpt = torch.load(args.concept_ckpt, map_location=device)
    concept_model.load_state_dict(cp_ckpt["model_state_dict"])
    print(f"Loaded concept predictor from {args.concept_ckpt} (val_acc={cp_ckpt.get('val_acc', 'N/A')})")

    # ----- Load label-from-concepts model -----
    label_from_concepts = LabelFromConcepts(
        num_concepts=num_concepts,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
    ).to(device)
    c2y_ckpt = torch.load(args.c2y_ckpt, map_location=device)
    label_from_concepts.load_state_dict(c2y_ckpt["model_state_dict"])
    print(f"Loaded label-from-concepts from {args.c2y_ckpt} (val_acc={c2y_ckpt.get('val_acc', 'N/A')})")

    # ----- Evaluate with TRUE concepts (upper bound) -----
    true_loss, true_acc = eval_true_concepts(label_from_concepts, test_loader, device)
    print(f"\nTEST (TRUE concepts → y): loss={true_loss:.4f}, acc={true_acc:.4f}")

    # ----- Evaluate with PREDICTED concepts (full CBM) -----
    cbm = CBMSequential(concept_model, label_from_concepts, threshold=args.threshold).to(device)
    cbm_loss, cbm_acc = eval_predicted_concepts(cbm, test_loader, device)
    print(f"TEST (x → ĉ → y): loss={cbm_loss:.4f}, acc={cbm_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)

    parser.add_argument("--concept_ckpt", type=str, required=True,
                        help="Path to concept_predictor_best.pt")
    parser.add_argument("--c2y_ckpt", type=str, required=True,
                        help="Path to label_from_concepts_best.pt")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    main(args)
