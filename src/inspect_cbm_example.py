# src/inspect_cbm_example.py

import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets import make_dataloaders
from src.models import ConceptPredictor, LabelFromConcepts, CBMSequential
from src.utils_cub import load_cub_attribute_names


@torch.no_grad()
def explain_one_example(cbm, label_from_concepts, batch, attr_names, device, top_k=10):
    """
    Given one batch (size 1) from the DataLoader, print:
      - true label
      - predicted label
      - top-k concepts that contributed most to the prediction
    """
    cbm.eval()
    label_from_concepts.eval()

    img     = batch["image"].to(device)          # [1, 3, H, W]
    true_y  = batch["label"].item()
    true_c  = batch["concepts"][0].numpy()       # [K]

    # Forward through CBM: image -> predicted concepts -> predicted label
    logits_y, c_hat, c_probs = cbm(img)          # logits_y [1,C], c_hat [1,K], c_probs [1,K]
    probs_y = torch.softmax(logits_y, dim=1)[0]  # [C]
    pred_y  = int(torch.argmax(probs_y).item())

    print(f"True label index:      {true_y}")
    print(f"Predicted label index: {pred_y}")
    print(f"Predicted label prob:  {probs_y[pred_y]:.3f}")
    print()

    # Get final linear layer weights for label_from_concepts
    # Assuming LabelFromConcepts is something like nn.Sequential(...)
    last_linear = None
    for m in label_from_concepts.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    assert last_linear is not None, "Could not find final linear layer in LabelFromConcepts"

    # weight for the predicted class: [K]
    w = last_linear.weight[pred_y].detach().cpu()      # [K]
    c_probs_vec = c_probs[0].detach().cpu()           # [K]

    # Contribution score: weight * prob (simple heuristic)
    contributions = w * c_probs_vec                   # [K]

    # Sort concepts by contribution
    K = contributions.shape[0]
    indices = torch.argsort(contributions, descending=True)[:top_k]

    print(f"Top {top_k} contributing concepts for predicted class:\n")
    for rank, idx in enumerate(indices):
        idx = int(idx.item())
        name = attr_names[idx] if idx < len(attr_names) else f"c_{idx}"
        prob = float(c_probs_vec[idx].item())
        contrib = float(contributions[idx].item())
        true_val = int(true_c[idx])
        print(f"{rank+1:2d}. {name:40s} | p(c=1)={prob:.3f} | contrib={contrib:.3f} | true={true_val}")
    print()

    # Also show some high-probability concepts (regardless of weight)
    print(f"Concepts with high predicted probability (p>0.8):\n")
    high_mask = (c_probs_vec > 0.8)
    high_indices = torch.nonzero(high_mask, as_tuple=False).flatten().tolist()
    for idx in high_indices[:top_k]:
        name = attr_names[idx] if idx < len(attr_names) else f"c_{idx}"
        prob = float(c_probs_vec[idx].item())
        true_val = int(true_c[idx])
        print(f"- {name:40s} | p(c=1)={prob:.3f} | true={true_val}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load attribute names for explanations
    attr_names = load_cub_attribute_names(args.cub_root)
    print(f"Loaded {len(attr_names)} attribute names.")

    # Build dataloaders with batch_size=1 so we can inspect per example
    _, _, test_loader, num_concepts, num_classes = make_dataloaders(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        args.img_root,
        batch_size=1,
        num_workers=0,
        img_size=args.img_size,
    )
    print(f"num_concepts={num_concepts}, num_classes={num_classes}")

    # ----- Load models -----
    # Concept predictor
    concept_model = ConceptPredictor(num_concepts=num_concepts, pretrained=False).to(device)
    cp_ckpt = torch.load(args.concept_ckpt, map_location=device)
    concept_model.load_state_dict(cp_ckpt["model_state_dict"])
    print(f"Loaded concept predictor from {args.concept_ckpt} (val_acc={cp_ckpt.get('val_acc', 'N/A')})")

    # Label-from-concepts
    label_from_concepts = LabelFromConcepts(
        num_concepts=num_concepts,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
    ).to(device)
    c2y_ckpt = torch.load(args.c2y_ckpt, map_location=device)
    label_from_concepts.load_state_dict(c2y_ckpt["model_state_dict"])
    print(f"Loaded label-from-concepts from {args.c2y_ckpt} (val_acc={c2y_ckpt.get('val_acc', 'N/A')})")

    # Build CBM
    cbm = CBMSequential(concept_model, label_from_concepts, threshold=args.threshold).to(device)

    # ----- Pick one example from the test set -----
    test_iter = iter(test_loader)
    batch = next(test_iter)   # first test example
    explain_one_example(cbm, label_from_concepts, batch, attr_names, device, top_k=args.top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_root", type=str, default="data/CUB_200_2011")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)

    parser.add_argument("--concept_ckpt", type=str, required=True)
    parser.add_argument("--c2y_ckpt", type=str, required=True)

    parser.add_argument("--hidden_dim", type=int, default=256)  # match your trained C2Y model
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()
    main(args)
