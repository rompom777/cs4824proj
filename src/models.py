# src/models.py

import torch
import torch.nn as nn
import torchvision.models as models


class LabelOnlyModel(nn.Module):
    """
    Baseline: x -> y
    """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


class ConceptPredictor(nn.Module):
    """
    x -> c (concept vector)
    """
    def __init__(self, num_concepts, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.concept_head = nn.Linear(in_features, num_concepts)

    def forward(self, x):
        feats = self.backbone(x)
        concept_logits = self.concept_head(feats)  # raw logits
        return concept_logits


class LabelFromConcepts(nn.Module):
    """
    c -> y
    """
    def __init__(self, num_concepts, num_classes, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, c):
        logits = self.net(c)
        return logits


class CBMSequential(nn.Module):
    """
    Full sequential CBM: x -> c_hat -> y_hat
    """
    def __init__(self, concept_predictor: ConceptPredictor, label_from_concepts: LabelFromConcepts, threshold=0.5):
        super().__init__()
        self.concept_predictor = concept_predictor
        self.label_from_concepts = label_from_concepts
        self.threshold = threshold

    def forward(self, x):
        c_logits = self.concept_predictor(x)          # [B, K]
        c_probs  = torch.sigmoid(c_logits)            # [B, K]
        c_hat    = (c_probs > self.threshold).float() # binarize
        y_logits = self.label_from_concepts(c_hat)    # [B, num_classes]
        return y_logits, c_hat, c_probs
