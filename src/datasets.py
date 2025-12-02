# src/datasets.py

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class ConceptBottleneckImageDataset(Dataset):
    """
    Each row in the CSV should have:
        image_path: relative or absolute path to the image
        y: integer class label
        c_0, c_1, ..., c_{K-1}: concept values (0/1, or floats)

    Example CSV header:
        image_path,y,c_0,c_1,c_2,...
    """
    def __init__(self, csv_path, img_root, transform=None, concept_cols=None):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

        # If concept_cols not provided, infer from columns starting with "c_"
        if concept_cols is None:
            self.concept_cols = [col for col in self.df.columns if col.startswith("c_")]
        else:
            self.concept_cols = concept_cols

        # Sanity
        assert "image_path" in self.df.columns, "CSV must have 'image_path' column"
        assert "y" in self.df.columns, "CSV must have 'y' column"
        assert len(self.concept_cols) > 0, "No concept columns found!"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        img_full = img_path if os.path.isabs(img_path) else os.path.join(self.img_root, img_path)

        image = Image.open(img_full).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # label
        y = int(row["y"])

        # concepts vector
        concepts = torch.tensor(row[self.concept_cols].values.astype("float32"))

        sample = {
            "image": image,          # [C, H, W]
            "concepts": concepts,    # [K]
            "label": torch.tensor(y, dtype=torch.long)
        }
        return sample


def get_default_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def make_dataloaders(train_csv, val_csv, test_csv, img_root,
                     batch_size=32, num_workers=4, img_size=224):

    transform = get_default_transforms(img_size)

    train_ds = ConceptBottleneckImageDataset(train_csv, img_root, transform=transform)
    val_ds   = ConceptBottleneckImageDataset(val_csv, img_root, transform=transform)
    test_ds  = ConceptBottleneckImageDataset(test_csv, img_root, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_concepts = len(train_ds.concept_cols)
    num_classes  = len(train_ds.df["y"].unique())

    return train_loader, val_loader, test_loader, num_concepts, num_classes
