# src/prepare_cub.py

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_cub_metadata(cub_root):
    """
    cub_root: path to CUB_200_2011 (folder containing images/, attributes/, images.txt, etc.)
    """
    images_txt = os.path.join(cub_root, "images.txt")
    labels_txt = os.path.join(cub_root, "image_class_labels.txt")
    split_txt  = os.path.join(cub_root, "train_test_split.txt")
    attr_txt   = os.path.join(cub_root, "attributes", "image_attribute_labels.txt")

    # images.txt: image_id image_path
    df_images = pd.read_csv(
        images_txt, sep=" ", header=None, names=["image_id", "image_path"]
    )

    # image_class_labels.txt: image_id class_id
    df_labels = pd.read_csv(
        labels_txt, sep=" ", header=None, names=["image_id", "class_id"]
    )

    # train_test_split.txt: image_id is_training_image
    df_split = pd.read_csv(
        split_txt, sep=" ", header=None, names=["image_id", "is_train"]
    )

    # ---------- MANUAL PARSE FOR ATTRIBUTES ----------
    # attributes/image_attribute_labels.txt:
    # image_id attribute_id is_present certainty time
    # some lines have extra fields; we only care about the first 5
    rows = []
    with open(attr_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            # skip weird empty lines, just in case
            if len(parts) < 5:
                continue
            # take only the first 5 tokens even if more are present
            image_id   = int(parts[0])
            attr_id    = int(parts[1])
            is_present = int(parts[2])
            certainty  = int(parts[3])
            time_val   = float(parts[4])
            rows.append((image_id, attr_id, is_present, certainty, time_val))

    df_attr = pd.DataFrame(
        rows,
        columns=["image_id", "attr_id", "is_present", "certainty", "time"]
    )
    # ---------- END MANUAL PARSE ----------

    return df_images, df_labels, df_split, df_attr


def build_concept_matrix(df_attr, num_images, num_attributes=312):
    """
    Build a [num_images, num_attributes] matrix of 0/1 concept values.

    image_id in CUB goes from 1..N; attr_id goes 1..312.
    We'll convert to zero-based indices: image_id-1, attr_id-1.
    """
    num_images = int(num_images)
    num_attributes = int(num_attributes)

    # Initialize all zeros
    C = np.zeros((num_images, num_attributes), dtype=np.float32)

    for _, row in df_attr.iterrows():
        # Force everything to int / float explicitly
        try:
            img_id = int(row["image_id"])
            attr_id = int(row["attr_id"])
            is_present = int(row["is_present"])
        except Exception:
            # skip any weird row
            continue

        img_idx = img_id - 1
        attr_idx = attr_id - 1

        # Guard against out-of-range indices just in case
        if (
            0 <= img_idx < num_images
            and 0 <= attr_idx < num_attributes
            and is_present == 1
        ):
            C[img_idx, attr_idx] = 1.0

    return C


def make_cub_csvs(cub_root, out_dir, val_frac=0.1, random_state=42):
    os.makedirs(out_dir, exist_ok=True)

    df_images, df_labels, df_split, df_attr = load_cub_metadata(cub_root)

    num_images = df_images["image_id"].max()
    num_attributes = df_attr["attr_id"].max()
    print(f"Found {num_images} images, {num_attributes} attributes.")

    C = build_concept_matrix(df_attr, num_images=num_images, num_attributes=num_attributes)

    # Merge images, labels, split into one DataFrame
    df = df_images.merge(df_labels, on="image_id").merge(df_split, on="image_id")

    # Convert class_id from 1..200 to 0..199 for nicer indexing
    df["y"] = df["class_id"] - 1

    # Attach concept vectors to df
    concept_cols = [f"c_{i}" for i in range(num_attributes)]

    # C is indexed by image_id-1, so align by image_id
    # We'll build a DataFrame for concepts with image_id, then merge
    concept_df = pd.DataFrame(C, columns=concept_cols)
    concept_df["image_id"] = np.arange(1, num_images + 1)

    df = df.merge(concept_df, on="image_id")

    # Now df has:
    # image_id, image_path, class_id, is_train, y, c_0...c_311

    # Train / test split according to is_train
    df_train_full = df[df["is_train"] == 1].reset_index(drop=True)
    df_test = df[df["is_train"] == 0].reset_index(drop=True)

    # Further split train into train/val
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=val_frac,
        random_state=random_state,
        stratify=df_train_full["y"],
    )

    # For CSVs, only keep what our Dataset needs: image_path, y, concepts
    keep_cols = ["image_path", "y"] + concept_cols

    train_csv = os.path.join(out_dir, "train.csv")
    val_csv   = os.path.join(out_dir, "val.csv")
    test_csv  = os.path.join(out_dir, "test.csv")

    df_train[keep_cols].to_csv(train_csv, index=False)
    df_val[keep_cols].to_csv(val_csv, index=False)
    df_test[keep_cols].to_csv(test_csv, index=False)

    print(f"Saved:\n  {train_csv}\n  {val_csv}\n  {test_csv}")
    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_root", type=str, required=True,
                        help="Path to CUB_200_2011 directory")
    parser.add_argument("--out_dir", type=str, default="data/cub_csvs")
    parser.add_argument("--val_frac", type=float, default=0.1)
    args = parser.parse_args()

    make_cub_csvs(args.cub_root, args.out_dir, val_frac=args.val_frac)


if __name__ == "__main__":
    import argparse
    main()
