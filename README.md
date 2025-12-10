# Interpretable Bird Classification Using Concept Bottleneck Models  
### Final Project — Machine Learning  
**Authors:** Roman Zrajevsky, Benjamin Clements  

---

## 🐦 Overview

This repository contains our full implementation of a **Concept Bottleneck Model (CBM)** for interpretable fine-grained bird species classification using the **CUB-200-2011 dataset**.

Standard CNNs achieve high accuracy but operate as *black boxes*. CBMs introduce a middle layer of **human-interpretable concepts** (e.g., wing color, bill shape, breast pattern) that the model must predict before classifying species.

This repository includes:

- Baseline ResNet-18 classifier (**x → y**)
- Concept predictor (**x → ĉ**)
- Label-from-concepts classifier (**ĉ → y**)
- Full CBM pipeline (**x → ĉ → y**)
- Concept explanations and manual interventions
- Complete preprocessing pipeline for CUB
- Training + evaluation scripts for all components

---

# 📁 Repository Structure
project-root/
│
├── data/
│ ├── cub_raw/ # Place downloaded dataset here
│ ├── cub_csvs/ # Output of preprocessing - created by the program
│
├── src/
│ ├── prepare_cub.py # Dataset preprocessing pipeline
│ ├── baseline.py # Baseline ResNet-18 (x → y)
│ ├── concept_predictor.py # Concept model (x → ĉ)
│ ├── c2y_classifier.py # Label-from-concepts classifier (ĉ → y)
│ ├── cbm_pipeline.py # Full CBM evaluation
│ ├── train_utils.py # Shared training utilities
│ ├── evaluate.py # Evaluation scripts
│ ├── explain.py # Concept explanations & interventions
│
├── checkpoints/ # Saved model weights
│
├── requirements.txt
└── README.md

---

# ⚙️ Installation

## 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

## 2. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate      # Mac / Linux
venv\Scripts\activate         # Windows

## 3. Install Dependencies
pip install -r requirements.txt


Key libraries:
PyTorch & Torchvision
NumPy / Pandas
scikit-learn
Matplotlib
tqdm
Pillow

📥 Downloading the Dataset

The dataset is not included because it is too large for GitHub.

Download CUB-200-2011:

Official page:
http://www.vision.caltech.edu/datasets/cub_200_2011/

Place it here:
data/cub_raw/CUB_200_2011/

🧹 Preprocessing the Dataset
Run this to generate clean CSVs and concept matrices:
python src/prepare_cub.py \
    --cub_root data/cub_raw/CUB_200_2011 \
    --output_dir data/cub_processed

This script:
Parses metadata
Extracts 312 concept attributes
Builds train/val/test splits
Produces numpy matrices for fast training

🏋️ Training the Models
You may train components individually or the entire CBM pipeline.
🔵 1. Train the Baseline ResNet-18 (x → y)
python src/baseline.py \
    --data_dir data/cub_processed \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001 \
    --save_path checkpoints/baseline_best.pt

🟢 2. Train the Concept Predictor (x → ĉ)
python src/concept_predictor.py \
    --data_dir data/cub_processed \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001 \
    --save_path checkpoints/concept_predictor_best.pt

🟡 3. Train the Label-From-Concepts Classifier (ĉ → y)
Using ground truth concepts (“oracle CBM”):
python src/c2y_classifier.py \
    --concept_dir data/cub_processed \
    --epochs 15 \
    --lr 0.001 \
    --save_path checkpoints/label_from_concepts_best.pt

🔴 4. Evaluate the Full CBM Pipeline (x → ĉ → y)
python src/cbm_pipeline.py \
    --concept_model checkpoints/concept_predictor_best.pt \
    --classifier checkpoints/label_from_concepts_best.pt \
    --data_dir data/cub_processed \
    --evaluate

📊 Evaluation
Evaluate any model:
python src/evaluate.py --model_type baseline --checkpoint checkpoints/baseline_best.pt
python src/evaluate.py --model_type concept --checkpoint checkpoints/concept_predictor_best.pt
python src/evaluate.py --model_type c2y --checkpoint checkpoints/label_from_concepts_best.pt
python src/evaluate.py --model_type cbm --config configs/cbm.yaml

📝 Notes for Graders
Dataset not included due to size limits
All scripts fully reproducible from raw data
Preprocessing must be run before training
Checkpoints included for convenience
Explanation engine demonstrates interpretability criteria
