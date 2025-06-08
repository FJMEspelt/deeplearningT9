# 🌄 Intel Image Classification – Deep Learning Practice (T9)

This project is the final evaluation exercise for the Deep Learning module. It implements and compares multiple convolutional neural network (CNN) models for multi-class image classification on the **Intel Image Classification** dataset.

## 📁 Dataset

The dataset comes from [Kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) and contains **6 classes** of natural scenes:

- 0: Buildings  
- 1: Forest  
- 2: Glacier  
- 3: Mountain  
- 4: Sea  
- 5: Street  

Each image is RGB and has shape **150×150×3**.

- `seg_train/` contains 14,000 labeled training images.
- `seg_test/` contains 3,000 labeled test images.
- `seg_pred/` contains 7,000 unlabeled images (not used in this project).

## 🧠 Models implemented

| Model ID                      | Script                         | Description                                              |
|-------------------------------|--------------------------------|----------------------------------------------------------|
| **1 – Base CNN**              | `scripts/02_model_base.py`     | Simple convolutional network with 3 blocks and dense head. |
| **2 – Deep CNN**              | `scripts/03_model_deep.py`     | Augmented CNN with extra conv blocks, batch norm & dropout. |
| **3 – Hyperparameter Tuning** | `scripts/04_hyperparam_tuning.py` | RandomSearch over L1/L2, initializers, activations, dropout, LR. |
| **4 – Transfer Learning**     | `scripts/05_transfer_learning.py` | Fine-tuning ResNet50 pretrained on ImageNet in two phases. |
| **5 – Data Augmentation**     | `scripts/06_data_augmentation.py`| Retraining best model with ImageDataGenerator augmentations. |
| **6 – Evaluation Plot**       | `scripts/07_evaluation.py`     | Compares `val_accuracy` curves for all models.           |
| **7 – Evaluation Table**      | `scripts/08_evaluation_table.py`| Table of test accuracy for all saved models.             |

## 📂 Folder structure

intel_image_classification/
│
├── src/                         ← Source code modules
│   ├── config.py                ← Configuration constants (paths, hyperparams)
│   └── data_utils.py            ← Data generator utilities and preprocessing
|   └── model_zoo.py   
├── scripts/                      ← Python scripts for training and evaluation
│   ├── 02_model_base.py
│   ├── 03_model_deep.py
│   ├── 04_hyperparam_tuning.py
│   ├── 05_transfer_learning.py
│   ├── 06_data_augmentation.py
│   ├── 07_evaluation.py
│   └── 08_evaluation_table.py
|
├── saved_models/                ← Trained models in .keras format
├── histories/                   ← Training histories (metrics.json, plots)
├── reports/
│   └── figures/                  ← Final comparison plots & tables


## 🚀 How to run

1. **Install dependencies**:
   ```bash
   conda env create -f environment.yml
   conda activate dl
   ```
2. **Train or re-train models**:
   ```bash
   python -m scripts/02_model_base
   python -m scripts/03_model_deep
   python -m scripts/04_hyperparam_tuning
   python -m scripts/05_transfer_learning
   python -m scripts/06_data_augmentation
   ```
3. **Compare validation accuracy curves**:
   ```bash
   python -m scripts/07_evaluation
   ```
4. **Generate test accuracy table**:
   ```bash
   python -m scripts/08_evaluation_table
   ```
5. **Results**:
   - Models in `saved_models/`
   - Histories and plots in `histories/` and `reports/figures/`