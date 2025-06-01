# 🌄 Intel Image Classification Project – Deep Learning (T9 Final Practice)

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

| Model ID | Description |
|----------|-------------|
| **1** | Base CNN: a simple convolutional + pooling + dense classifier |
| **2** | Deep CNN: additional convolutional blocks and dropout layers |
| **3** | Hyperparameter tuning: Keras Tuner applied to Model 2 |
| **4** | Transfer learning: fine-tuned `MobileNetV2` pretrained on ImageNet |
| **5** | Data augmentation: applied to best model using `ImageDataGenerator` |

All models are trained using **Keras + TensorFlow 2.15**.

## 📂 Folder structure

intel_image_classification/
│
├── data/
│ ├── raw/ ← Original dataset (seg_train, seg_test, seg_pred)
│ └── processed/ ← Optional .npy conversion (if used)
│
├── notebooks/ ← All step-by-step notebooks
│ ├── 00_dataset_exploration.ipynb
│ ├── 01_preprocessing.ipynb
│ ├── 02_model_base.ipynb
│ ├── 03_model_deep.ipynb
│ ├── 04_hyperparam_tuning.ipynb
│ ├── 05_transfer_learning.ipynb
│ ├── 06_data_augmentation.ipynb
│ └── 07_evaluation.ipynb
│
├── src/ ← Project modules
│ ├── config.py
│ ├── data_utils.py
│ ├── model_zoo.py
│ ├── train_utils.py
│ └── evaluate.py
│
├── saved_models/ ← Trained .h5 models (downloaded from Colab)
├── histories/ ← Training history .json files
├── reports/
│ └── figures/ ← Final accuracy plot
│
├── environment.yml ← Conda env definition
├── get_data.py ← Script to download/extract dataset from Kaggle
├── run_local.bat ← Activates env and launches Jupyter Lab
└── README.md


## 🚀 How to run

### 🖥️ Local setup (Surface / PC)

1. Create Conda env:

   ```bash
   conda env create -f environment.yml
   conda activate intel_cnn

2. (Once) Create Kernel for jupyter
python -m ipykernel install --user --name intel_cnn --display-name "Python (intel_cnn)"

3. Download the dataset:


4. Launch Jupyter Lab:
run_local.bat

5. Google Colab (for model training)
    1. Upload the project folder to your Drive or GitHub.
    2. Open notebooks 02 → 06 individually.
    3. Train each model (GPU recommended).

    4. Download:
        Trained models (saved_models/*.h5)
        Histories (histories/*.json)
        Move them to your local folders to run evaluation.

6. Evaluation
python -m src.evaluate

This script:
    1. Loads each model in saved_models/
    2. Evaluates it on the test set
    3. Produces a CSV and markdown table
    4. Generates a val_accuracy comparison plot

    Example output:
Model Name	Test Accuracy
data_aug	0.9143
fine_tuning	0.9081
hp	0.9012
advance_model	0.8944
base_model	0.8729

7. Deliverables
✅ All .ipynb notebooks
✅ Corresponding .html exports
✅ README.md
✅ Final accuracy plot (reports/figures/val_comparison.png)
❌ No need to submit: raw images, .npy, .h5 models, or seg_pred

8. Requirements
Python 3.10

Conda or virtualenv

TensorFlow 2.15

Keras Tuner

Kaggle CLI (pip install kaggle)

9. 🙌 Credits
Dataset by Puneet6060 on Kaggle
Project created as part of the Deep Learning module (T9)
Developed by Javier Molina Espelt