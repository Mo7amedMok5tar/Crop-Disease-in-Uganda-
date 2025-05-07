# Crop Disease Detection – Kaggle Competition (Uganda Dataset)

This project develops an image classification model using PyTorch to identify crop diseases from leaf images. It was built for the [Kaggle competition: Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification), and incorporates ensembling, cross-validation, and deployment via a Streamlit web app.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Project Structure](#project-structure)  
3. [Data Preparation](#data-preparation)  
   - [Dataset Overview](#dataset-overview)  
   - [Handling Imbalanced Data](#handling-imbalanced-data)  
4. [Model Training](#model-training)  
   - [Cross-Validation Strategy (K-Fold)](#cross-validation-strategy-k-fold)  
   - [Model Architecture](#model-architecture)  
   - [Callbacks and Training Enhancements](#callbacks-and-training-enhancements)  
5. [Model Evaluation](#model-evaluation)  
6. [Model Ensembling](#model-ensembling)  
7. [Deployment](#deployment)  
   - [Streamlit Web App](#streamlit-web-app)  
8. [How to Run](#how-to-run)  
9. [Results Summary](#results-summary)

---

## Project Overview

This project aims to build a robust crop disease classifier using PyTorch. It tackles class imbalance, leverages K-Fold cross-validation, and ensembles the final predictions for enhanced accuracy. A Streamlit app is also built for easy interaction and real-time prediction.

---

## Project Structure

```
├── notebooks/           # EDA and model training notebooks
├── src/
│   ├── utils/           # Helper functions (data loading, training, etc.)
├── model/               # Trained model checkpoints
│   ├── fold_0/          # Model for Fold 0
│   ├── fold_1/          # Model for Fold 1
│   └── ...              # etc.
├── app/                 # Streamlit web app code
```

---

## Data Preparation

### Dataset Overview

The dataset includes images of diseased and healthy leaves of cassava crops. The labels represent different disease types.

- 📂 **Dataset link**: [Cassava Plant Disease – Merged 2019/2020](https://www.kaggle.com/datasets/srg9000/cassava-plant-disease-merged-20192020)  
- 🏆 **Competition link**: [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)

### Handling Imbalanced Data

The dataset was significantly imbalanced. To address this:

- **Under-sampling** of majority classes was applied to ensure fair training across folds.

---

## Model Training

### Cross-Validation Strategy (K-Fold)

To ensure model generalization and robustness, a 5-fold **Stratified K-Fold Cross-Validation** was implemented. Each fold trains and validates the model on different splits of the data.

### Model Architecture

- CNN architecture with transfer learning (ResNet, EfficientNet, etc.)
- Layers include:
  - Convolutional blocks
  - Dropout
  - Fully connected layers

### Callbacks and Training Enhancements

- **EarlyStopping**: Stop training when validation loss stops improving.
- **Learning Rate Scheduler**: ReduceLROnPlateau to fine-tune training.
- **Model Checkpointing**: Best weights saved per fold.

---

## Model Evaluation

Each fold was evaluated using:

- **Validation Accuracy**: ~91% average across all folds.
- **Validation Loss**: Averaging around 0.4.
- **Confusion Matrix**: To assess per-class performance.

---

## Model Ensembling

After training on 5 folds, predictions from all five models were combined using **soft voting** to produce the final classification output, improving robustness and accuracy.

---

## Deployment

### Streamlit Web App

A user-friendly interface was created using **Streamlit**, allowing users to upload an image and receive the predicted disease class.

**Features:**

- Image upload  
- Real-time prediction  
- Visual feedback  

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Mo7amedMok5tar/Crop-Disease-in-Uganda-.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

> **Note**: Ensure trained models are located in the `model/` directory inside subfolders per fold as expected by the app.

---

## Results Summary

| Metric              | Value     |
|---------------------|-----------|
| Avg. Val Accuracy   | ~91%      |
| Avg. Val Loss       | ~0.4      |
| Final Ensemble      | 5-fold CNN |
| Deployment          | Streamlit Web App (Local) |

---

**Author**: Mohamed Mokhtar  
**Contact**: mohamedmokhtar26027@gmail.com
