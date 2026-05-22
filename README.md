# Mines 2024 Data Challenge

## ⭐ STAR Summary

**Situation**  
The Mines x Invent 2024 Data Challenge required classifying agricultural parcels from multi-temporal Sentinel-2 satellite imagery using the PASTIS dataset — a large-scale benchmark for panoptic segmentation of agricultural fields across time.

**Task**  
Build and evaluate machine learning models capable of accurately identifying crop types from sequences of satellite images acquired over a full agricultural season, where the temporal dimension is a key discriminating signal.

**Action**  
Trained and evaluated models on the PASTIS dataset, leveraging its multi-temporal structure. Two approaches were compared:

- A **temporal deep learning baseline** exploiting the time-series nature of the acquisitions (multiple Sentinel-2 images captured at different dates over the same parcel), designed to capture crop phenology and growth dynamics.

- A **Vision Transformer (ViT)** approach: ViTs adapt the Transformer architecture — originally developed for NLP — to image data by dividing each satellite image into fixed-size spatial patches, embedding them linearly, and processing the resulting sequence with a standard Transformer encoder. Self-attention mechanisms allow the model to capture long-range spatial dependencies that convolutional networks may miss. Applied to satellite imagery, ViTs can relate distant regions of a field and generalize well when pre-trained on large image datasets. We compared its spatial feature extraction capabilities against the temporal model's ability to capture crop evolution over time.

**Result**  
The two approaches offer complementary strengths: the temporal baseline excels at capturing crop growth dynamics across acquisition dates, while the ViT better captures fine-grained spatial structure within individual acquisitions. Combining both perspectives provides the most robust classification signal.

---

Welcome to the **Mines x Invent 2024 Data Challenge**! This repository is designed to help Mines Paris students get started quickly with the challenge.

Link to the competition: [Mines 2024 Data Challenge](https://www.kaggle.com/competitions/data-challenge-invent-mines-2024/overview) (works only if you already entered the competition)

## 🏗️ Setup Instructions

### 1. Clone the repo

```console
git clone https://github.com/LouisStefanuto/hackathon-mines-invent-2024.git
```

### 2. Create a Virtual Environment

First, you'll need to create a Python virtual environment. Make sure you're using Python 3.12:

```console
conda create -n env_challenge python=3.12
conda activate env_challenge
```

> **Note:** For those familiar with Poetry, you can by-pass the Conda step. We add it to simplify the install.

### 3. Install Dependencies

Next, install the required dependencies for the project:

```console
pip install poetry
poetry install --with dev
```

For more information about Poetry, check out their [**documentation**](https://python-poetry.org).

## 📊 Dataset

The challenge is based on the PASTIS dataset from Vivien Sainte Fare Garnot and Loic Landrieu.

The dataset for this challenge is quite large, but for initial experiments, we recommend starting with the mini dataset.

The data is available on Kaggle in the [Data section](https://www.kaggle.com/competitions/data-challenge-invent-mines-2024/data). It contains:

- a **mini Dataset**: 10 samples, to test the starter kit
- and the **full Dataset**: much larger (40Gb), download it at home.

## 🧪 Running the Demo

To get a quick start, check out the `demo.ipynb` notebook. It will guide you through loading and visualizing the dataset, helping you familiarize yourself with the data.

## Training example

We also provide a minimal training pipeline. Note it contains the bare minimal material.

```console
python baseline.train.py
```

## 📚 Documentation

This static documentation website is built using MkDocs and hosted on GitHub Pages. Make sure to document your work thoroughly—your future self will thank you for presenting your project clearly! 😇
