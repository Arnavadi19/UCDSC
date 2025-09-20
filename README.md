# UCDSC: UnCertainty aware Deep Simplex Classifier for Imbalanced Medical Image Datasets

This repository contains the implementation code for the paper "UCDSC: UnCertainty aware Deep Simplex Classifier for Imbalanced Medical Image Datasets".

## Dataset Setup

### MedMNIST Dataset
The MedMNIST dataset can be downloaded from:
- **Official source**: https://zenodo.org/records/10519652

### Background Data
For extended experiments, download the 300K random images:
- **Source**: https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy

### Augmented Skin Conditions dataset
- **Source**: https://www.kaggle.com/datasets/syedalinaqvi/augmented-skin-conditions-image-dataset/data

### Basic Training
Run the main training script with default parameters:
```bash
python NirvanaOSR.py --dataset dataset-name --dataroot ./data --outf ./results
```

### Hyperparameter Configuration

| Parameter | Description |
|-----------|-------------|
| `--batch-size` | Training batch size | 
| `--lr` | Learning rate | 
| `--max-epoch` | Maximum training epochs |
| `--optim` | Optimizer to be used |
| `--margin` | Margin for hinge loss | 
| `--Expand` | Expand factor of centers |
| `--uncertainty-weight` | Weight for uncertainty loss |
| `--outlier-weight` | Weight for outlier triplet loss |
| `--model` | Backbone network to be used |



## Acknowledgments

This codebase is derived from the Deep Simplex Classifier implementation available at https://github.com/Cevikalp/dsc. We thank the authors for making their code available.