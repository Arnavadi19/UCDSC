# UCDSC: UnCertainty aware Deep Simplex Classifier for Imbalanced Medical Image Datasets

This repository contains the implementation code for the paper "UCDSC: UnCertainty aware Deep Simplex Classifier for Imbalanced Medical Image Datasets".



### MedMNIST Dataset
The MedMNIST dataset can be downloaded from:
- **Official source**: https://zenodo.org/records/10519652

### Background Data
For extended experiments, download the 300K random images:
- **Source**: https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy

## Usage

Run the main training script with default parameters:
```bash
python NirvanaOSR.py --dataset dataset-name --dataroot ./data --outf ./results
```



## Acknowledgments

This codebase is derived from the Deep Simplex Classifier implementation available at https://github.com/Cevikalp/dsc. We thank the authors for making their code available.
