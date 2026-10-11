# NeuroVision V2

Research demonstration for 2D brain MRI segmentation and four-class image classification.

## Pipeline

1. Upload a supported 2D MRI image.
2. HSegFormer generates a predicted segmentation mask.
3. The interface displays the mask, overlay, and extracted region of interest.
4. ConvNeXt-Tiny produces scores for glioma, meningioma, no_tumor, and pituitary.

## Requirements

Python 3.10 or 3.11 is a sensible starting point. Install compatible PyTorch and
TorchVision builds for your deployment hardware, then install the remaining packages
from `requirements.txt`.

## Run locally

The backend must use a project-root path derived from the location of this repository.
The copied backend currently needs a deployment-path review before it can be launched
from this release folder.

## Models

See `models/README.md` for checkpoint requirements and distribution notes.

## Research notice

This is a research demonstration, not a medical diagnostic device. Predictions may be
incorrect and must not be used as the sole basis for clinical decisions.
