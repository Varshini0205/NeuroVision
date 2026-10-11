# NeuroVision 🧠

**Brain MRI Segmentation and Tumor Classification using Deep Learning**

NeuroVision V2 is a research-oriented project that analyzes brain MRI images using deep learning. It combines **HSegFormer** for brain tumor segmentation and **ConvNeXt-Tiny** for four-class image classification through a web-based interface.

## Features

- **Tumor segmentation:** Uses HSegFormer to generate a predicted segmentation mask.
- **Image classification:** Uses ConvNeXt-Tiny to classify MRI images into four categories.
- **Region of interest (ROI):** Provides a localized region based on the predicted segmentation.
- **Visual results:** Displays the MRI image, segmentation mask, overlay, and ROI.
- **Web interface:** Provides an interface for uploading an MRI image and viewing the model outputs.
- **API backend:** Built with FastAPI.

## Classification Categories

The classifier predicts one of the following categories:

- `glioma`
- `meningioma`
- `no_tumor`
- `pituitary`

## Models

| Task | Model |
|---|---|
| Image segmentation | HSegFormer |
| Image classification | ConvNeXt-Tiny |
| Backend API | FastAPI |

The segmentation model generates a predicted mask, while the classification model predicts the image category.

## Project Structure

```text
NeuroVision/
├── app/
│   ├── static/
│   │   └── index.html
│   ├── __init__.py
│   └── backend.py
├── configs/
│   └── label_config.json
├── models/
│   └── README.md
├── src/
│   ├── inference/
│   │   ├── __init__.py
│   │   └── segmentation_inference.py
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── hsegformer_backbone.py
│   │   ├── hsegformer_decoder.py
│   │   └── hsegformer_fusion.py
│   └── __init__.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or a compatible Python version
- pip
- PyTorch
- A working environment for the dependencies listed in `requirements.txt`

A CUDA-compatible GPU is recommended for faster inference. CPU inference may also be possible, but it can be slower.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Varshini0205/NeuroVision.git
cd NeuroVision
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Model Checkpoints

The trained model checkpoint files are not included in this GitHub repository because of their large size.

Before running inference, obtain the required trained checkpoints and place them in the expected locations:

```text
models/
├── segmentation/
│   └── hsegformer_best.pt
└── classification/
    └── convnext_tiny_4class_candidate_best.pt
```

The application requires the checkpoints expected by its backend. The application will not be able to perform model inference if the required files are missing.

## Run the Application

From the repository root, start the FastAPI server:

```bash
python -m uvicorn app.backend:app --host 0.0.0.0 --port 8000
```

Open the application in your browser:

```text
http://localhost:8000
```

If the backend is running on a different port, use the corresponding local URL.

## Using NeuroVision

1. Start the backend server.
2. Open the web interface in your browser.
3. Upload a supported brain MRI image.
4. Submit the image for analysis.
5. Review the predicted class and segmentation results.
6. Inspect the segmentation overlay and region of interest, when available.

Results depend on the input image, trained model checkpoints, and model predictions.

## API

The backend provides an API endpoint for image analysis:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Checks backend and model status |
| `/api/scan` | POST | Processes an uploaded MRI image |

The health endpoint can be checked at:

```text
http://localhost:8000/api/health
```

The scan endpoint accepts an image upload and returns the available prediction results.

## Important Notes

- NeuroVision V2 is intended for **research and educational purposes**.
- It is not a medical device and must not be used as a substitute for professional medical diagnosis.
- Predictions may be incorrect and should be reviewed by qualified medical professionals.
- Do not upload identifiable patient information to an untrusted or publicly accessible deployment.
- Dataset access and usage must comply with the applicable dataset licenses and terms.
- Model performance depends on the training data, preprocessing, and input image quality.

## Limitations

- Predictions are not guaranteed to be accurate for every MRI image.
- Performance may vary across scanners, imaging protocols, and datasets.
- Segmentation quality depends on the learned model and the input image.
- The trained checkpoints must be obtained separately from the source code.

## Future Improvements

Potential future improvements include:

- Additional evaluation on appropriately authorized datasets
- Expanded testing and documentation
- More robust error handling
- Additional model interpretability tools
- Deployment and usability improvements

## License

No license has been specified yet. Until a license is added to this repository, reuse and redistribution permissions should not be assumed.

## Author

**Varshini**

GitHub: [@Varshini0205](https://github.com/Varshini0205)

---

*NeuroVision V2 — Deep learning for brain MRI segmentation and classification.*
