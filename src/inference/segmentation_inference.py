from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from src.segmentation.hsegformer_backbone import HSegFormerBackbone
from src.segmentation.hsegformer_decoder import HSegFormerDecoder
from src.segmentation.hsegformer_fusion import HSegFormerFusion

class HSegFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HSegFormerBackbone()
        self.fusion = HSegFormerFusion()
        self.decoder = HSegFormerDecoder()

    def forward(self, x):
        return self.decoder(self.fusion(self.backbone(x)))

def load_segmentation_model(root, device=None):
    root = Path(root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = root / "models/segmentation/hsegformer_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model = HSegFormer()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, device

def predict_segmentation(model, image, device, threshold=0.5):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
        logits = output["main"]
        probabilities = torch.sigmoid(logits)
        mask = (probabilities >= threshold).to(torch.uint8)
    return (
        probabilities[0, 0].cpu().numpy(),
        mask[0, 0].cpu().numpy(),
    )
