import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

class CO2Regressor:
    def __init__(self, model_path='co2_regressor_model.pth'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path)

    def _build_model(self):
        m = models.mobilenet_v3_small(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        head = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_f, 1))
        return nn.Sequential(m, head)

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location=self.device)
        model = self._build_model()
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device); model.eval()

        mean = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
        std  = ckpt.get("normalize_std",  [0.229, 0.224, 0.225])
        self.log_target = ckpt.get("log_target", False)

        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def predict(self, image_path: str) -> float:
        """Return predicted CO2 value for a single image."""
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.model(x).squeeze().item()
        return float(torch.expm1(torch.tensor(y)).item()) if self.log_target else float(y)

# Example usage
if __name__ == "__main__":
    reg = CO2Regressor()
    # Try a sample from your validate split if available
    sample = None
    if os.path.exists('dataset_products/validate'):
        # pick any nested image
        for root, _, files in os.walk('dataset_products/validate'):
            imgs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if imgs:
                sample = os.path.join(root, imgs[0]); break
    if sample:
        print("Testing with:", sample)
        pred = reg.predict(sample)
        print("Predicted CO₂:", pred)
    else:
        print("Put an image path and call CO2Regressor().predict(path).")
