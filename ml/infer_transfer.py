#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def build_model(num_classes=2):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_reasons(ai_confidence: int):
    if ai_confidence >= 70:
        return [
            "Transfer model found strong synthetic-generation cues.",
            "Global structure and texture patterns are atypical for camera-captured faces.",
            "Confidence is high under the ResNet-based classifier.",
        ]
    if ai_confidence >= 50:
        return [
            "Transfer model found moderate synthetic-like cues.",
            "Some regions look less consistent with natural camera capture.",
            "Prediction is near the decision boundary; verify with additional images.",
        ]
    if ai_confidence <= 30:
        return [
            "Transfer model found strong natural-photo cues.",
            "Face structure and texture align with camera-captured patterns.",
            "Confidence is high under the ResNet-based classifier.",
        ]
    return [
        "Transfer model found moderate natural-photo cues.",
        "Most regions align with camera-captured face characteristics.",
        "Prediction is near the decision boundary; verify with additional images.",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="ml/artifacts/best_resnet18.pt")
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.exists():
        print(json.dumps({"ok": False, "error": "Image file not found."}))
        return

    if not checkpoint_path.exists():
        print(json.dumps({"ok": False, "error": "Checkpoint not found."}))
        return

    device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    p_fake = float(probs[0])
    ai_confidence = int(round(max(0.0, min(1.0, p_fake)) * 100))

    print(
        json.dumps(
            {
                "ok": True,
                "aiConfidence": ai_confidence,
                "reasons": get_reasons(ai_confidence),
                "source": "transfer-resnet18",
            }
        )
    )


if __name__ == "__main__":
    main()
