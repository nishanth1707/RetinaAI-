import torch
import torch.nn as nn
import timm


class DRClassifier(nn.Module):
    def __init__(self, num_classes=5, model_name="efficientnet_b0"):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        output = self.classifier(features)
        return output

    def get_features(self, x):
        return self.backbone(x)


def load_classifier(model_path, num_classes=5, device="cpu"):
    model = DRClassifier(num_classes=num_classes)
    state = torch.load(model_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_classifier(model, image_tensor, device="cpu"):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    return pred_class, confidence, probs[0].cpu().numpy()
