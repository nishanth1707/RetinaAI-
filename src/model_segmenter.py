import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DRSegmenter(nn.Module):
    def __init__(self, num_lesions=4, encoder_name="efficientnet-b0"):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_lesions,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)


def load_segmenter(model_path, num_lesions=4, device="cpu"):
    model = DRSegmenter(num_lesions=num_lesions)
    state = torch.load(model_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_segmenter(model, image_tensor, device="cpu", threshold=0.5):
    model.eval()
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probs = torch.sigmoid(logits)
    masks = (probs > threshold).cpu().numpy()
    return masks[0]
