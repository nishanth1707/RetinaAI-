import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.hooks = []

        if target_layer is None:
            target_layer = model.backbone.bn2

        self._register_hooks(target_layer)

    def _register_hooks(self, layer):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(layer.register_forward_hook(forward_hook))
        self.hooks.append(layer.register_full_backward_hook(backward_hook))

    def generate(self, x, class_idx=None):
        logits = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], device=x.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()

        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def __del__(self):
        for hook in self.hooks:
            hook.remove()


def apply_gradcam_heatmap(image_np, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if image_np.shape[:2] != cam.shape[:2]:
        image_np = cv2.resize(image_np, (cam.shape[1], cam.shape[0]))

    overlay = (heatmap * alpha + image_np * (1 - alpha)).astype(np.uint8)
    return overlay, heatmap
