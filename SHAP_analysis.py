import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import models, transforms

from skimage.segmentation import slic
from skimage.util import img_as_float


CHECKPOINT_PATH = "./checkpoints/vgg16_flowers_epoch_100.pt"   # <-- change to your checkpoint path
NUM_CLASSES = 5
CLASS_NAMES = ["daisy","dandelion","rose","sunflower","tulip"]  # adjust if different
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Preprocessing (VGG default)
# ----------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

to_numpy = transforms.Compose([
    transforms.Resize((224, 224)),
])

# ----------------------
# Model load helper (finetuned VGG16)
# ----------------------
def load_finetuned_vgg16(checkpoint_path, num_classes=3, device=DEVICE):
    """
    Load VGG16 and adapt classifier to num_classes, then load checkpoint state_dict.
    Accepts either full model state_dict or a dict with 'model_state_dict' key.
    """
    model = models.vgg16(pretrained=False)  # don't load imagenet weights; we'll load checkpoint
    # Replace final classifier layer (classifier[6] in torchvision's VGG)
    in_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(in_features, num_classes)

    # load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Try different checkpoint structures
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # If keys were saved with 'module.' prefix (DataParallel), strip it
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    model.to(device).eval()
    return model

# ----------------------
# Superpixel & masking helpers (same as earlier)
# ----------------------
def superpixel_segment(pil_image, n_segments=50, compactness=10):
    image = np.array(to_numpy(pil_image)).astype(float) / 255.0
    segments = slic(img_as_float(image), n_segments=n_segments, compactness=compactness, start_label=0)
    return segments, image

def mask_image_with_superpixels(image, segments, mask, background=None):
    out = image.copy()
    if background is None:
        bg = image.mean(axis=(0,1))
    else:
        bg = np.array(background).reshape(1,1,3)
    for seg_id in range(mask.shape[0]):
        if not mask[seg_id]:
            out[segments == seg_id] = bg
    return out

def model_predict_batch(model, batch_images, return_logits=False):
    """
    batch_images: numpy array (N x H x W x 3) in [0,1]
    returns: logits (N x C) if return_logits True, otherwise softmax probabilities
    """
    model.eval()
    with torch.no_grad():
        tensors = []
        for img in batch_images:
            pil = Image.fromarray((img*255).astype(np.uint8))
            t = preprocess(pil)
            tensors.append(t)
        batch = torch.stack(tensors).to(DEVICE)
        logits = model(batch)
        if return_logits:
            return logits.cpu().numpy()
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

# Kernel SHAP weight and mask generation (same)
def kernel_shap_weight(M, s):
    if s == 0 or s == M:
        return 1e-6
    comb = math.comb(M, s)
    return (M - 1) / (comb * s * (M - s))

def generate_masks_random(M, nsamples, seed=None):
    rng = np.random.default_rng(seed)
    masks = np.zeros((nsamples, M), dtype=bool)
    for i in range(nsamples):
        s = rng.integers(1, M)  # avoid extremes
        ones = rng.choice(M, size=s, replace=False)
        masks[i, ones] = True
    return masks

def solve_weighted_ridge(X, y, weights, l2_reg=1e-6):
    W = weights.reshape(-1)
    WX = X * W[:, None]
    A = X.T @ WX
    reg = l2_reg * np.eye(A.shape[0])
    reg[0,0] = 0.0
    A += reg
    b = np.linalg.solve(A, X.T @ (W * y))
    return b

# ----------------------
# Kernel SHAP main adapted for finetuned model
# ----------------------
def kernel_shap_image(model, pil_image, nsamples=800, n_superpixels=50, compactness=10,
                      target_class=None, batch_size=64, seed=None, l2_reg=1e-6,
                      background=None, explain_logits=False):
    """
    explain_logits: if True, uses logits as target y (recommended when model trained with logits).
    Otherwise uses softmax probabilities (0..1).
    """
    model = model.to(DEVICE)
    model.eval()

    segments, image = superpixel_segment(pil_image, n_segments=n_superpixels, compactness=compactness)
    M = segments.max() + 1
    if background is None:
        background = image.mean(axis=(0,1))

    # Determine target class using model on original image
    orig_logits = model_predict_batch(model, np.expand_dims(image, axis=0), return_logits=True)[0]
    orig_probs = torch.softmax(torch.from_numpy(orig_logits), dim=0).numpy()
    top_pred = int(np.argmax(orig_probs))
    if target_class is None:
        target_class = top_pred

    # Generate masks and append empty + full for baseline/full
    masks = generate_masks_random(M, nsamples, seed=seed)
    special = np.vstack([np.zeros((1,M), dtype=bool), np.ones((1,M), dtype=bool)])
    masks = np.vstack([masks, special])
    nsamples_total = masks.shape[0]

    # Create masked images
    masked_images = []
    for i in range(nsamples_total):
        mi = mask_image_with_superpixels(image, segments, masks[i], background=background)
        mi = np.clip(mi, 0.0, 1.0)
        masked_images.append((mi * 255).astype(np.uint8))
    masked_images = np.array(masked_images)

    # Predict (logits or probs)
    preds = []
    for i in range(0, nsamples_total, batch_size):
        batch = masked_images[i:i+batch_size].astype(np.uint8)
        batch_float = batch.astype(float)/255.0
        if explain_logits:
            p = model_predict_batch(model, batch_float, return_logits=True)
        else:
            p = model_predict_batch(model, batch_float, return_logits=False)
        preds.append(p)
    preds = np.vstack(preds)
    if explain_logits:
        y = preds[:, target_class]            # logits
    else:
        y = preds[:, target_class]            # probabilities

    # Regression matrix
    X = np.concatenate([np.ones((nsamples_total, 1)), masks.astype(float)], axis=1)
    s_counts = masks.sum(axis=1)
    weights = np.array([kernel_shap_weight(M, int(s)) for s in s_counts])

    coef = solve_weighted_ridge(X, y, weights, l2_reg=l2_reg)
    base_value = coef[0]
    shap_values = coef[1:]
    full_val = y[-1]  # last is full mask
    return {
        "shap_values": shap_values,
        "segments": segments,
        "base_value": base_value,
        "full_value": full_val,
        "predicted_class": target_class,
        "orig_probs": orig_probs,
        "image": image
    }

# ----------------------
# Visualization
# ----------------------
def shap_to_heatmap(shap_values, segments, image, normalize=True):
    H, W, _ = image.shape
    heat = np.zeros((H, W), dtype=float)
    for i, val in enumerate(shap_values):
        heat[segments == i] = val
    if normalize:
        maxabs = np.max(np.abs(heat)) + 1e-12
        heat = heat / maxabs
    return heat

def plot_shap_overlay(pil_image, heatmap, alpha=0.6, cmap='bwr', title=None):
    plt.figure(figsize=(7,7))
    img = np.array(to_numpy(pil_image)).astype(float) / 255.0
    plt.imshow(img)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=-1, vmax=1)
    plt.colorbar(label='SHAP (normalized)')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    IMAGE_PATH = "./dataset/test_split/dandelion/3998275481_651205e02d.jpg"   # <-- a sample iris flower image
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Put an image at {IMAGE_PATH} or change IMAGE_PATH.")

    pil = Image.open(IMAGE_PATH).convert("RGB")

    # Load your finetuned VGG16
    vgg = load_finetuned_vgg16(CHECKPOINT_PATH, num_classes=NUM_CLASSES, device=DEVICE)

    # Run Kernel SHAP (using logits is usually better for classification explanations)
    print("Done.")
    result = kernel_shap_image(vgg, pil_image=pil,
                              nsamples=700,
                              n_superpixels=50,
                              compactness=10,
                              target_class=None,
                              batch_size=64,
                              seed=42,
                              l2_reg=1e-6,
                              background=None,
                              explain_logits=True)   

    shap_vals = result["shap_values"]
    segments = result["segments"]
    base_val = result["base_value"]
    full_val = result["full_value"]
    predicted_class = result["predicted_class"]
    orig_probs = result["orig_probs"]
    print("Done.")
    print(f"Predicted class: {predicted_class} ({CLASS_NAMES[predicted_class]}), orig prob: {orig_probs[predicted_class]:.4f}")
    print(f"Base value: {base_val:.6f}, Full-image value: {full_val:.6f}")
    print(f"Sum SHAP ~ {shap_vals.sum():.6f}, base + sum ~ {base_val + shap_vals.sum():.6f}")

    heat = shap_to_heatmap(shap_vals, segments, result["image"], normalize=True)
    plot_shap_overlay(pil, heat, alpha=0.6, title=f"SHAP overlay for class {CLASS_NAMES[predicted_class]}")