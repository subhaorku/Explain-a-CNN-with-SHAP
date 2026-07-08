import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math, time

import torch
import torch.nn.functional as F
from torchvision import models, transforms

import shap
from skimage.segmentation import slic
from skimage.util import img_as_float

IMAGE_PATH = "./dataset/test_split/dandelion/3998275481_651205e02d.jpg"                # <- image to explain
CHECKPOINT_PATH = "./checkpoints/vgg16_flowers_epoch_100.pt"   # <-- change to your checkpoint path
NUM_CLASSES = 5
CLASS_NAMES = ["daisy","dandelion","rose","sunflower","tulip"] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SEGMENTS = 50       # number of superpixels
NSAMPLES_SHAP = 1024  # number of Kernel SHAP samples (increase for accuracy)
BATCH_SIZE = 32       # model batch size for mask predictions
USE_LOGITS = True     # explain logits (recommended) or probabilities (False)

# -------------------------
# Helpers: model & preprocessing
# -------------------------
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def load_finetuned_vgg16(checkpoint_path, num_classes=3, device=DEVICE):
    model = models.vgg16(pretrained=False)
    in_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(in_features, num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "state_dict" in ckpt):
        state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    else:
        state = ckpt
    new_state = {}
    for k,v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=True)
    model.to(device).eval()
    return model

def model_predict_batch_from_uint8_images(model, images_uint8, return_logits=True):
    """
    images_uint8: np.array (N,H,W,3), dtype uint8 (0..255)
    returns: np.array (N, C) logits (if return_logits) else probs
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, images_uint8.shape[0], BATCH_SIZE):
            batch = images_uint8[i:i+BATCH_SIZE]
            tensors = []
            for img in batch:
                pil = Image.fromarray(img)
                tensors.append(preprocess(pil))
            xb = torch.stack(tensors).to(DEVICE)
            logits = model(xb)
            if return_logits:
                preds.append(logits.cpu().numpy())
            else:
                preds.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(preds)

# -------------------------
# Superpixel segmentation & masking utilities
# -------------------------
def segment_image(pil_image, n_segments=N_SEGMENTS, compactness=10):
    """
    Returns:
      segments: HxW int map (labels 0..M-1)
      image_float: HxWx3 float in [0,1]
    """
    img_resized = pil_image.resize((224,224))
    image = np.array(img_resized).astype(float) / 255.0
    segments = slic(img_as_float(image), n_segments=n_segments, compactness=compactness, start_label=0)
    return segments, image

def mask_image_with_superpixels(image_float, segments, mask_bool, background_color=None):
    """
    image_float: HxWx3 float [0,1]
    mask_bool: length M boolean array. True => KEEP that superpixel; False => replace by background
    background_color: 3-vector float [0,1] or None -> use mean color
    returns uint8 image (H,W,3) in 0..255 (uint8)
    """
    H,W,_ = image_float.shape
    out = image_float.copy()
    if background_color is None:
        bg = image_float.mean(axis=(0,1))
    else:
        bg = np.array(background_color).reshape(1,1,3)
    M = mask_bool.shape[0]
    for s in range(M):
        if not mask_bool[int(s)]:
            out[segments == s] = bg
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)

# -------------------------
# Build a predict function that maps mask vectors -> model outputs
# -------------------------
def make_mask_predict_fn(model, image_float, segments, background_color=None, return_logits=USE_LOGITS):
    """
    Returns a function f(masks_array) -> np.array (N, C)
    masks_array: shape (N, M) (0/1 or floats between 0/1)
    """
    M = segments.max() + 1
    if background_color is None:
        background_color = image_float.mean(axis=(0,1))

    def f(masks_array):
        masks_array = np.atleast_2d(masks_array)
        # ensure shape (N, M)
        imgs = np.zeros((masks_array.shape[0], 224, 224, 3), dtype=np.uint8)
        for i in range(masks_array.shape[0]):
            mask_bool = masks_array[i].astype(bool)
            imgs[i] = mask_image_with_superpixels(image_float, segments, mask_bool, background_color)
        preds = model_predict_batch_from_uint8_images(model, imgs, return_logits=return_logits)
        return preds
    return f

# -------------------------
# Main: run Kernel SHAP on the full-image mask
# -------------------------
def explain_with_kernel_shap(model, pil_image, n_segments=N_SEGMENTS, nsamples=NSAMPLES_SHAP, return_logits=USE_LOGITS):
    segments, image_float = segment_image(pil_image, n_segments=n_segments)
    M = segments.max() + 1
    print(f"[info] {M} superpixels (features).")

    # create predict function
    predict_fn = make_mask_predict_fn(model, image_float, segments, background_color=None, return_logits=return_logits)

    # KernelExplainer background: use baseline mask = all zeros (no superpixels) as representative
    background_masks = np.zeros((1, M))  # one baseline
    explainer = shap.KernelExplainer(predict_fn, background_masks)

    # full image mask (all ones)
    full_mask = np.ones((1, M))

    print(f"[info] Running KernelExplainer.shap_values for nsamples={nsamples} ... this may take time.")
    t0 = time.time()
    shap_values = explainer.shap_values(full_mask, nsamples=nsamples)
    t1 = time.time()
    print(f"[info] SHAP computed in {t1-t0:.1f}s")

    # shap_values format:
    # - For classification with C outputs: shap_values is a list of length C, each element shape (1, M)
    # - For single-output: shap_values is array (1, M)
    return {
        "shap_values": shap_values,
        "segments": segments,
        "image_float": image_float
    }

# -------------------------
# Visualization: map shap values to pixel heatmap and overlay
# -------------------------
def shap_values_to_heatmap(shap_vals_for_class, segments, image_float, normalize=True):
    H,W,_ = image_float.shape
    heat = np.zeros((H,W), dtype=float)
    for i,val in enumerate(shap_vals_for_class.ravel()):
        heat[segments == i] = val
    if normalize:
        m = np.max(np.abs(heat)) + 1e-12
        heat = heat / m
    return heat

def plot_overlay(pil_image, heatmap, alpha=0.6, cmap='bwr', title=None):
    plt.figure(figsize=(6,6))
    img = np.array(pil_image.resize((224,224))).astype(float) / 255.0
    plt.imshow(img)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=-1, vmax=1)
    plt.colorbar(label='Normalized SHAP')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    pil = Image.open(IMAGE_PATH).convert("RGB")
    model = load_finetuned_vgg16(CHECKPOINT_PATH, num_classes=NUM_CLASSES, device=DEVICE)

    # determine model's prediction on original image
    orig_uint8 = np.array(pil.resize((224,224))).astype(np.uint8)[np.newaxis,...]
    pred = model_predict_batch_from_uint8_images(model, orig_uint8, return_logits=USE_LOGITS)[0]
    if USE_LOGITS:
        # convert logits to probs to show predicted class
        probs = F.softmax(torch.from_numpy(pred), dim=0).numpy()
    else:
        probs = pred
    pred_class = int(np.argmax(probs))
    print(f"[info] Model predicted class {pred_class} ({CLASS_NAMES[pred_class]}) with prob {probs[pred_class]:.4f}")

    # run Kernel SHAP
    res = explain_with_kernel_shap(model, pil, n_segments=N_SEGMENTS, nsamples=NSAMPLES_SHAP, return_logits=USE_LOGITS)
    shap_vals = res["shap_values"]
    segments = res["segments"]
    image_float = res["image_float"]

    # select shap values for predicted class (if multioutput)
    if isinstance(shap_vals, list) and len(shap_vals) == NUM_CLASSES:
        sv_class = shap_vals[pred_class]  # shape (1, M)
    else:
        # single-output array (1, M)
        sv_class = np.array(shap_vals).reshape(1,-1)

    heat = shap_values_to_heatmap(sv_class, segments, image_float, normalize=True)
    plot_overlay(pil, heat, alpha=0.6, title=f"SHAP for class {CLASS_NAMES[pred_class]}")
    # Optionally print top superpixels
    seg_vals = sv_class.ravel()
    topk = np.argsort(seg_vals)[-8:][::-1]
    print("Top superpixels (id, shap):", [(int(i), float(seg_vals[i])) for i in topk])