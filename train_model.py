"""
train_model.py — Oil Spill Image Classifier
Uses transfer learning with EfficientNet-B0 (pretrained on ImageNet).

Dataset structure expected:
    train/
        oil_spill/    (satellite images with oil spills)
        no_oil_spill/ (clean water images)
    test/
        oil_spill/
        no_oil_spill/
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_DIR    = "train"
TEST_DIR     = "test"
MODEL_SAVE   = "oil_spill_model.pth"
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES  = 2

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for name, param in model.features.named_parameters():
        block_idx = name.split(".")[0]
        param.requires_grad = block_idx in {"5", "6", "7", "8"}
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


# ─────────────────────────────────────────────
# TRAINING & EVAL FUNCTIONS
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# MAIN — must be guarded on Windows
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Validate dataset folders exist
    for split, path in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}/' folder not found.\n"
                f"Expected structure:\n"
                f"  {path}/oil_spill/\n"
                f"  {path}/no_oil_spill/"
            )

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=val_transforms)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Test  samples : {len(test_dataset)}")
    print(f"Classes       : {train_dataset.classes}")

    if len(train_dataset.classes) != 2:
        raise ValueError(
            f"Expected 2 classes (oil_spill, no_oil_spill), found: {train_dataset.classes}\n"
            f"Make sure your '{TRAIN_DIR}/' folder contains exactly two subfolders:\n"
            f"  {TRAIN_DIR}/oil_spill/\n"
            f"  {TRAIN_DIR}/no_oil_spill/"
        )

    # Weighted sampler for class imbalance
    class_counts   = np.bincount(train_dataset.targets)
    class_weights  = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # num_workers=0 required on Windows
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False,  num_workers=0, pin_memory=False)

    model = build_model(NUM_CLASSES).to(DEVICE)

    weights_tensor = torch.tensor(
        [1.0 / c for c in class_counts], dtype=torch.float
    ).to(DEVICE)
    weights_tensor = weights_tensor / weights_tensor.sum()

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc   = 0.0
    best_state = copy.deepcopy(model.state_dict())

    print(f"\n{'Epoch':>5} {'Train Loss':>10} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8} {'LR':>10}")
    print("-" * 62)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc        = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_acc:
            best_acc   = vl_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch":       epoch,
                "model_state": best_state,
                "classes":     train_dataset.classes,
                "accuracy":    best_acc,
            }, MODEL_SAVE)

        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f"{epoch:>5} {tr_loss:>10.4f} {tr_acc:>9.2f}% "
              f"{vl_loss:>9.4f} {vl_acc:>7.2f}% {lr_now:>10.2e}  ({elapsed:.1f}s)")

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved -> {MODEL_SAVE}")

    # Final evaluation
    model.load_state_dict(best_state)
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion)

    print(f"\nTest Accuracy: {test_acc:.2f}%\n")
    print(classification_report(labels, preds, target_names=train_dataset.classes))

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"],   label="Val Loss",   linewidth=2)
    axes[0].set_title("Loss Curve", fontsize=14)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc", linewidth=2)
    axes[1].plot(history["val_acc"],   label="Val Acc",   linewidth=2)
    axes[1].set_title("Accuracy Curve", fontsize=14)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes, ax=axes[2])
    axes[2].set_title("Confusion Matrix", fontsize=14)
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.show()
    print("Training plots saved -> training_results.png")

    # ─────────────────────────────────────────────
# DOMAIN VALIDATOR (runs BEFORE EfficientNet)
# ─────────────────────────────────────────────
def is_ocean_image(image: Image.Image) -> tuple[bool, str]:
    """
    Pixel-level heuristic gate.
    Returns (is_valid, reject_reason)
    """
    img = np.array(image.resize((128, 128))).astype(float)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    # ── Rule 1: Water pixel detection
    # Water = blue-dominant OR very dark (oil/deep ocean)
    blue_dominant  = (b > r + 15) & (b > g + 5)
    dark_water     = (r < 55) & (g < 65) & (b < 80)
    water_mask     = blue_dominant | dark_water
    water_ratio    = water_mask.sum() / water_mask.size

    # ── Rule 2: Green dominance = land/vegetation
    green_dominant = (g > r + 20) & (g > b + 20)
    green_ratio    = green_dominant.sum() / green_dominant.size

    # ── Rule 3: Warm/sandy tones = land/desert/urban
    warm_mask      = (r > 140) & (r > b + 30)
    warm_ratio     = warm_mask.sum() / warm_mask.size

    # ── Rule 4: Very bright = clouds/snow/ice (not ocean surface)
    bright_mask    = (r > 210) & (g > 210) & (b > 210)
    bright_ratio   = bright_mask.sum() / bright_mask.size

    # ── Decision tree
    if water_ratio < 0.25:
        return False, "Insufficient water coverage detected (<25%)"
    if green_ratio > 0.30:
        return False, "High vegetation/land content detected"
    if warm_ratio > 0.35:
        return False, "Land, desert, or urban surface detected"
    if bright_ratio > 0.60:
        return False, "Image too bright — clouds, snow, or non-water surface"

    return True, "OK"


# ─────────────────────────────────────────────
# INFERENCE (with domain gate)
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 65.0

def predict(model, classes, image: Image.Image):
    # ── Gate 1: Domain check
    is_valid, reason = is_ocean_image(image)
    if not is_valid:
        return "rejected", 0.0, None, classes, reason

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    # ── Gate 2: Confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return "uncertain", confidence, probs, classes, "Model confidence too low for this image"

    return pred_label, confidence, probs, classes, "OK"


