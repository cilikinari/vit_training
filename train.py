import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter
import os

from dataset import get_dataloader
from model import VisionTransformer


config = {
    "batch_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "img_size": 224,
    "patch_size": 16,
    "attention_heads": 4,
    "embed_dim": 128,
    "transformer_blocks": 4,
    "mlp_nodes": 128,
    "learning_rate": 0.001,  
    "epochs": 10,               
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ===== LOAD DATA =====
train_loader, val_loader = get_dataloader(config["batch_size"])


# ===== AUTO CLASS WEIGHT =====
targets = train_loader.dataset.targets
class_counts = Counter(targets)

counts = [class_counts[i] for i in range(len(class_counts))]

weights = 1.0 / torch.tensor(counts, dtype=torch.float)
weights = weights / weights.sum()

print("Class counts:", counts)
print("Class weights:", weights)


# ===== MODEL =====
model = VisionTransformer(config).to(device)

if os.path.exists("vit_last.pth"):
    model.load_state_dict(torch.load("vit_last.pth"))
    print("🔥 Lanjut dari model sebelumnya!")

criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"]
)


# ===== METRICS =====
train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float('inf')    


# ===== TRAINING LOOP =====
for epoch in range(config["epochs"]):
    print(f"\nEpoch [{epoch+1}/{config['epochs']}]")

    # ===== TRAIN =====
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_acc = 100 * correct_train / total_train
    avg_train_loss = total_train_loss / len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_acc = 100 * correct_val / total_val
    avg_val_loss = total_val_loss / len(val_loader)

    scheduler.step()

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # 🔥 SAVE BEST MODEL
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "vit_best.pth")
        print("🔥 Best model updated!")

    # save last model
    torch.save(model.state_dict(), "vit_last.pth")


# ===== PLOT =====
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.show()

print("Training selesai!")