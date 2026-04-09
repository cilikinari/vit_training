import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloader
from model import VisionTransformer

config = {
	"batch_size": 32,
	"num_classes": 3,
	"num_channels": 3,
	"img_size": 224,
	"patch_size": 16,
	"patch_num": (224 // 16) * (224 // 16),
	"attention_heads": 4,
	"embed_dim": 128,
	"transformer_blocks": 4,
	"mlp_nodes": 128,
	"learning_rate": 0.0001,
	"epochs": 10,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(config).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])
criterion = nn.CrossEntropyLoss()

train_data, val_data = get_dataloader(config["batch_size"])

for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    correct_epoch = 0
    total_epoch = 0

    print(f"\nEpoch {epoch+1}")

    for batch_idx, (images, labels) in enumerate(train_data):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()

        correct_epoch += correct
        total_epoch += labels.size(0)

        if batch_idx % 100 == 0:
            acc = 100.0 * correct / labels.size(0)
            print(f"Batch {batch_idx}: Loss={loss.item():.4f}, Acc={acc:.2f}%")

    epoch_acc = 100.0 * correct_epoch / total_epoch
    print(f"Epoch {epoch+1} done | Loss={total_loss:.4f} | Acc={epoch_acc:.2f}%")