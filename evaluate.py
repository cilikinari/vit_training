import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Import fungsi dan class dari file kamu sendiri
from model import VisionTransformer
from dataset import get_dataloader

# ==========================================
# 1. KONFIGURASI (Sama Persis dengan train.py)
# ==========================================
config = {
    "batch_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "img_size": 224,
    "patch_size": 16,
    "attention_heads": 4,       # Diambil dari train.py
    "embed_dim": 128,
    "transformer_blocks": 4,    # Diambil dari train.py
    "mlp_nodes": 128,           # Diambil dari train.py
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'vit_best.pth' # Pakai model dengan loss terendah

# Sesuai urutan abjad folder kamu
CLASS_NAMES = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro'
]

def run_evaluation():
    print(f"Mengevaluasi model di {DEVICE}...")

    # ==========================================
    # 2. SIAPKAN DATA & MODEL
    # ==========================================
    # Panggil dataloader (kita cuma butuh val_loader-nya aja, train_loader di-ignore pakai '_')
    _, val_loader = get_dataloader(config["batch_size"])

    # Buat raga model
    model = VisionTransformer(config).to(DEVICE)

    # Suntikkan otak model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ==========================================
    # 3. PROSES EVALUASI
    # ==========================================
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ==========================================
    # 4. CETAK HASIL & GRAFIK
    # ==========================================
    print("\n" + "="*50)
    print("HASIL EVALUASI DATA VALIDATION")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Bikin Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10)) # Dibesarin dikit karena ada 10 kelas
    
    # Pakai warna YlGnBu (Kuning-Hijau-Biru) biar enak dilihat
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    
    plt.title('Confusion Matrix ViT - Hama & Penyakit Padi')
    plt.ylabel('Penyakit Asli')
    plt.xlabel('Prediksi Model')
    
    # Rotasi teks x-axis biar nggak nabrak karena nama kelasnya panjang
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('hasil_evaluasi_vit.png', dpi=300)
    print("\nGrafik berhasil disimpan sebagai 'hasil_evaluasi_vit.png' ")
    plt.show()

if __name__ == "__main__":
    run_evaluation()