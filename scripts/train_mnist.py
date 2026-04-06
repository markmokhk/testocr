"""
MNIST 手寫數字分類訓練腳本（小型 CNN）。
使用專案內 data/mnist 資料，並可合併 my_digits/ 內自訂圖片；權重可另存至 checkpoints/。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

# -----------------------------------------------------------------------------
# 1. 專案路徑：資料放在倉庫根目錄下的 data/mnist（與 download_mnist.py 一致）
#    my_digits/：檔名 my_<0-9>.png（或 .jpg）表示該張圖片的標籤，例如 my_7.png → 7
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "mnist"
MY_DIGITS_ROOT = PROJECT_ROOT / "my_digits"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "mnist_cnn.pt"

_MY_DIGIT_NAME = re.compile(r"^my_(\d+)\.(png|jpe?g)$", re.IGNORECASE)


class MyDigitsDataset(Dataset):
    """從 my_digits/ 讀取圖片；檔名必須為 my_<標籤>.png（標籤 0–9）。"""

    def __init__(self, root: Path, transform) -> None:
        self.root = root
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        if root.is_dir():
            for p in sorted(root.iterdir()):
                if not p.is_file():
                    continue
                m = _MY_DIGIT_NAME.match(p.name)
                if not m:
                    continue
                label = int(m.group(1))
                if 0 <= label <= 9:
                    self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class MnistCNN(nn.Module):
    """小型卷積網路：兩層卷積＋池化，再接全連接層輸出 10 類。"""

    def __init__(self) -> None:
        super().__init__()
        # 特徵抽取：1 通道灰階 → 32 → 64 通道；每經一次 MaxPool 邊長約减半
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 28→14→7，故展平後維度為 64 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MNIST CNN")
    p.add_argument("--epochs", type=int, default=3, help="訓練輪數")
    p.add_argument("--batch-size", type=int, default=64, help="批次大小")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam 學習率")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -------------------------------------------------------------------------
    # 2. 超參數與裝置：有 GPU 則用 CUDA，否則 CPU（MNIST 在 CPU 也可訓練）
    # -------------------------------------------------------------------------
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device}")
    MY_DIGITS_ROOT.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 3. 資料前處理：MNIST 已為 28×28 灰階；my_digits 需先縮放與灰階，再與訓練相同正規化
    # -------------------------------------------------------------------------
    mnist_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    my_digits_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # -------------------------------------------------------------------------
    # 4. 資料集與 DataLoader：MNIST 訓練集 + my_digits/（ConcatDataset）；Windows 下 num_workers=0
    # -------------------------------------------------------------------------
    train_set: Dataset | ConcatDataset = MNIST(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=mnist_transform,
    )
    my_extra = MyDigitsDataset(MY_DIGITS_ROOT, my_digits_transform)
    if len(my_extra) > 0:
        train_set = ConcatDataset([train_set, my_extra])
        print(f"已合併 my_digits/：{len(my_extra)} 張（my_<0-9>.png）")
    else:
        print("my_digits/ 無符合 my_<數字>.png/.jpg 的檔案，僅使用 MNIST 訓練集")
    test_set = MNIST(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=mnist_transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # -------------------------------------------------------------------------
    # 5. 模型、損失函數、優化器：10 類分類用 CrossEntropy；Adam 做參數更新
    # -------------------------------------------------------------------------
    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------------------------------------------------------------------
    # 6. 訓練迴圈：每個 epoch 內對每個 batch 前向、算 loss、反向傳播、更新權重
    # -------------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        avg_loss = running_loss / total_train
        train_acc = correct_train / total_train
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss: {avg_loss:.4f} | train acc: {train_acc:.4f}"
        )

        # ---------------------------------------------------------------------
        # 7. 評估：關閉梯度，在測試集上計算準確率
        # ---------------------------------------------------------------------
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total
        print(f"         test acc: {test_acc:.4f}")

    # -------------------------------------------------------------------------
    # 8. 儲存權重：寫入 checkpoints/（此目錄已列於 .gitignore，需自行備份若要上傳）
    # -------------------------------------------------------------------------
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
        CHECKPOINT_PATH,
    )
    print(f"已儲存檢查點: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
