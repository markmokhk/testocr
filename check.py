"""
以訓練好的 MNIST CNN 預測單張 PNG 中的數字（0–9）。

使用方式（請先 conda activate testocr）：
  python check.py path/to/input.png

注意：模型在 28×28、類 MNIST 的單一數字上較可靠；一般照片或大圖會先被縮放成 28×28，結果可能不佳。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent


def _load_train_mnist_module():
    path = ROOT / "scripts" / "train_mnist.py"
    spec = importlib.util.spec_from_file_location("train_mnist", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description="Predict digit from PNG using trained MNIST CNN")
    p.add_argument("image", type=Path, help="輸入 PNG（或常見圖檔）路徑")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="自訂檢查點路徑（預設：checkpoints/mnist_cnn.pt）",
    )
    args = p.parse_args()

    tm = _load_train_mnist_module()
    MnistCNN = tm.MnistCNN
    checkpoint_path = args.checkpoint if args.checkpoint is not None else tm.CHECKPOINT_PATH

    if not checkpoint_path.is_file():
        print(f"找不到檢查點：{checkpoint_path}", file=sys.stderr)
        print("請先執行：python scripts/train_mnist.py", file=sys.stderr)
        sys.exit(1)

    if not args.image.is_file():
        print(f"找不到圖片：{args.image}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 與 train_mnist.py 相同的前處理；圖片轉灰階並縮放至 28×28
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    img = Image.open(args.image).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    model = MnistCNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(logits.argmax(dim=1).item())
        conf = float(probs[0, pred].item())

    print(f"預測數字: {pred}")
    print(f"信心度: {conf:.4f}")


if __name__ == "__main__":
    main()
