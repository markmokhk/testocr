"""Download MNIST via torchvision into data/mnist (train + test)."""
from pathlib import Path

from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parent.parent / "data" / "mnist"
ROOT.mkdir(parents=True, exist_ok=True)

for train in (True, False):
    ds = MNIST(root=str(ROOT), train=train, download=True)
    split = "train" if train else "test"
    print(f"{split}: {len(ds)} samples")

print(f"Done. Root: {ROOT}")
