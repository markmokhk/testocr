"""Export 30 MNIST digit images (3 per class 0-9) into samples/."""
from pathlib import Path

from torchvision.datasets import MNIST

PROJECT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT / "data" / "mnist"
OUT = PROJECT / "samples"
OUT.mkdir(parents=True, exist_ok=True)

ds = MNIST(root=str(DATA_ROOT), train=True, download=True)
targets = ds.targets

for d in range(10):
    idxs = (targets == d).nonzero(as_tuple=True)[0][:3].tolist()
    if len(idxs) < 3:
        raise RuntimeError(f"Not enough samples for digit {d}")
    for k, idx in enumerate(idxs, start=1):
        img, _ = ds[idx]
        out_path = OUT / f"mnist_digit{d}_{k}.png"
        img.save(out_path)
        print(out_path.name)

print(f"Done. {OUT}")
