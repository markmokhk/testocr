from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "preview_5.png"
OUTPUT = ROOT / "preview_5_chords_red.png"

# These narrow horizontal bands correspond to the chord-only rows visible on
# this specific preview page.
ROWS = [
    (145, 171),
    (300, 322),
    (426, 446),
    (548, 571),
    (691, 714),
    (830, 853),
    (953, 974),
]


def find_clusters(image: Image.Image, top: int, bottom: int) -> list[tuple[int, int]]:
    clusters: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for x in range(70, min(image.width, 850)):
        dark = 0
        for y in range(top, bottom):
            r, g, b = image.getpixel((x, y))
            if r < 185 and g < 185 and b < 185:
                dark += 1
        if dark >= 2 and not in_run:
            start = x
            in_run = True
        elif dark < 2 and in_run:
            clusters.append((start, x - 1))
            in_run = False
    if in_run:
        clusters.append((start, min(image.width, 850) - 1))
    return clusters


def recolor_rows(image: Image.Image) -> None:
    px = image.load()
    for top, bottom in ROWS:
        for left, right in find_clusters(image, top, bottom):
            for y in range(max(0, top), min(image.height, bottom)):
                for x in range(max(0, left), min(image.width, right + 1)):
                    r, g, b = px[x, y]
                    # Recolor dark printed ink while preserving anti-aliasing.
                    if r < 185 and g < 185 and b < 185:
                        darkness = 255 - max(r, g, b)
                        px[x, y] = (max(175, 150 + darkness), 25, 25)


def main() -> None:
    image = Image.open(INPUT).convert("RGB")
    recolor_rows(image)
    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
