import re
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "scores_samples.pdf"
INPUT_IMAGE = ROOT / "preview_5.png"
OUTPUT_IMAGE = ROOT / "preview_5_chords_red_pymupdf.png"
PAGE_INDEX = 4  # page 5

# Chord-like tokens after OCR normalization.
CHORD_RE = re.compile(
    r"^[A-G](?:[#b])?(?:(?:maj|min|m|dim|aug|sus|add)?\d*)?(?:/[A-G](?:[#b])?)?$"
)


def normalize_token(text: str) -> str:
    text = text.strip()
    translation = str.maketrans(
        {
            "／": "/",
            "∕": "/",
            "⁄": "/",
            "﹟": "#",
            "＃": "#",
            "♯": "#",
            "♭": "b",
            "﹣": "-",
            "—": "-",
            "–": "-",
            "（": "(",
            "）": ")",
            "【": "",
            "】": "",
            "「": "",
            "」": "",
            "『": "",
            "』": "",
            "﹙": "(",
            "﹚": ")",
            "︵": "(",
            "︶": ")",
        }
    )
    text = text.translate(translation)
    text = text.replace(" ", "")
    text = text.replace("ln", "m")
    text = text.replace("I", "1")
    text = text.replace("l", "")
    text = text.replace("╴", "m")
    text = text.replace("Ⅱ", "m")
    text = text.replace("Ⅲ", "m")
    text = text.replace("叱", "7")
    text = text.replace("圳", "7")
    text = text.replace("叫", "7")
    text = text.replace("朋", "m")
    text = text.replace("阱", "F#")
    text = text.replace("司", "F")
    text = text.replace("酣", "F#")
    text = re.sub(r"[^A-Ga-z0-9#/()]", "", text)
    return text


def is_chord(text: str) -> bool:
    if not text or ":" in text:
        return False
    if text in {"A", "B", "C", "D", "E", "F", "G"}:
        return True
    return bool(CHORD_RE.fullmatch(text))


def chord_boxes(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    words = sorted(page.get_text("words"), key=lambda w: (round(w[1], 1), w[0]))
    boxes: list[tuple[float, float, float, float]] = []
    used: set[int] = set()

    for i, word in enumerate(words):
        if i in used:
            continue

        x0, y0, x1, y1, text = word[:5]
        norm = normalize_token(text)

        if is_chord(norm):
            boxes.append((x0, y0, x1, y1))
            used.add(i)
            continue

        # Merge 2-3 nearby OCR fragments on the same line, e.g. "B" + "m╴".
        parts = [word]
        combined_text = text
        last_x1 = x1
        max_y_delta = 3.5
        max_gap = 18.0

        for j in range(i + 1, min(i + 3, len(words))):
            nx0, ny0, nx1, ny1, ntext = words[j][:5]
            if abs(ny0 - y0) > max_y_delta or nx0 - last_x1 > max_gap:
                break
            parts.append(words[j])
            combined_text += ntext
            last_x1 = nx1

            norm = normalize_token(combined_text)
            if is_chord(norm):
                boxes.append((parts[0][0], min(p[1] for p in parts), parts[-1][2], max(p[3] for p in parts)))
                used.update(range(i, j + 1))
                break

    return boxes


def recolor_box(image: Image.Image, box: tuple[int, int, int, int]) -> None:
    px = image.load()
    left, top, right, bottom = box
    for y in range(max(0, top), min(image.height, bottom)):
        for x in range(max(0, left), min(image.width, right)):
            r, g, b = px[x, y]
            if r < 185 and g < 185 and b < 185:
                darkness = 255 - max(r, g, b)
                px[x, y] = (max(175, 150 + darkness), 25, 25)


def main() -> None:
    doc = fitz.open(PDF_PATH)
    page = doc[PAGE_INDEX]
    image = Image.open(INPUT_IMAGE).convert("RGB")

    scale_x = image.width / page.rect.width
    scale_y = image.height / page.rect.height

    for x0, y0, x1, y1 in chord_boxes(page):
        # Slight padding to catch anti-aliased edges.
        box = (
            int(x0 * scale_x) - 2,
            int(y0 * scale_y) - 2,
            int(x1 * scale_x) + 2,
            int(y1 * scale_y) + 2,
        )
        recolor_box(image, box)

    image.save(OUTPUT_IMAGE)
    print(OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
