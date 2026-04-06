from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "preview_10.png"
OUTPUT = ROOT / "preview_10_chords_red.png"

# Manually placed chord regions for preview_10.png. These boxes target the
# handwritten chord symbols above each notation line and avoid the lyric text.
BOXES = [
    (388, 128, 424, 170),
    (435, 128, 474, 170),
    (484, 128, 523, 170),
    (531, 126, 571, 170),
    (780, 126, 824, 170),
    (358, 246, 380, 285),
    (474, 246, 502, 285),
    (589, 246, 612, 285),
    (709, 244, 742, 285),
    (121, 366, 134, 410),
    (166, 364, 194, 410),
    (274, 364, 300, 410),
    (392, 362, 418, 410),
    (474, 364, 500, 410),
    (592, 364, 616, 410),
    (708, 362, 734, 410),
    (166, 571, 193, 617),
    (296, 571, 322, 617),
    (426, 571, 463, 617),
    (556, 571, 582, 617),
    (710, 571, 735, 617),
    (171, 662, 186, 706),
    (206, 662, 222, 706),
    (242, 662, 258, 706),
    (288, 662, 304, 706),
    (407, 662, 422, 706),
    (442, 662, 458, 706),
    (477, 662, 492, 706),
    (526, 662, 575, 706),
    (596, 662, 610, 706),
    (679, 662, 692, 706),
    (713, 662, 727, 706),
    (740, 662, 774, 706),
]


def recolor_box(image: Image.Image, bbox: tuple[int, int, int, int]) -> None:
    px = image.load()
    left, top, right, bottom = bbox
    for y in range(max(0, top), min(image.height, bottom)):
        for x in range(max(0, left), min(image.width, right)):
            r, g, b = px[x, y]
            if r < 185 and g < 185 and b < 185:
                darkness = 255 - max(r, g, b)
                px[x, y] = (max(175, 150 + darkness), 25, 25)


def main() -> None:
    image = Image.open(INPUT).convert("RGB")
    for bbox in BOXES:
        recolor_box(image, bbox)
    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
