#!/usr/bin/env python3
"""Draw the one-request CPCA example (Figure 2)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
INK = (35, 41, 54)
MUTED = (95, 105, 120)
BLUE = (226, 238, 252)
BLUE_L = (47, 85, 151)
GREEN = (226, 242, 219)
GREEN_L = (84, 130, 53)
YELLOW = (255, 242, 204)
YELLOW_L = (191, 144, 0)
GRAY = (241, 244, 248)
GRAY_L = (120, 134, 153)


def f(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def box(draw, x, y, w, h, lines, fill, line, size=24, bold=False):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=fill,
                           outline=line, width=3)
    ft = f(size, bold)
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, font=ft, spacing=6, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text((x + (w - tw) / 2, y + (h - th) / 2), text, font=ft,
                        fill=INK, spacing=6, align="center")


def arrow(draw, x1, y1, x2, y2, color=MUTED, width=4):
    import math
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    a = math.atan2(y2 - y1, x2 - x1)
    s = 15
    draw.polygon([(x2, y2),
                  (x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4)),
                  (x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4))], fill=color)


def label(draw, x, y, text, size=22, color=MUTED, bold=False):
    draw.text((x, y), text, font=f(size, bold), fill=color)


def main() -> int:
    img = Image.new("RGB", (2000, 1000), "white")
    d = ImageDraw.Draw(img)
    label(d, 50, 25, "CPCA on one request: restore a service, replay the span tree",
          size=34, color=INK, bold=True)
    label(d, 50, 75, "frontend → checkout → payment; entry latency 1000 ms",
          size=24, color=MUTED)

    # Tree
    box(d, 250, 150, 520, 110, ["frontend    d = 1000 ms", "self = 200   local = 0"],
        BLUE, BLUE_L)
    box(d, 250, 360, 520, 110, ["checkout    d = 800 ms", "self = 180   local = 0"],
        BLUE, BLUE_L)
    box(d, 250, 570, 520, 110, ["payment    d = 620 ms", "local = 620   (normal 50)"],
        YELLOW, YELLOW_L)
    arrow(d, 510, 260, 510, 360)
    arrow(d, 510, 470, 510, 570)
    label(d, 530, 285, "pre 20 / post 180", size=20)
    label(d, 530, 495, "pre 10 / post 170", size=20)

    # Right panel: observed vs restored
    d.rounded_rectangle((900, 150, 1950, 880), radius=20, fill=(248, 250, 252),
                        outline=(215, 222, 232), width=2)
    label(d, 930, 170, "Restore payment to its normal reference", size=26,
          color=INK, bold=True)

    rows = [
        ("payment local", "620", "50"),
        ("checkout→payment pre / post", "10 / 170", "5 / 5"),
        ("checkout local", "0", "0"),
        ("frontend→checkout pre / post", "20 / 180", "20 / 180"),
        ("frontend local", "0", "0"),
    ]
    label(d, 930, 220, "quantity", size=22, bold=True)
    label(d, 1450, 220, "observed", size=22, bold=True)
    label(d, 1700, 220, "restored", size=22, bold=True)
    y = 260
    for name, obs, res in rows:
        label(d, 930, y, name, size=21)
        label(d, 1450, y, obs, size=21)
        label(d, 1700, y, res, size=21, color=GREEN_L, bold=True)
        d.line((930, y + 32, 1900, y + 32), fill=(225, 230, 238), width=1)
        y += 45

    # Replay result
    d.rounded_rectangle((930, 600, 1900, 850), radius=16, fill=GREEN,
                        outline=GREEN_L, width=3)
    label(d, 960, 620, "Replay:", size=24, color=GREEN_L, bold=True)
    label(d, 960, 665, "checkout = 5 + 50 + 5 + 0 = 60 ms", size=24, color=INK)
    label(d, 960, 705, "frontend = 20 + 60 + 180 = 260 ms", size=24, color=INK)
    label(d, 960, 745, "Δ = 1000 − 260 = 740 ms        r = 0.74", size=26,
          color=GREEN_L, bold=True)

    # Bottom note
    label(d, 250, 760, "Restore checkout or frontend → Δ = 0:", size=24,
          color=INK, bold=True)
    label(d, 250, 800, "they only wait for payment; their own cost is already normal.",
          size=23, color=MUTED)
    label(d, 250, 850, "So the ranking is payment first, although all three spans look slow.",
          size=23, color=MUTED)

    path = OUT / "span_example.png"
    img = img.crop((0, 0, 2000, 920))
    img.save(path, dpi=(200, 200))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
