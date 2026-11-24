#!/usr/bin/env python3
"""Draw the OCSID architecture figure (PNG) with Pillow.

The figure separates the learned prediction stage (blue) from the zero-training
structural attribution stage (green).  English labels are used so the figure can
go into an English paper without font substitution.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 2400, 1600
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

INK = (35, 41, 54)
MUTED = (95, 105, 120)
BLUE_FILL = (226, 238, 252)
BLUE_LINE = (47, 85, 151)
GREEN_FILL = (226, 242, 219)
GREEN_LINE = (84, 130, 53)
YELLOW_FILL = (255, 242, 204)
YELLOW_LINE = (191, 144, 0)
INPUT_FILL = (241, 244, 248)
INPUT_LINE = (120, 134, 153)
PANEL_BLUE = (246, 250, 255)
PANEL_GREEN = (246, 251, 243)


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def box(draw, x, y, w, h, lines, fill, outline, size=26, bold=False,
        radius=16, line_width=3):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill,
                           outline=outline, width=line_width)
    f = font(size, bold)
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, font=f, spacing=6, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text((x + (w - tw) / 2, y + (h - th) / 2),
                        text, font=f, fill=INK, spacing=6, align="center")


def arrow(draw, x1, y1, x2, y2, color=MUTED, width=4, dashed=False):
    if dashed:
        # simple dashed line
        import math
        total = math.hypot(x2 - x1, y2 - y1)
        steps = max(int(total / 14), 1)
        for i in range(0, steps, 2):
            a = i / steps
            b = min((i + 1) / steps, 1)
            draw.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a,
                       x1 + (x2 - x1) * b, y1 + (y2 - y1) * b),
                      fill=color, width=width)
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
    # arrowhead
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 16
    draw.polygon([
        (x2, y2),
        (x2 - size * math.cos(angle - 0.4), y2 - size * math.sin(angle - 0.4)),
        (x2 - size * math.cos(angle + 0.4), y2 - size * math.sin(angle + 0.4)),
    ], fill=color)


def label(draw, x, y, text, size=22, color=MUTED, bold=False):
    draw.text((x, y), text, font=font(size, bold), fill=color)


def main() -> int:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    # Title
    label(draw, 60, 30, "OCSID: Operation-Conditioned Structural Counterfactual Attribution",
          size=42, color=INK, bold=True)
    label(draw, 60, 90,
          "Stage 1 learns to predict the next-minute failure; Stage 2 attributes it "
          "with a training-free structural counterfactual.",
          size=24, color=MUTED)

    # Panels: Stage 1 on top, Stage 2 below.
    draw.rounded_rectangle((520, 140, 1760, 660), radius=24, fill=PANEL_BLUE,
                           outline=(205, 220, 240), width=2)
    label(draw, 545, 160, "STAGE 1 — PREDICTION   (learned parameters)", size=24,
          color=BLUE_LINE, bold=True)
    draw.rounded_rectangle((520, 710, 1760, 1070), radius=24, fill=PANEL_GREEN,
                           outline=(205, 226, 195), width=2)
    label(draw, 545, 730, "STAGE 2 — ATTRIBUTION   (zero learned parameters)", size=24,
          color=GREEN_LINE, bold=True)

    # Inputs
    box(draw, 60, 170, 400, 120, ["Operation features  O", "[T x M x F_o]"],
        INPUT_FILL, INPUT_LINE, size=26)
    box(draw, 60, 330, 400, 120, ["Call graph  A", "[T x N x N]"],
        INPUT_FILL, INPUT_LINE, size=26)
    box(draw, 60, 490, 400, 120, ["Panel features  X", "[T x N x F]"],
        INPUT_FILL, INPUT_LINE, size=26)
    box(draw, 60, 850, 400, 140, ["Raw traces", "request span DAGs"],
        INPUT_FILL, INPUT_LINE, size=26)
    label(draw, 60, 1000, "only completed spans", size=20, color=MUTED)

    # Stage 1 boxes
    box(draw, 560, 210, 300, 130, ["Operation Encoder", "(shared GRU)"], BLUE_FILL, BLUE_LINE)
    box(draw, 930, 210, 300, 130, ["Operation Risk", "q_o  ->  p_op"], BLUE_FILL, BLUE_LINE)
    box(draw, 1300, 270, 320, 150,
        ["Fusion", "p = sigma( w1 logit p_sys", "+ w2 logit p_op )"],
        BLUE_FILL, BLUE_LINE, size=22)
    box(draw, 560, 485, 300, 130, ["Service Temporal", "Encoder"], BLUE_FILL, BLUE_LINE)
    box(draw, 930, 485, 300, 130, ["Graph Attention", "(2-layer GAT)"], BLUE_FILL, BLUE_LINE)
    box(draw, 1300, 485, 320, 130, ["System Risk Head", "p_sys"], BLUE_FILL, BLUE_LINE)

    # Stage 2 boxes
    box(draw, 560, 850, 280, 130, ["Span Tree", "Decomposition"], GREEN_FILL, GREEN_LINE)
    box(draw, 900, 850, 260, 130, ["Operation", "Selection"], GREEN_FILL, GREEN_LINE)
    box(draw, 1220, 850, 380, 130,
        ["Cost Attribution", "+ Counterfactual Replay"], GREEN_FILL, GREEN_LINE, size=24)
    label(draw, 560, 1000, "local / pre-gap / post-gap", size=20, color=MUTED)
    label(draw, 900, 1000, "riskiest entry operation", size=20, color=MUTED)
    label(draw, 1220, 1000, "normal reference: first 10 min", size=20, color=MUTED)

    # Outputs
    box(draw, 1830, 430, 440, 150, ["Next-minute failure", "probability  p"],
        YELLOW_FILL, YELLOW_LINE, size=26)
    box(draw, 1830, 850, 440, 150, ["Root-cause ranking", "r_i  (removable entry latency)"],
        YELLOW_FILL, YELLOW_LINE, size=24)

    # Stage 1 arrows
    arrow(draw, 460, 230, 560, 275)
    arrow(draw, 860, 275, 930, 275)
    arrow(draw, 1230, 275, 1300, 320)
    arrow(draw, 460, 550, 560, 550)
    arrow(draw, 460, 390, 975, 480)
    arrow(draw, 860, 550, 930, 550)
    arrow(draw, 1230, 550, 1300, 550)
    arrow(draw, 1460, 485, 1460, 420)
    arrow(draw, 1620, 345, 1830, 505)

    # Stage 2 arrows
    arrow(draw, 460, 920, 560, 915)
    arrow(draw, 840, 915, 900, 915)
    arrow(draw, 1160, 915, 1220, 915)
    arrow(draw, 1600, 915, 1830, 920)

    # legend
    lx, ly = 60, 1120
    draw.rounded_rectangle((lx, ly, lx + 34, ly + 24), radius=6, fill=BLUE_FILL,
                           outline=BLUE_LINE, width=2)
    label(draw, lx + 46, ly - 2, "learnable", size=22)
    draw.rounded_rectangle((lx + 220, ly, lx + 254, ly + 24), radius=6, fill=GREEN_FILL,
                           outline=GREEN_LINE, width=2)
    label(draw, lx + 266, ly - 2, "zero learned parameters", size=22)
    draw.rounded_rectangle((lx + 640, ly, lx + 674, ly + 24), radius=6, fill=YELLOW_FILL,
                           outline=YELLOW_LINE, width=2)
    label(draw, lx + 686, ly - 2, "output", size=22)

    # Crop the unused bottom margin.
    img = img.crop((0, 0, W, 1170))
    path = OUT / "ocsid_architecture.png"
    img.save(path, dpi=(200, 200))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
