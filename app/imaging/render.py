import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.imaging.numbering import Region

PALETTE_ROW_HEIGHT = 60
SWATCH_SIZE = 60


def render_preview_image(
    region_map: np.ndarray, region_labels: np.ndarray, palette: np.ndarray
) -> Image.Image:
    # 색 라벨맵이 아니라 영역맵에서 칠한다. 라벨맵에는 병합으로 사라진 잔점이 그대로
    # 남아 있어서, 그걸로 미리보기를 만들면 윤곽선도 번호도 없는 색 얼룩이 보인다.
    # 즉 사용자가 번호대로 칠한 결과와 미리보기가 서로 다른 그림이 된다.
    return Image.fromarray(palette[region_labels[region_map]], mode="RGB")


def render_outline_image(
    shape: tuple[int, int],
    seams: list[np.ndarray],
    regions: list[Region],
    palette: np.ndarray,
) -> Image.Image:
    h, w = shape
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for seam in seams:
        if len(seam) >= 2:
            draw.line([tuple(int(v) for v in p) for p in seam], fill="black", width=1)

    for region in regions:
        text = str(region.number)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # 영역 안에 글자가 못 들어가면 이웃 영역 위로 삐져나와 어느 쪽 번호인지
        # 알 수 없게 된다. 그런 영역은 번호를 생략한다.
        if math.hypot(tw, th) / 2 > region.label_radius:
            continue
        cx, cy = region.label_anchor
        draw.text((cx - tw / 2, cy - th / 2), text, fill="black", font=font)

    bar = _render_palette_bar(len(palette), palette, w)
    composed = Image.new("RGB", (w, h + bar.height), "white")
    composed.paste(bar, (0, 0))
    composed.paste(canvas, (0, bar.height))
    return composed


def _render_palette_bar(color_count: int, palette: np.ndarray, width: int) -> Image.Image:
    cols = max(1, min(color_count, width // SWATCH_SIZE))
    rows = (color_count + cols - 1) // cols
    bar_height = rows * PALETTE_ROW_HEIGHT

    bar = Image.new("RGB", (width, bar_height), "white")
    draw = ImageDraw.Draw(bar)
    font = ImageFont.load_default()

    for number in range(1, color_count + 1):
        idx = number - 1
        row, col = divmod(idx, cols)
        x0 = col * SWATCH_SIZE + 4
        y0 = row * PALETTE_ROW_HEIGHT + 4
        x1, y1 = x0 + SWATCH_SIZE - 12, y0 + SWATCH_SIZE - 24
        color = tuple(int(c) for c in palette[idx])
        draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")
        draw.text((x0, y1 + 2), str(number), fill="black", font=font)

    return bar
