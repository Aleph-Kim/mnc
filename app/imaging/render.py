import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

from app.imaging.numbering import Region
from app.imaging.quantize import to_lab, _nearest

PALETTE_ROW_HEIGHT = 60
SWATCH_SIZE = 60


def render_preview_image(
    region_map: np.ndarray,
    region_labels: np.ndarray,
    palette: np.ndarray,
    line_mask: np.ndarray | None = None,
    source_rgb: np.ndarray | None = None,
) -> Image.Image:
    # 색 라벨맵이 아니라 영역맵에서 칠한다. 라벨맵에는 병합으로 사라진 잔점이 그대로
    # 남아 있어서, 그걸로 미리보기를 만들면 윤곽선도 번호도 없는 색 얼룩이 보인다.
    # 즉 사용자가 번호대로 칠한 결과와 미리보기가 서로 다른 그림이 된다.
    image = palette[region_labels[region_map]]
    if line_mask is not None and line_mask.any():
        # 인쇄되는 원화 선을 얹는다 — 사용자가 칠하는 대상은 면뿐
        image = image.copy()
        if source_rgb is None:
            image[line_mask] = (0, 0, 0)
        else:
            # Use the same solid inks as the paint palette. Copying JPEG pixels
            # left visible patches where a stroke entered a larger filled area.
            ink_labels = _nearest(to_lab(source_rgb[line_mask]), to_lab(palette))
            image[line_mask] = palette[ink_labels]
    return Image.fromarray(image, mode="RGB")


# 이 면적을 넘는 넓은 영역은 번호를 여러 번 찍어 어느 칸인지 읽기 쉽게 한다
REPEAT_LABEL_AREA = 60000


def render_outline_image(
    shape: tuple[int, int],
    seams: list[np.ndarray],
    regions: list[Region],
    palette: np.ndarray,
    line_mask: np.ndarray | None = None,
    source_rgb: np.ndarray | None = None,
    detail_path: Path | None = None,
) -> tuple[Image.Image, list[dict]]:
    """윤곽선 + 번호 + 팔레트 바를 합성한다. 영역별 번호 배치 결과(reason 포함)를 함께 돌려준다."""
    h, w = shape
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # Draw shared face boundaries once, including at printed-ink junctions.
    seam_pixels = np.zeros((h, w), np.uint8)
    for seam in seams:
        if len(seam) >= 2:
            cv2.polylines(seam_pixels, [np.asarray(seam, np.int32)], False, 1, 1)
    if line_mask is not None:
        # Shared ownership is extended to the centre of ink. These seams close
        # the paint faces; only add free-ended ink not represented by a seam.
        covered = cv2.dilate(seam_pixels, np.ones((9, 9), np.uint8)).astype(bool)
        seam_pixels[skeletonize(line_mask) & ~covered] = 1
        if source_rgb is not None:
            # Neutral dark ink is printed solid at its original width; coloured
            # decorative strokes use a centreline rather than black ribbons.
            dark = source_rgb.max(axis=2) < 105
            seam_pixels[line_mask & dark] = 1
    painted = np.full((h, w, 3), 255, np.uint8)
    painted[seam_pixels != 0] = 0
    canvas = Image.fromarray(painted)
    draw = ImageDraw.Draw(canvas)

    clean_outline = canvas.copy()
    details = []
    placed: list[tuple[float, float, float, float]] = []
    number_log: list[dict] = []
    # 넓은 영역부터 자리를 잡아, 작은 영역이 큰 영역의 유일한 자리를 뺏지 않게 한다
    for region in sorted(regions, key=lambda r: -r.area):
        text = str(region.number)
        b = draw.textbbox((0, 0), text, font=font)
        tw, th = b[2] - b[0], b[3] - b[1]
        entry = {"id": region.id, "number": region.number, "placed": False,
                 "kind": None, "reason": None}

        def box_at(cx: float, cy: float) -> tuple[float, float, float, float]:
            return (cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2)

        def try_put(cx: float, cy: float) -> bool:
            box = box_at(cx, cy)
            if any(_boxes_overlap(box, o) for o in placed):
                return False
            if line_mask is not None and _box_hits_mask(box, line_mask):
                return False
            draw.text((box[0] - b[0], box[1] - b[1]), text, fill="black", font=font)
            placed.append(box)
            return True

        fit = [(x, y, r) for (x, y, r) in region.label_candidates if math.hypot(tw, th) / 2 <= r]
        done = 0
        for (x, y, _) in fit:
            if try_put(x, y):
                done += 1
                entry.update(placed=True, kind="internal", reason=None)
                if region.area < REPEAT_LABEL_AREA or done >= 3:
                    break

        if done == 0:
            details.append(region)
            entry.update(placed=detail_path is not None, kind="detail",
                         reason="enlarged detail sheet" if detail_path else "no readable internal spot")
        number_log.append(entry)

    if detail_path is not None:
        if details:
            render_detail_sheet(clean_outline, details).save(detail_path)
        else:
            detail_path.unlink(missing_ok=True)

    bar = _render_palette_bar(len(palette), palette, w)
    composed = Image.new("RGB", (w, h + bar.height), "white")
    composed.paste(bar, (0, 0))
    composed.paste(canvas, (0, bar.height))
    return composed, number_log


def _box_hits_mask(box: tuple[float, float, float, float], mask: np.ndarray) -> bool:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    sub = mask[y0 : y1 + 1, x0 : x1 + 1]
    return bool(sub.any())


def _boxes_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


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


def render_detail_sheet(outline: Image.Image, regions: list[Region]) -> Image.Image:
    """Each small face gets an enlarged crop, an inside target and a locator."""
    cols, cell_w, cell_h = 3, 280, 260
    sheet = Image.new("RGB", (cols * cell_w, ((len(regions) + cols - 1) // cols) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for i, r in enumerate(regions):
        ox, oy = (i % cols) * cell_w, (i // cols) * cell_h
        cx, cy = r.label_anchor
        half = 28
        tile = Image.new("RGB", (2 * half, 2 * half), "white")
        bounds = (max(0, cx-half), max(0, cy-half),
                  min(outline.width, cx+half), min(outline.height, cy+half))
        tile.paste(outline.crop(bounds), (bounds[0] - cx + half, bounds[1] - cy + half))
        crop = tile.resize((168, 168), Image.Resampling.NEAREST)
        sheet.paste(crop, (ox+8, oy+40))
        # Crosshair identifies the exact region; the colour number is outside
        # the crop so it cannot obscure a narrow cell.
        tx, ty = ox+92, oy+124
        draw.ellipse((tx-3, ty-3, tx+3, ty+3), outline="red", width=2)
        draw.text((ox+8, oy+10), f"Color {r.number} | Detail {i+1}", fill="black", font=font)
        thumb = outline.copy()
        thumb.thumbnail((82, 160))
        sx, sy = ox+188, oy+40
        sheet.paste(thumb, (sx, sy))
        px = sx + cx * thumb.width / outline.width
        py = sy + cy * thumb.height / outline.height
        draw.rectangle((px-4, py-4, px+4, py+4), outline="red", width=2)
        draw.text((ox+8, oy+222), "Red mark = area to paint", fill="black", font=font)
        draw.rectangle((ox, oy, ox+cell_w-1, oy+cell_h-1), outline="#cccccc")
    return sheet
