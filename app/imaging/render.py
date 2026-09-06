import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.imaging.numbering import Region

PALETTE_ROW_HEIGHT = 60
SWATCH_SIZE = 60


def render_preview_image(
    region_map: np.ndarray,
    region_labels: np.ndarray,
    palette: np.ndarray,
    line_mask: np.ndarray | None = None,
) -> Image.Image:
    # 색 라벨맵이 아니라 영역맵에서 칠한다. 라벨맵에는 병합으로 사라진 잔점이 그대로
    # 남아 있어서, 그걸로 미리보기를 만들면 윤곽선도 번호도 없는 색 얼룩이 보인다.
    # 즉 사용자가 번호대로 칠한 결과와 미리보기가 서로 다른 그림이 된다.
    image = palette[region_labels[region_map]]
    if line_mask is not None:
        # 인쇄되는 원화 선을 얹는다 — 사용자가 칠하는 대상은 면뿐
        image = image.copy()
        image[line_mask] = (0, 0, 0)
    return Image.fromarray(image, mode="RGB")


# seam이 이 비율 이상 인쇄 선에 덮이면 그 경계는 선이 담당 — seam을 그리지 않는다
LINE_COVERS_SEAM_FRAC = 0.6

# seam-덮임 판정 시 선 마스크를 넓히는 반지름. 경계 재배정이 영역 경계를 선 바깥으로
# 살짝 밀어내므로(boundary.UNCERTAIN_RADIUS), 그만큼 여유를 둬야 이중선이 안 남는다
LINE_NEAR_RADIUS = 3

# 내부에 번호를 못 넣어도 이 면적 이상이면 바깥 번호 + 인출선으로 안내
MIN_EXTERNAL_LABEL_AREA = 120

# 이 면적을 넘는 넓은 영역은 번호를 여러 번 찍어 어느 칸인지 읽기 쉽게 한다
REPEAT_LABEL_AREA = 60000


def render_outline_image(
    shape: tuple[int, int],
    seams: list[np.ndarray],
    regions: list[Region],
    palette: np.ndarray,
    line_mask: np.ndarray | None = None,
) -> tuple[Image.Image, list[dict]]:
    """윤곽선 + 번호 + 팔레트 바를 합성한다. 영역별 번호 배치 결과(reason 포함)를 함께 돌려준다."""
    h, w = shape
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    line_near = None
    if line_mask is not None:
        k = 2 * LINE_NEAR_RADIUS + 1
        line_near = cv2.dilate(
            line_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        ).astype(bool)

    for seam in seams:
        if len(seam) < 2:
            continue
        pts = [tuple(int(v) for v in p) for p in seam]
        if line_near is not None and _seam_covered_by_line(pts, line_near, (h, w)):
            continue
        draw.line(pts, fill="black", width=1)

    # 원화 선을 번호보다 먼저 얹는다 — 선이 번호를 가리면 안 되므로 번호가 맨 위
    if line_mask is not None:
        painted = np.array(canvas)
        painted[line_mask] = (0, 0, 0)
        canvas = Image.fromarray(painted, mode="RGB")
        draw = ImageDraw.Draw(canvas)

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
            if region.area >= MIN_EXTERNAL_LABEL_AREA:
                spot = _external_spot(region.centroid, tw, th, placed, (h, w), line_mask)
                if spot is not None:
                    ex, ey = spot
                    draw.line([region.centroid, (int(ex), int(ey))], fill="black", width=1)
                    draw.text((ex - tw / 2 - b[0], ey - th / 2 - b[1]), text, fill="black", font=font)
                    placed.append(box_at(ex, ey))
                    entry.update(placed=True, kind="external", reason="internal placement failed")
                else:
                    entry["reason"] = "no internal or external spot found"
            else:
                entry["reason"] = "region too small for glyph"
        number_log.append(entry)

    bar = _render_palette_bar(len(palette), palette, w)
    composed = Image.new("RGB", (w, h + bar.height), "white")
    composed.paste(bar, (0, 0))
    composed.paste(canvas, (0, bar.height))
    return composed, number_log


def _seam_covered_by_line(pts: list[tuple[int, int]], line_near: np.ndarray, shape) -> bool:
    tmp = np.zeros(shape, dtype=np.uint8)
    cv2.polylines(tmp, [np.array(pts, dtype=np.int32)], False, 1, 1)
    total = int(tmp.sum())
    if total == 0:
        return False
    return (tmp.astype(bool) & line_near).sum() / total >= LINE_COVERS_SEAM_FRAC


def _external_spot(centroid, tw, th, placed, shape, line_mask):
    """중심에서 바깥으로 링을 넓히며 번호와 선을 피한 자리를 찾는다. 인출선 목적지."""
    h, w = shape
    cx, cy = centroid
    for radius in range(14, 90, 8):
        for ang in range(0, 360, 30):
            ex = cx + radius * math.cos(math.radians(ang))
            ey = cy + radius * math.sin(math.radians(ang))
            if not (tw < ex < w - tw and th < ey < h - th):
                continue
            box = (ex - tw / 2, ey - th / 2, ex + tw / 2, ey + th / 2)
            if any(_boxes_overlap(box, o) for o in placed):
                continue
            if line_mask is not None and _box_hits_mask(box, line_mask):
                continue
            return ex, ey
    return None


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
