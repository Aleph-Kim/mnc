import cv2
import numpy as np

from app.imaging.quantize import to_lab

# Thresholds use the pipeline's normalized working resolution.
INK_BLACKHAT_RADIUS = 15
INK_CONTRAST = 2.5  # CIELAB luminance valley after JPEG smoothing
MIN_FILL_AREA = 150
FILL_BBOX_RATIO = 0.55
MIN_LINE_AREA = 25
MIN_LINE_EXTENT = 22


def detect_line_layer(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """원화의 어두운 선과 유색 선을 별도 마스크로 뽑는다.

    (선 마스크, 선/면 분리 전 잉크 마스크)를 돌려준다 — 두 번째는 디버그에서 어떤
    덩어리가 면으로 빠졌는지 보기 위함.
    """
    h, w = rgb.shape[:2]
    lab = to_lab(rgb.reshape(-1, 3)).reshape(h, w, 3)
    # A stroke is a local luminance valley, including coloured ink. Smooth JPEG
    # grain first; chroma exclusion previously missed olive and blue outlines.
    light = cv2.GaussianBlur(lab[:, :, 0].astype(np.float32), (0, 0), 1.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INK_BLACKHAT_RADIUS + 1,) * 2)
    valley = cv2.morphologyEx(light, cv2.MORPH_BLACKHAT, kernel)
    ink = (valley > INK_CONTRAST).astype(np.uint8)

    # Dark compact interiors (pupils/shoes) remain paint. Coloured strokes
    # can be wider, so they need a larger core before being classified as fill.
    def fill_core(radius):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
        return _compact_blobs(opened.astype(bool), MIN_FILL_AREA, FILL_BBOX_RATIO)

    is_fill = fill_core(12) | (fill_core(6) & (light < 35))
    # Do not cut a compact spot in half. A complete pupil/spot is a paint
    # component, unlike an open or hollow contour with a low bbox occupancy.
    is_fill |= _compact_blobs(ink.astype(bool), MIN_FILL_AREA, FILL_BBOX_RATIO)

    line = ink.astype(bool) & ~is_fill
    line = cv2.morphologyEx(
        line.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)
    line = _drop_small(line, MIN_LINE_AREA)
    _, ids, stats, _ = cv2.connectedComponentsWithStats(line.astype(np.uint8), connectivity=8)
    # Compact freckles and sand grain are not printed strokes.
    keep = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= MIN_LINE_EXTENT
    keep[0] = False
    return keep[ids], ink.astype(bool)


def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    _, components, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[components]


def _compact_blobs(mask: np.ndarray, min_area: int, bbox_ratio: float) -> np.ndarray:
    """자기 bbox를 꽉 채우는(속이 안 빈) 큰 덩어리만 True — 동공·신발.

    외곽선은 연결요소가 커도 bbox 대비 속이 비어 걸러진다. 그래서 굵은 외곽선 구간도
    선으로 남고, 같은 윤곽선이 구간마다 선/면으로 갈리지 않는다.
    """
    n, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for c in range(1, n):
        area = stats[c, cv2.CC_STAT_AREA]
        bbox = stats[c, cv2.CC_STAT_WIDTH] * stats[c, cv2.CC_STAT_HEIGHT]
        if area >= min_area and bbox > 0 and area / bbox >= bbox_ratio:
            keep[c] = True
    return keep[components]
