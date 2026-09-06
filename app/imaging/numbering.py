from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Region:
    id: int
    number: int
    color_hex: str
    points: list[list[int]]
    label_anchor: tuple[int, int]
    label_radius: float
    # 내부 배치 후보 (거리변환 봉우리 상위 몇 개), 넓은 순. (x, y, 여유반지름)
    label_candidates: list[tuple[int, int, float]]
    # 중심 근처 좌표 — 내부에 못 넣을 때 바깥 번호+인출선의 목적지
    centroid: tuple[int, int]
    area: int


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def assign_numbers(
    region_map: np.ndarray,
    region_labels: np.ndarray,
    palette: np.ndarray,
    contours_by_region: dict[int, np.ndarray],
    line_mask: np.ndarray | None = None,
) -> list[Region]:
    height, width = region_map.shape
    order, starts, ends = _region_pixels(region_map, len(region_labels))
    # 인쇄 선이 지나는 픽셀은 칠할 수 없으므로 번호 배치 후보에서 뺀다
    blocked = line_mask if line_mask is not None else np.zeros(region_map.shape, dtype=bool)

    regions: list[Region] = []
    for region_id, points in contours_by_region.items():
        color_label = int(region_labels[region_id])
        pixels = order[starts[region_id] : ends[region_id]]
        candidates = _label_candidates(region_map, region_id, pixels, width, blocked)
        ys, xs = pixels // width, pixels % width
        centroid = (int(xs.mean()), int(ys.mean())) if pixels.size else (0, 0)
        anchor, radius = (candidates[0][:2], candidates[0][2]) if candidates else (centroid, 0.0)

        regions.append(
            Region(
                id=region_id,
                number=color_label + 1,
                color_hex=rgb_to_hex(palette[color_label]),
                points=points.tolist(),
                label_anchor=tuple(anchor),
                label_radius=radius,
                label_candidates=candidates,
                centroid=centroid,
                area=int(pixels.size),
            )
        )

    return regions


def _region_pixels(region_map: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """영역별 픽셀 위치를 한 번에 뽑는다.

    영역마다 region_map == id로 마스크를 만들면 영역 수 × 이미지 크기가 되어, 잔디나
    털처럼 영역이 수천 개 나오는 사진에서 급격히 느려진다.
    """
    flat = region_map.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_ids = flat[order]
    ids = np.arange(count)
    return order, np.searchsorted(sorted_ids, ids), np.searchsorted(sorted_ids, ids, side="right")


def _label_candidates(
    region_map: np.ndarray, region_id: int, pixels: np.ndarray, width: int,
    blocked: np.ndarray, top_n: int = 3,
) -> list[tuple[int, int, float]]:
    """영역 안에서 (인쇄 선을 뺀) 칠할 수 있는 자리 중 경계로부터 먼 봉우리 상위 top_n개.

    거리변환 최댓값만 쓰면 큰 영역에 번호 하나뿐이라 읽기 어렵고, 그 한 자리가
    다른 번호와 겹치면 통째로 생략된다. 봉우리를 여러 개 두면 render가 겹침을 피해
    고르거나, 넓은 영역에 반복 배치할 수 있다.
    """
    if pixels.size == 0:
        return []

    ys, xs = pixels // width, pixels % width
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    mask = ((region_map[y0:y1, x0:x1] == region_id) & ~blocked[y0:y1, x0:x1]).astype(np.uint8)
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)

    out: list[tuple[int, int, float]] = []
    work = distance.copy()
    for _ in range(top_n):
        cy, cx = np.unravel_index(int(np.argmax(work)), work.shape)
        r = float(work[cy, cx])
        if r <= 0:
            break
        out.append((int(x0 + cx - 1), int(y0 + cy - 1), r))
        cv2.circle(work, (cx, cy), max(int(r * 1.5), 8), 0.0, -1)
    return out
