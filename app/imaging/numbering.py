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


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def assign_numbers(
    region_map: np.ndarray,
    region_labels: np.ndarray,
    palette: np.ndarray,
    contours_by_region: dict[int, np.ndarray],
) -> list[Region]:
    height, width = region_map.shape
    order, starts, ends = _region_pixels(region_map, len(region_labels))

    regions: list[Region] = []
    for region_id, points in contours_by_region.items():
        color_label = int(region_labels[region_id])
        pixels = order[starts[region_id] : ends[region_id]]
        anchor, radius = _label_anchor(region_map, region_id, pixels, width)

        regions.append(
            Region(
                id=region_id,
                number=color_label + 1,
                color_hex=rgb_to_hex(palette[color_label]),
                points=points.tolist(),
                label_anchor=anchor,
                label_radius=radius,
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


def _label_anchor(
    region_map: np.ndarray, region_id: int, pixels: np.ndarray, width: int
) -> tuple[tuple[int, int], float]:
    """영역 안에서 경계로부터 가장 먼 지점과 그 여유 반지름.

    무게중심은 C자·고리·초승달 모양에서 영역 바깥으로 나간다. 그러면 번호가 엉뚱한
    영역 위에 찍히고, 오목한 영역이 여럿 모인 자리에서는 여러 번호가 같은 빈 공간에
    겹쳐 쌓인다. 거리변환 최댓값 지점은 항상 영역 내부이며 가장 넓게 트인 자리라
    번호를 놓기에 맞다.
    """
    if pixels.size == 0:
        return (0, 0), 0.0

    ys, xs = pixels // width, pixels % width
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    mask = (region_map[y0:y1, x0:x1] == region_id).astype(np.uint8)
    # 이미지 가장자리에 닿은 영역도 경계로 치도록 0으로 한 겹 두른다.
    # 번호가 잘리는 자리로 가지 않는다.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)

    cy, cx = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return (int(x0 + cx - 1), int(y0 + cy - 1)), float(distance.max())
