import cv2
import numpy as np

# 선 마스크를 이만큼 넓혀 AA 전이 픽셀까지 "불확실"로 잡는다
UNCERTAIN_RADIUS = 2


def reassign_uncertain(
    label_map: np.ndarray, line_mask: np.ndarray, rgb: np.ndarray
) -> np.ndarray:
    """선 + 그 주변 AA 전이 픽셀을 실제 이미지 경계를 따라 이웃 확정 영역에 다시 배정한다.

    양자화는 실루엣의 반투명 혼합색을 엉뚱한 팔레트 색(노랑·파랑 사이의 연두 등)에 넣어
    실루엣을 도는 가짜 띠를 만든다. 이 자리를 watershed로 진짜 경계 기준으로 다시 나눈다.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * UNCERTAIN_RADIUS + 1, 2 * UNCERTAIN_RADIUS + 1)
    )
    uncertain = cv2.dilate(line_mask.astype(np.uint8), kernel).astype(bool)
    if not uncertain.any():
        return label_map

    # 0은 "미정" 자리라 라벨을 1씩 올려 쓴다. watershed가 경계 픽셀을 -1로 남긴다
    markers = label_map.astype(np.int32) + 1
    markers[uncertain] = 0
    cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), markers)

    out = label_map.copy()
    filled = uncertain & (markers > 0)
    out[filled] = markers[filled] - 1

    leftover = uncertain & (markers <= 0)
    if leftover.any():
        out = _fill_holes(out, leftover)
    return out


def _fill_holes(label_map: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """watershed가 남긴 1px 경계선을 이웃 최빈 라벨로 메운다."""
    h, w = label_map.shape
    filled = label_map.copy()
    for y, x in zip(*np.nonzero(holes)):
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        patch = label_map[y0:y1, x0:x1]
        near = patch[~holes[y0:y1, x0:x1]]
        if near.size:
            values, counts = np.unique(near, return_counts=True)
            filled[y, x] = values[counts.argmax()]
    return filled
