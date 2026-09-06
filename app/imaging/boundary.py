import cv2
import numpy as np

# 선 마스크를 이만큼 넓혀 AA 전이 픽셀까지 "불확실"로 잡는다
UNCERTAIN_RADIUS = 2

# 이보다 얇은 색 띠는 실루엣을 따라 도는 가짜 띠 후보 — 양옆 확정 면이 서로 같은 색이고
# 띠 색이 그 색과 다르면 불확실로 넣는다. 단 진짜 가는 물체는 THIN_OBJECT_EXTENT로 보호
SUSPECT_BAND_HALFWIDTH = 2
THIN_OBJECT_EXTENT = 40.0


def reassign_uncertain(
    label_map: np.ndarray, line_mask: np.ndarray, rgb: np.ndarray
) -> tuple[np.ndarray, dict]:
    """선 + 그 주변 AA 전이 픽셀 + 의심스러운 얇은 색 띠를, 실제 이미지 경계를 따라
    이웃 확정 면에 다시 배정한다.

    시드는 팔레트 색 번호가 아니라 경계에서 떨어진 확정 면의 연결요소 ID다. 같은 색을
    가진 두 면도 공간적으로 별도라 서로 다른 시드를 갖는다. 가짜 띠 자체는 불확실로
    들어가 시드가 되지 않는다.
    """
    h, w = label_map.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * UNCERTAIN_RADIUS + 1, 2 * UNCERTAIN_RADIUS + 1)
    )
    uncertain = cv2.dilate(line_mask.astype(np.uint8), kernel).astype(bool)
    uncertain |= _suspect_bands(label_map, line_mask)

    stats = {
        "uncertain_px": int(uncertain.sum()),
        "unresolved_px": 0,
        "changed_px": 0,
        "fallback": None,
    }
    if not uncertain.any():
        return label_map, stats

    # 확정 면의 연결요소 = 시드 (같은 색이어도 공간적으로 나뉘면 별도 ID)
    confident = ~uncertain
    seed_id = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    seed_color: list[int] = [0]
    for color in np.unique(label_map[confident]):
        blob = confident & (label_map == color)
        n, comp = cv2.connectedComponents(blob.astype(np.uint8), connectivity=4)
        seed_id[blob] = comp[blob] + (next_id - 1)
        seed_color.extend([int(color)] * (n - 1))
        next_id += n - 1

    if next_id <= 1:  # 확정 시드가 전혀 없는 예외 입력
        stats["fallback"] = "no confident seed — label_map unchanged"
        return label_map, stats

    markers = seed_id.copy()
    markers[uncertain] = 0
    cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), markers)

    color_of = np.array(seed_color, dtype=label_map.dtype)
    out = label_map.copy()
    resolved = uncertain & (markers > 0)
    out[resolved] = color_of[markers[resolved]]

    holes = uncertain & (markers <= 0)
    out, remaining = _fill_holes(out, holes)
    stats["unresolved_px"] = int(remaining)
    if remaining:
        stats["fallback"] = f"{remaining}px kept original label (no valid neighbour)"
    stats["changed_px"] = int((out != label_map).sum())
    return out, stats


def _suspect_bands(label_map: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    """검출된 선 바로 옆에 붙은 얇은 색 조각 = 실루엣 halo 후보.

    선에서 먼 곳의 얇은 조각(하늘 잡티 등)은 건드리지 않는다 — 그건 병합 단계 몫이다.
    진짜 가는 물체(선인장 등)는 길게 뻗으므로 THIN_OBJECT_EXTENT로 보호.
    """
    r = SUSPECT_BAND_HALFWIDTH
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near_line = cv2.dilate(
        line_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    ).astype(bool)

    thin = np.zeros(label_map.shape, dtype=bool)
    for color in np.unique(label_map):
        band = (label_map == color).astype(np.uint8)
        skinny = band.astype(bool) & (cv2.erode(band, kernel) == 0)
        if not skinny.any():
            continue
        n, comp, st, _ = cv2.connectedComponentsWithStats(skinny.astype(np.uint8), connectivity=8)
        for c in range(1, n):
            piece = comp == c
            if not (piece & near_line).any():
                continue
            diag = np.hypot(st[c, cv2.CC_STAT_WIDTH], st[c, cv2.CC_STAT_HEIGHT])
            if diag < THIN_OBJECT_EXTENT:
                thin[piece] = True
    return thin


def _fill_holes(label_map: np.ndarray, holes: np.ndarray) -> tuple[np.ndarray, int]:
    """watershed가 남긴 픽셀을 이웃 최빈 라벨로 메운다. 유효한 이웃이 생길 때까지 반복."""
    out = label_map.copy()
    todo = holes.copy()
    for _ in range(8):
        if not todo.any():
            break
        known = ~todo
        filled = out.copy()
        progressed = np.zeros(out.shape, dtype=bool)
        for y, x in zip(*np.nonzero(todo)):
            y0, y1 = max(0, y - 1), min(out.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(out.shape[1], x + 2)
            near = out[y0:y1, x0:x1][known[y0:y1, x0:x1]]
            if near.size:
                values, counts = np.unique(near, return_counts=True)
                filled[y, x] = values[counts.argmax()]
                progressed[y, x] = True
        out = filled
        if not progressed.any():
            break
        todo &= ~progressed
    return out, int(todo.sum())
