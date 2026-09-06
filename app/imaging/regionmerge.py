"""원본 이미지의 경계 근거로 인접 영역을 병합한다.

merge_small_regions는 면적·두께만 보므로 하늘·바닥의 넓은 가짜 색 띠는 손대지 못한다.
여기서는 초기 영역을 노드로 하는 인접 그래프를 만들고, 두 영역이 맞닿은 자리에 원본
이미지가 실제로 경계 근거를 갖는지(여러 스케일의 색·밝기 변화)를 평가해서, 근거가
약한 경계를 지운다. 선·구멍·가는 물체·실루엣은 보호한다.
"""

import cv2
import numpy as np

from app.imaging.quantize import to_lab

# 공유 경계의 원본 gradient(dE/px) 상위 백분위가 이보다 낮으면 "진짜 경계 아님"
EDGE_P90_MAX = 6.0

# 경계 픽셀 중 이 gradient를 넘는 비율이 이보다 크면 진짜 경계로 보고 보호
STRONG_GRAD = 8.0
STRONG_FRAC_MAX = 0.15

# 이보다 원본 경계 근거가 약하면 "명백한 그라데이션 띠"로 본다. 이 경우 두 팔레트 색이
# 멀어도 병합한다 — 원본이 실제로 연속 변하는 램프라 색차는 "지우는 그라데이션"이지
# "서로 다른 두 면"이 아니기 때문
GRADIENT_BAND_P90 = 3.0
GRADIENT_BAND_STRONG_FRAC = 0.03
GRADIENT_MERGE_ERROR_MAX = 22.0

# 근거가 어중간할 때(그라데이션 띠는 아니지만 경계도 약할 때) 병합 후 늘어나는 평균 오차 상한
MERGED_ERROR_MAX = 8.0

# 한 영역이 연쇄 병합으로 누적할 수 있는 평균 오차(dE) 상한
CUM_ERROR_MAX = 25.0

# 가는 영역은 실제 경계 근거가 있을 때 보호한다. 경계 근거 없는 얇은 띠는 병합한다.
THIN_RADIUS = 4



def merge_by_edge_evidence(
    region_map: np.ndarray,
    region_labels: np.ndarray,
    rgb: np.ndarray,
    line_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    count = len(region_labels)
    if count <= 1:
        return region_map, region_labels, []
    if line_mask is None:
        line_mask = np.zeros(region_map.shape, dtype=bool)

    lab = to_lab(rgb.reshape(-1, 3)).reshape(*rgb.shape[:2], 3)
    grad_max = _multiscale_gradient(lab)
    line_near = cv2.dilate(line_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    areas = np.bincount(region_map[region_map >= 0].ravel(), minlength=count).astype(np.float64)
    mean_lab = _region_means(region_map, lab, count)
    interior_var = _region_interior_variation(region_map, grad_max, count)
    thin = _thin_regions(region_map, count)

    pairs, edges = _adjacency(region_map, count)

    candidates: list[dict] = []
    for (a, b), (ys, xs) in edges.items():
        border_len = len(ys)
        if border_len < 12:
            continue
        # Printed ink already preserves this boundary. Measure the exposed
        # boundary separately: a flower and a gradient band can share the same
        # pair of colour regions. Pooling them protected the entire false band.
        exposed = ~line_near[ys, xs]
        if exposed.sum() < 12:
            continue
        edge_vals = grad_max[ys[exposed], xs[exposed]]
        p90 = float(np.percentile(edge_vals, 90))
        strong_frac = float((edge_vals > STRONG_GRAD).mean())
        on_line = float(line_near[ys, xs].mean())
        merged_err = _merge_error(mean_lab[a], mean_lab[b], areas[a], areas[b])
        interior = max(interior_var[a], interior_var[b])

        is_gradient_band = p90 < GRADIENT_BAND_P90 and strong_frac < GRADIENT_BAND_STRONG_FRAC

        reason = None
        if strong_frac > STRONG_FRAC_MAX:
            reason = "real edge on >15% of border"
        elif is_gradient_band:
            if merged_err > GRADIENT_MERGE_ERROR_MAX:
                reason = "gradient band error too high"
        elif thin[a] or thin[b]:
            reason = "thin region with boundary evidence"
        elif p90 > max(EDGE_P90_MAX, interior * 1.3):
            reason = "edge sharper than region interiors"
        elif merged_err > MERGED_ERROR_MAX:
            reason = "merged color error too high"

        candidates.append({
            "a": int(a), "b": int(b), "border_len": border_len,
            "edge_p90": round(p90, 2), "strong_frac": round(strong_frac, 3),
            "on_line_frac": round(on_line, 3), "interior_var": round(float(interior), 2),
            "merged_error": round(merged_err, 2),
            "merge": reason is None, "reject_reason": reason,
        })

    parent = np.arange(count)
    cum_error = np.zeros(count)
    comp_area = areas.copy()
    comp_lab = mean_lab.copy()

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    for cand in sorted((c for c in candidates if c["merge"]), key=lambda c: c["merged_error"]):
        ra, rb = root(cand["a"]), root(cand["b"])
        if ra == rb:
            cand["merge"], cand["reject_reason"] = False, "already merged"
            continue
        err = _merge_error(comp_lab[ra], comp_lab[rb], comp_area[ra], comp_area[rb])
        if cum_error[ra] + err > CUM_ERROR_MAX or cum_error[rb] + err > CUM_ERROR_MAX:
            cand["merge"], cand["reject_reason"] = False, "cumulative error cap"
            continue
        keep, drop = (ra, rb) if comp_area[ra] >= comp_area[rb] else (rb, ra)
        total = comp_area[keep] + comp_area[drop]
        comp_lab[keep] = (comp_lab[keep] * comp_area[keep] + comp_lab[drop] * comp_area[drop]) / total
        comp_area[keep] = total
        cum_error[keep] = max(cum_error[ra], cum_error[rb]) + err
        parent[drop] = keep

    final = np.array([root(i) for i in range(count)])
    survivors, compact = np.unique(final, return_inverse=True)
    lookup = np.append(compact.astype(np.int32), -1)
    new_map = lookup[region_map]

    # 대표색: 병합된 그룹에서 면적이 가장 큰 원본 영역의 팔레트 색을 잇는다
    new_labels = np.empty(len(survivors), dtype=region_labels.dtype)
    for new_id, members in _group_members(compact, len(survivors)):
        new_labels[new_id] = region_labels[members[np.argmax(areas[members])]]

    return new_map, new_labels, candidates


def _multiscale_gradient(lab: np.ndarray) -> np.ndarray:
    """Measure gradients after smoothing JPEG grain at two scales."""
    out = None
    for sigma in (1.5, 3.0):
        src = lab if sigma == 0 else cv2.GaussianBlur(lab, (0, 0), sigma)
        gx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
        g = np.sqrt((gx ** 2 + gy ** 2).sum(axis=2)) / 4.0
        out = g if out is None else np.maximum(out, g)
    return out


def _region_means(region_map: np.ndarray, lab: np.ndarray, count: int) -> np.ndarray:
    flat = region_map.ravel()
    valid = flat >= 0
    ids = flat[valid]
    px = lab.reshape(-1, 3)[valid]
    sums = np.stack([np.bincount(ids, px[:, c], minlength=count) for c in range(3)], axis=1)
    n = np.bincount(ids, minlength=count).clip(1)
    return sums / n[:, None]


def _region_interior_variation(region_map: np.ndarray, grad_max: np.ndarray, count: int) -> np.ndarray:
    """영역 내부(경계 제외)의 gradient 중앙값. 질감이 거친 영역은 이 값이 크다."""
    field = region_map.astype(np.float32)
    kernel = np.ones((5, 5), np.uint8)
    same = cv2.erode(field, kernel) == cv2.dilate(field, kernel)
    flat = region_map.ravel()
    g = grad_max.ravel()
    out = np.zeros(count)
    ids = flat[same.ravel() & (flat >= 0)]
    gv = g[same.ravel() & (flat >= 0)]
    order = np.argsort(ids, kind="stable")
    ids, gv = ids[order], gv[order]
    starts = np.searchsorted(ids, np.arange(count))
    ends = np.searchsorted(ids, np.arange(count), side="right")
    for i in range(count):
        if ends[i] > starts[i]:
            out[i] = np.median(gv[starts[i]:ends[i]])
    return out


def _thin_regions(region_map: np.ndarray, count: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * THIN_RADIUS + 1, 2 * THIN_RADIUS + 1))
    field = region_map.astype(np.float32)
    lo = cv2.erode(field, kernel, borderType=cv2.BORDER_REPLICATE)
    hi = cv2.dilate(field, kernel, borderType=cv2.BORDER_REPLICATE)
    fat = np.bincount(region_map[(lo == hi) & (region_map >= 0)], minlength=count)
    return fat == 0


def _merge_error(lab_a: np.ndarray, lab_b: np.ndarray, area_a: float, area_b: float) -> float:
    total = area_a + area_b
    merged = (lab_a * area_a + lab_b * area_b) / total
    da = np.linalg.norm(lab_a - merged)
    db = np.linalg.norm(lab_b - merged)
    return float((da * area_a + db * area_b) / total)


def _adjacency(region_map: np.ndarray, count: int):
    h, w = region_map.shape
    edges: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = region_map[: h - dy, : w - dx]
        b = region_map[dy:, dx:]
        m = (a != b) & (a >= 0) & (b >= 0)
        ys, xs = np.nonzero(m)
        ka = a[ys, xs]
        kb = b[ys, xs]
        lo = np.minimum(ka, kb)
        hi = np.maximum(ka, kb)
        for y, x, l, g in zip(ys, xs, lo, hi):
            edges.setdefault((int(l), int(g)), ([], []))
            edges[(int(l), int(g))][0].append(int(y))
            edges[(int(l), int(g))][1].append(int(x))
    edges = {k: (np.array(v[0]), np.array(v[1])) for k, v in edges.items()}
    return list(edges.keys()), edges


def _group_members(compact: np.ndarray, n: int):
    order = np.argsort(compact, kind="stable")
    sorted_c = compact[order]
    starts = np.searchsorted(sorted_c, np.arange(n))
    ends = np.searchsorted(sorted_c, np.arange(n), side="right")
    for i in range(n):
        yield i, order[starts[i]:ends[i]]
