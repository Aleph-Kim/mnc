"""원인 검증용 진단 스크립트 (프로덕션 코드 아님).

test2.jpeg를 작업 해상도로 올린 뒤 파이프라인 단계를 수동으로 돌리며,
사용자가 지적한 원인 A~F의 실제 발생 여부를 수치로 측정한다.

실행: docker compose run --rm -v "$PWD/scripts:/app/scripts" web python scripts/diag.py
결과: storage/outputs/_diag/<git-rev>/ 에 npz·json·overlay
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from app.imaging.contours import extract_seams
from app.imaging.denoise import denoise_label_map
from app.imaging.lineart import detect_line_layer
from app.imaging.pipeline import _resize_to_working_dim
from app.imaging.quantize import (
    ATLAS_COLORS,
    _atlas,
    _flat_mask,
    _nearest,
    _pick_palette,
    _refine,
    _merge_indistinct,
    quantize_colors,
    to_lab,
    to_rgb,
)
from app.imaging.segment import _shared_borders, merge_small_regions, segment_regions

SRC = Path("/data/test2.jpeg")
K = 11


def git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd="/app").decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd="/app") != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main() -> None:
    out = Path("/app/storage/outputs/_diag") / git_rev()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"git_rev": git_rev(), "k": K}

    bgr = cv2.imread(str(SRC))
    report["input_md5"] = hashlib.md5(SRC.read_bytes()).hexdigest()
    rgb = cv2.cvtColor(_resize_to_working_dim(bgr, 1200), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    report["working_size"] = [h, w]
    lab = to_lab(rgb.reshape(-1, 3)).reshape(h, w, 3)
    L = lab[:, :, 0]

    # ---------- E/8: 밝은 면(눈·치아·셔츠)이 어느 단계에서 대표성을 잃나 ----------
    # 밝은 무채색 자리 = 흰자/치아/셔츠 후보
    bright = (L > 75) & (np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2) < 18)
    report["bright_white_px"] = int(bright.sum())
    report["bright_white_pct"] = round(100 * bright.mean(), 3)

    pixels = to_lab(rgb.reshape(-1, 3))
    flat_mask = _flat_mask(rgb)
    report["bright_in_flat_pct"] = round(
        100 * flat_mask.reshape(h, w)[bright].mean(), 1
    ) if bright.any() else None

    flat = pixels[flat_mask.ravel()]
    if len(flat) < max(K, ATLAS_COLORS):
        flat = pixels
    atlas, weights = _atlas(flat)
    # atlas 중 밝은 무채색 후보
    atlas_rgb = to_rgb(atlas)
    atlas_bright = (atlas[:, 0] > 75) & (np.sqrt(atlas[:, 1] ** 2 + atlas[:, 2] ** 2) < 18)
    report["atlas_bright_candidates"] = [
        {"lab": atlas[i].round(1).tolist(), "rgb": atlas_rgb[i].round().tolist(),
         "weight": float(weights[i]), "weight_pct": round(100 * weights[i] / weights.sum(), 3)}
        for i in np.where(atlas_bright)[0]
    ]

    picked = _pick_palette(atlas, weights, K)
    picked_rgb = to_rgb(picked)
    report["pick_stage"] = {
        "colors_rgb": picked_rgb.round().astype(int).tolist(),
        "any_bright": bool(((picked[:, 0] > 75) & (np.sqrt(picked[:, 1] ** 2 + picked[:, 2] ** 2) < 18)).any()),
        "min_L": float(picked[:, 0].min()),
        "max_L": float(picked[:, 0].max()),
    }
    # 밝은 후보가 pick 안 됐다면: 가장 가까운 picked 색과의 거리, "가려짐" 여부
    if atlas_bright.any() and not report["pick_stage"]["any_bright"]:
        bidx = np.where(atlas_bright)[0][np.argmax(weights[atlas_bright])]
        d = np.linalg.norm(picked - atlas[bidx], axis=1)
        report["brightest_white_candidate_vs_pick"] = {
            "candidate_rgb": atlas_rgb[bidx].round().tolist(),
            "nearest_picked_rgb": picked_rgb[int(np.argmin(d))].round().tolist(),
            "dE_to_nearest": round(float(d.min()), 1),
        }

    refined = _refine(atlas, weights, picked)
    merged = _merge_indistinct(atlas, weights, refined)
    report["refine_shift_per_color"] = np.linalg.norm(
        to_rgb(refined) - picked_rgb, axis=1
    ).round(1).tolist()
    report["final_palette_rgb"] = np.clip(np.round(to_rgb(merged)), 0, 255).astype(int).tolist()

    # ---------- B/A: 배경 띠가 인접 그래프에서 어떻게 점수화되나 ----------
    label_map, palette = quantize_colors(rgb, K)
    label_map = denoise_label_map(label_map, palette)
    line_mask, ink = detect_line_layer(rgb)
    np.savez_compressed(out / "stage_arrays.npz", rgb=rgb, label_map=label_map,
                        palette=palette, line_mask=line_mask, ink=ink)

    region_map, region_labels = segment_regions(label_map)
    report["regions_after_segment"] = int(len(region_labels))

    # 원본 경계 근거: 여러 스케일의 LAB gradient
    grads = []
    for sigma in (0.0, 1.0, 2.0):
        src = lab if sigma == 0 else cv2.GaussianBlur(lab, (0, 0), sigma)
        gx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
        grads.append(np.sqrt((gx ** 2 + gy ** 2).sum(axis=2)) / 4.0)
    grad_max = np.maximum.reduce(grads)

    owners, neighbours, shared = _shared_borders(region_map, len(region_labels))
    areas = np.bincount(region_map[region_map >= 0].ravel(), minlength=len(region_labels))

    # 가장 큰 영역 몇 개의 인접쌍 경계 근거를 본다 (배경/바닥 띠 후보)
    big = np.argsort(-areas)[:12]
    pairs = []
    boundary = _boundary_pixels(region_map)
    for a in big:
        span = slice(*np.searchsorted(owners, [a, a + 1]))
        for nb, ln in zip(neighbours[span], shared[span]):
            if nb <= a or ln < 30:
                continue
            edge = boundary.get((int(a), int(nb)), [])
            if not edge:
                continue
            ys, xs = np.array(edge).T
            ev = grad_max[ys, xs]
            la, lb = int(region_labels[a]), int(region_labels[nb])
            col_de = float(np.linalg.norm(to_lab(palette[[la]])[0] - to_lab(palette[[lb]])[0]))
            pairs.append({
                "a": int(a), "b": int(nb), "area_a": int(areas[a]), "area_b": int(areas[nb]),
                "border_len": int(ln),
                "palette_dE": round(col_de, 1),
                "orig_edge_p50": round(float(np.percentile(ev, 50)), 1),
                "orig_edge_p90": round(float(np.percentile(ev, 90)), 1),
                "frac_border_strong": round(float((ev > 6).mean()), 2),
                "on_line_frac": round(float(line_mask[ys, xs].mean()), 2),
            })
    pairs.sort(key=lambda p: p["orig_edge_p90"])
    report["adjacent_pairs_biggest_regions"] = pairs

    # ---------- D/6: boundary 재배정 미해결 픽셀 ----------
    from app.imaging.boundary import reassign_uncertain
    before = label_map.copy()
    after = reassign_uncertain(label_map, line_mask, rgb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    uncertain = cv2.dilate(line_mask.astype(np.uint8), kernel).astype(bool)
    markers = before.astype(np.int32) + 1
    markers[uncertain] = 0
    cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), markers)
    report["boundary"] = {
        "uncertain_px": int(uncertain.sum()),
        "watershed_unresolved_px": int((uncertain & (markers <= 0)).sum()),
        "changed_px": int((before != after).sum()),
        "markers_are": "palette_label+1 (독립 면 ID 아님)",
    }

    # ---------- B: 검출된 선 연속성 (스펀지밥 몸통 테두리 샘플) ----------
    # 스펀지밥 몸통 대략 bbox (작업좌표) — preview 크롭 기준 근사
    report["line_layer"] = {
        "line_px": int(line_mask.sum()),
        "ink_px": int(ink.sum()),
        "line_as_frac_of_ink": round(float(line_mask.sum() / max(ink.sum(), 1)), 2),
    }

    # merge_small_regions 후 최종
    t0 = time.time()
    rm2, rl2 = merge_small_regions(region_map, region_labels, palette, min_area=400)
    report["merge_small_time_s"] = round(time.time() - t0, 2)
    report["regions_after_merge_small"] = int(len(rl2))

    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _boundary_pixels(region_map: np.ndarray) -> dict:
    """인접 영역쌍 -> 경계에 놓인 픽셀 좌표 목록 (4-이웃 crack)."""
    out: dict = {}
    h, w = region_map.shape
    for (dy, dx) in ((0, 1), (1, 0)):
        a = region_map[: h - dy, : w - dx]
        b = region_map[dy:, dx:]
        mask = (a != b) & (a >= 0) & (b >= 0)
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys, xs):
            k1, k2 = int(a[y, x]), int(b[y, x])
            key = (min(k1, k2), max(k1, k2))
            out.setdefault(key, []).append((y, x))
    return out


if __name__ == "__main__":
    main()
