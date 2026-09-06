"""파이프라인 단계별 중간 결과 덤프. 진단 전용이라 파이프라인 로직에는 영향이 없다."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.imaging.numbering import Region
from app.imaging.quantize import palette_distances


def dump_label_map(path: Path, label_map: np.ndarray, palette: np.ndarray) -> None:
    Image.fromarray(palette[label_map], mode="RGB").save(path)


def dump_regions(path: Path, region_map: np.ndarray) -> None:
    # 영역 id를 시드 고정 난수색으로 매핑, 인접 영역이 서로 다른 색으로 보여
    # 과분할·잔조각을 눈으로 셀 수 있게 한다
    count = max(int(region_map.max()) + 1, 1)
    lut = np.random.default_rng(0).integers(40, 240, size=(count, 3), dtype=np.uint8)

    canvas = np.zeros((*region_map.shape, 3), dtype=np.uint8)
    painted = region_map >= 0
    canvas[painted] = lut[region_map[painted]]
    Image.fromarray(canvas, mode="RGB").save(path)


def dump_number_overlay(
    path: Path,
    shape: tuple[int, int],
    seams: list[np.ndarray],
    regions: list[Region],
    placed: set[int],
) -> None:
    # 실제로 찍힌 번호(파랑)와 생략된 번호(빨강, 사유는 number_log.json)를 윤곽선 위에 겹친다
    h, w = shape
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for seam in seams:
        if len(seam) >= 2:
            draw.line([tuple(int(v) for v in p) for p in seam], fill="black", width=1)

    for region in regions:
        cx, cy = region.label_anchor
        colour = "blue" if region.id in placed else "red"
        draw.text((cx + 2, cy + 2), f"{region.number}#{region.id}", fill=colour, font=font)

    canvas.save(path)


def dump_line_layer(
    path: Path, rgb: np.ndarray, dark_mask: np.ndarray, line_mask: np.ndarray
) -> None:
    # 원본을 흐리게 깔고 선(빨강)과 면으로 빠진 어두운 덩어리(초록)를 겹쳐 분리 결과를 본다
    canvas = (rgb.astype(np.float32) * 0.3 + 255 * 0.7).astype(np.uint8)
    canvas[dark_mask & ~line_mask] = (0, 170, 0)
    canvas[line_mask] = (220, 0, 0)
    Image.fromarray(canvas, mode="RGB").save(path)


def dump_boundary(
    path: Path,
    before: np.ndarray,
    after: np.ndarray,
    palette: np.ndarray,
) -> None:
    # 왼쪽=재배정 전, 오른쪽=재배정 후(바뀐 픽셀 자홍). 실루엣 halo가 사라졌는지 확인
    left = palette[before]
    right = palette[after].copy()
    right[before != after] = (255, 0, 255)
    Image.fromarray(np.concatenate([left, right], axis=1), mode="RGB").save(path)


def dump_palette_trace(path: Path, trace: list) -> None:
    path.write_text(
        json.dumps(
            [{"stage": stage, "colors": colors} for stage, colors in trace],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def dump_merge_log(path: Path, merge_log: list[dict]) -> None:
    merged = [c for c in merge_log if c.get("merge")]
    rejected = [c for c in merge_log if not c.get("merge")]
    reasons: dict[str, int] = {}
    for c in rejected:
        reasons[c.get("reject_reason") or "?"] = reasons.get(c.get("reject_reason") or "?", 0) + 1
    path.write_text(
        json.dumps(
            {"candidates": len(merge_log), "merged": len(merged),
             "rejected_by_reason": reasons, "detail": merge_log},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


def dump_summary(
    path: Path,
    stages: dict[str, int],
    region_map: np.ndarray,
    regions: list[Region],
    palette: np.ndarray,
    number_log: list[dict],
    extra: dict | None = None,
) -> None:
    areas = np.bincount(region_map[region_map >= 0].ravel())
    by_id = {e["id"]: e for e in number_log}

    per_region = []
    for region in regions:
        pts = np.array(region.points)
        x0, y0 = pts.min(axis=0).tolist()
        x1, y1 = pts.max(axis=0).tolist()
        entry = by_id.get(region.id, {})
        per_region.append(
            {
                "id": region.id,
                "number": region.number,
                "color_hex": region.color_hex,
                "area": int(areas[region.id]) if region.id < len(areas) else 0,
                "bbox": [x0, y0, x1, y1],
                "anchor": list(region.label_anchor),
                "radius": round(region.label_radius, 1),
                "number_drawn": entry.get("placed", False),
                "number_kind": entry.get("kind"),
                "number_reason": entry.get("reason"),
            }
        )

    distances = palette_distances(palette)
    off_diag = distances[~np.eye(len(palette), dtype=bool)]
    reasons: dict[str, int] = {}
    for r in per_region:
        if not r["number_drawn"]:
            reasons[r["number_reason"] or "?"] = reasons.get(r["number_reason"] or "?", 0) + 1

    summary = {
        "region_count": stages,
        "palette_size": len(palette),
        "palette_min_dE": round(float(off_diag.min()), 2) if off_diag.size else None,
        "palette_dE_matrix": np.round(distances, 1).tolist(),
        "numbers_placed": sum(1 for r in per_region if r["number_drawn"]),
        "numbers_internal": sum(1 for r in per_region if r["number_kind"] == "internal"),
        "numbers_external": sum(1 for r in per_region if r["number_kind"] == "external"),
        "numbers_skipped": sum(1 for r in per_region if not r["number_drawn"]),
        "numbers_skipped_by_reason": reasons,
        **(extra or {}),
        "regions": per_region,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
