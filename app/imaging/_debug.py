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
    path: Path, shape: tuple[int, int], seams: list[np.ndarray], regions: list[Region]
) -> None:
    # 윤곽선 위에 각 영역의 번호 위치·여유 반지름·id를 겹쳐 찍는다. "빈 곳에 찍힌 번호"가
    # 거대 영역의 정상 배치인지 좌표 버그인지, 같은 번호 중복이 과분할 탓인지 확인용
    h, w = shape
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for seam in seams:
        if len(seam) >= 2:
            draw.line([tuple(int(v) for v in p) for p in seam], fill="black", width=1)

    for region in regions:
        cx, cy = region.label_anchor
        r = region.label_radius
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="red")
        draw.text((cx + 2, cy + 2), f"{region.number}#{region.id}", fill="blue", font=font)

    canvas.save(path)


def dump_summary(
    path: Path,
    stages: dict[str, int],
    region_map: np.ndarray,
    regions: list[Region],
    palette: np.ndarray,
    drawn_ids: list[int],
) -> None:
    areas = np.bincount(region_map[region_map >= 0].ravel())
    drawn = set(drawn_ids)

    per_region = []
    for region in regions:
        pts = np.array(region.points)
        x0, y0 = pts.min(axis=0).tolist()
        x1, y1 = pts.max(axis=0).tolist()
        per_region.append(
            {
                "id": region.id,
                "number": region.number,
                "color_hex": region.color_hex,
                "area": int(areas[region.id]) if region.id < len(areas) else 0,
                "bbox": [x0, y0, x1, y1],
                "anchor": list(region.label_anchor),
                "radius": round(region.label_radius, 1),
                "number_drawn": region.id in drawn,
            }
        )

    distances = palette_distances(palette)
    off_diag = distances[~np.eye(len(palette), dtype=bool)]

    summary = {
        "region_count": stages,
        "palette_size": len(palette),
        "palette_min_dE": round(float(off_diag.min()), 2) if off_diag.size else None,
        "palette_dE_matrix": np.round(distances, 1).tolist(),
        "numbers_skipped": sum(1 for r in per_region if not r["number_drawn"]),
        "regions": per_region,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
