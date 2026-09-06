import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.imaging import _debug
from app.imaging.boundary import reassign_uncertain
from app.imaging.contours import extract_contours, extract_seams
from app.imaging.denoise import denoise_label_map
from app.imaging.lineart import detect_line_layer
from app.imaging.numbering import Region, assign_numbers
from app.imaging.quantize import drop_unused_colors, quantize_colors
from app.imaging.regionmerge import merge_by_edge_evidence
from app.imaging.render import render_outline_image, render_preview_image
from app.imaging.segment import merge_small_regions, segment_regions


@dataclass
class DesignResult:
    outline_image_path: Path
    preview_image_path: Path
    regions: list[Region]


# ablation 토큰 — 단계별 효과 분리 검증용 (scripts/ablation.py)
ABLATION_TOKENS = {"no_edge_merge", "no_line_layer", "legacy_palette"}


def _resize_to_working_dim(image: np.ndarray, target_dim: int) -> np.ndarray:
    # denoise/segment/contour의 튜닝값(커널 크기, 최소 영역, epsilon)이 전부
    # target_dim급 해상도를 전제한 절대 픽셀값이라, 작은 원본을 그대로 두면
    # 세부 형태가 뭉개진다. 축소뿐 아니라 확대도 해서 항상 같은 작업
    # 해상도로 맞춘다.
    h, w = image.shape[:2]
    scale = target_dim / max(h, w)
    if scale == 1:
        return image
    # 확대에 NEAREST를 쓰면 원본 픽셀이 그대로 블록이 되어 모든 경계가 계단으로 남는다.
    # 확대 보간이 만드는 중간색은 팔레트 후보에서 빠지므로(quantize의 균일 조각 선별)
    # 여기서는 매끄러운 보간을 쓴다.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=interpolation)


def generate_design(
    image_path: Path,
    color_count: int,
    output_dir: Path,
    mode: str = "illustration",
    process_max_dim: int = 1200,
    min_region_area: int = 400,
    ablation: frozenset[str] = frozenset(),
) -> DesignResult:
    started = time.time()
    timings: dict[str, float] = {}

    def tick(name: str, t0: float) -> None:
        timings[name] = round(time.time() - t0, 3)

    raw = image_path.read_bytes()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read image at {image_path}")
    rgb = cv2.cvtColor(_resize_to_working_dim(bgr, process_max_dim), cv2.COLOR_BGR2RGB)

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug = settings.debug_pipeline
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # 일러스트는 원화 선을 별도 레이어로 분리 — 선은 칠할 면이 아니라 인쇄되는 선
    line_mask = None
    boundary_stats: dict = {}
    if mode == "illustration" and "no_line_layer" not in ablation:
        t0 = time.time()
        line_mask, ink_mask = detect_line_layer(rgb)
        tick("line_detect", t0)

    t0 = time.time()
    trace: list = []
    label_map, palette = quantize_colors(
        rgb, color_count, trace=trace, legacy_pick="legacy_palette" in ablation
    )
    tick("quantize", t0)
    if debug:
        _debug.dump_label_map(debug_dir / "01_quantized.png", label_map, palette)
        _debug.dump_palette_trace(debug_dir / "palette_trace.json", trace)

    t0 = time.time()
    label_map = denoise_label_map(label_map, palette)
    tick("denoise", t0)
    if debug:
        _debug.dump_label_map(debug_dir / "02_denoised.png", label_map, palette)

    if line_mask is not None:
        t0 = time.time()
        before = label_map
        label_map, boundary_stats = reassign_uncertain(label_map, line_mask, rgb)
        tick("boundary_reassign", t0)
        if debug:
            _debug.dump_line_layer(debug_dir / "06_line_layer.png", rgb, ink_mask, line_mask)
            _debug.dump_boundary(debug_dir / "07_boundary.png", before, label_map, palette)

    t0 = time.time()
    region_map, region_labels = segment_regions(label_map)
    regions_after_segment = len(region_labels)
    tick("segment", t0)
    if debug:
        _debug.dump_regions(debug_dir / "03_regions_raw.png", region_map)

    merge_log: list[dict] = []
    if "no_edge_merge" not in ablation:
        t0 = time.time()
        region_map, region_labels, merge_log = merge_by_edge_evidence(
            region_map, region_labels, rgb, line_mask
        )
        tick("edge_merge", t0)
        if debug:
            _debug.dump_regions(debug_dir / "03b_regions_edge_merged.png", region_map)
            _debug.dump_merge_log(debug_dir / "merge_log.json", merge_log)
    regions_after_edge_merge = len(region_labels)

    t0 = time.time()
    region_map, region_labels = merge_small_regions(
        region_map, region_labels, palette, min_area=min_region_area
    )
    tick("merge_small", t0)
    if debug:
        _debug.dump_regions(debug_dir / "04_regions_merged.png", region_map)
    region_labels, palette = drop_unused_colors(region_labels, palette)

    t0 = time.time()
    contours_by_region = extract_contours(region_map)
    seams = extract_seams(region_map)
    regions = assign_numbers(region_map, region_labels, palette, contours_by_region, line_mask)
    tick("contours_numbering", t0)

    preview_path = output_dir / "preview.png"
    outline_path = output_dir / "outline.png"

    render_preview_image(region_map, region_labels, palette, line_mask).save(preview_path)
    outline_image, number_log = render_outline_image(
        rgb.shape[:2], seams, regions, palette, line_mask
    )
    outline_image.save(outline_path)
    tick("total", started)

    if debug:
        placed = {e["id"] for e in number_log if e["placed"]}
        _debug.dump_number_overlay(
            debug_dir / "05_number_overlay.png", rgb.shape[:2], seams, regions, placed
        )
        _debug.dump_summary(
            debug_dir / "summary.json",
            {
                "segment": regions_after_segment,
                "edge_merge": regions_after_edge_merge,
                "merge_small": len(region_labels),
                "final": len(regions),
            },
            region_map, regions, palette, number_log,
            extra={"boundary": boundary_stats,
                   "edge_merges_applied": sum(1 for c in merge_log if c.get("merge"))},
        )
        _write_manifest(debug_dir / "manifest.json", {
            "input_md5": hashlib.md5(raw).hexdigest(),
            "git_rev": _git_rev(),
            "params": {"color_count": color_count, "mode": mode,
                       "process_max_dim": process_max_dim, "min_region_area": min_region_area,
                       "ablation": sorted(ablation)},
            "working_size": list(rgb.shape[:2]),
            "timings_s": timings,
            "palette_size": len(palette),
            "region_counts": {"segment": regions_after_segment,
                              "edge_merge": regions_after_edge_merge,
                              "final": len(regions)},
        })

    return DesignResult(
        outline_image_path=outline_path,
        preview_image_path=preview_path,
        regions=regions,
    )


def _git_rev() -> str:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL
        ) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _write_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
