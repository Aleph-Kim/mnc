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
from app.imaging.render import render_outline_image, render_preview_image
from app.imaging.segment import merge_small_regions, segment_regions


@dataclass
class DesignResult:
    outline_image_path: Path
    preview_image_path: Path
    regions: list[Region]


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
) -> DesignResult:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"could not read image at {image_path}")
    rgb = cv2.cvtColor(_resize_to_working_dim(bgr, process_max_dim), cv2.COLOR_BGR2RGB)

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    if settings.debug_pipeline:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # 일러스트는 원화 선을 별도 레이어로 분리 — 선은 칠할 면이 아니라 인쇄되는 선
    line_mask = None
    if mode == "illustration":
        line_mask, dark_mask = detect_line_layer(rgb)

    trace: list = []
    label_map, palette = quantize_colors(rgb, color_count, trace=trace)
    if settings.debug_pipeline:
        _debug.dump_label_map(debug_dir / "01_quantized.png", label_map, palette)
        _debug.dump_palette_trace(debug_dir / "palette_trace.json", trace)
    label_map = denoise_label_map(label_map, palette)
    if settings.debug_pipeline:
        _debug.dump_label_map(debug_dir / "02_denoised.png", label_map, palette)

    if line_mask is not None:
        before = label_map
        label_map = reassign_uncertain(label_map, line_mask, rgb)
        if settings.debug_pipeline:
            _debug.dump_line_layer(
                debug_dir / "06_line_layer.png", rgb, dark_mask, line_mask
            )
            _debug.dump_boundary(
                debug_dir / "07_boundary.png", before, label_map, palette
            )

    region_map, region_labels = segment_regions(label_map)
    regions_after_segment = len(region_labels)
    if settings.debug_pipeline:
        _debug.dump_regions(debug_dir / "03_regions_raw.png", region_map)
    region_map, region_labels = merge_small_regions(
        region_map, region_labels, palette, min_area=min_region_area
    )
    if settings.debug_pipeline:
        _debug.dump_regions(debug_dir / "04_regions_merged.png", region_map)
    region_labels, palette = drop_unused_colors(region_labels, palette)

    contours_by_region = extract_contours(region_map)
    seams = extract_seams(region_map)
    regions = assign_numbers(region_map, region_labels, palette, contours_by_region)

    preview_path = output_dir / "preview.png"
    outline_path = output_dir / "outline.png"

    render_preview_image(region_map, region_labels, palette, line_mask).save(preview_path)
    outline_image, drawn_ids = render_outline_image(
        rgb.shape[:2], seams, regions, palette, line_mask
    )
    outline_image.save(outline_path)

    if settings.debug_pipeline:
        _debug.dump_number_overlay(
            debug_dir / "05_number_overlay.png", rgb.shape[:2], seams, regions
        )
        _debug.dump_summary(
            debug_dir / "summary.json",
            {
                "segment": regions_after_segment,
                "merge": len(region_labels),
                "final": len(regions),
            },
            region_map,
            regions,
            palette,
            drawn_ids,
        )

    return DesignResult(
        outline_image_path=outline_path,
        preview_image_path=preview_path,
        regions=regions,
    )
