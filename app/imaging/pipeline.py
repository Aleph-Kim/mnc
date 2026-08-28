from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.imaging.contours import extract_contours, extract_seams
from app.imaging.denoise import denoise_label_map
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
    process_max_dim: int = 1200,
    min_region_area: int = 400,
) -> DesignResult:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"could not read image at {image_path}")
    rgb = cv2.cvtColor(_resize_to_working_dim(bgr, process_max_dim), cv2.COLOR_BGR2RGB)

    label_map, palette = quantize_colors(rgb, color_count)
    label_map = denoise_label_map(label_map, palette)

    region_map, region_labels = segment_regions(label_map)
    region_map, region_labels = merge_small_regions(
        region_map, region_labels, palette, min_area=min_region_area
    )
    region_labels, palette = drop_unused_colors(region_labels, palette)

    contours_by_region = extract_contours(region_map)
    seams = extract_seams(region_map)
    regions = assign_numbers(region_map, region_labels, palette, contours_by_region)

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "preview.png"
    outline_path = output_dir / "outline.png"

    render_preview_image(region_map, region_labels, palette).save(preview_path)
    render_outline_image(rgb.shape[:2], seams, regions, palette).save(outline_path)

    return DesignResult(
        outline_image_path=outline_path,
        preview_image_path=preview_path,
        regions=regions,
    )
