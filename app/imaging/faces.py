"""Build paintable faces, then extend their ownership to stroke centre lines."""
import cv2
import numpy as np
from scipy import ndimage
from app.imaging.segment import segment_regions


def finalize_faces(
    region_map: np.ndarray, region_labels: np.ndarray, line_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = region_labels[region_map].astype(np.int32)
    labels[line_mask] = -1
    faces, colors = segment_regions(labels)
    area = np.bincount(faces.ravel(), minlength=len(colors))
    # Measure thickness per face, not just distance to ink.
    field = faces.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    core = cv2.erode(field, kernel) == cv2.dilate(field, kernel)
    core_count = np.bincount(faces[core], minlength=len(colors))
    away = cv2.distanceTransform((~line_mask).astype(np.uint8), cv2.DIST_L2, 5) > 4
    away_count = np.bincount(faces[away], minlength=len(colors))
    keep = (colors >= 0) & (area >= 80) & ((core_count > 0) | (away_count >= 40))
    if not keep.any():
        keep = colors >= 0
    if not keep.any():
        return region_map, region_labels
    missing = ~keep[faces]
    # A deleted sliver has no independent paint identity. Voronoi extension of
    # surviving faces supplies a single shared boundary inside the ink strip.
    indices = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
    faces[missing] = faces[tuple(indices[:, missing])]
    used, compact = np.unique(faces, return_inverse=True)
    return compact.reshape(faces.shape).astype(np.int32), colors[used]
