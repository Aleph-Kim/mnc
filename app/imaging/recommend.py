import cv2
import numpy as np
from sklearn.cluster import KMeans

MIN_K = 5
MAX_K = 30
DOWNSAMPLE_MAX_DIM = 180
SAMPLE_PIXELS = 5000


def recommend_color_count(image: np.ndarray) -> int:
    h, w = image.shape[:2]
    scale = DOWNSAMPLE_MAX_DIM / max(h, w)
    if scale < 1:
        image = cv2.resize(
            image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    pixels = image.reshape(-1, 3).astype(np.float32)
    if pixels.shape[0] > SAMPLE_PIXELS:
        idx = np.random.choice(pixels.shape[0], SAMPLE_PIXELS, replace=False)
        pixels = pixels[idx]

    ks = list(range(MIN_K, MAX_K + 1))
    inertias = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=4, random_state=0)
        kmeans.fit(pixels)
        inertias.append(kmeans.inertia_)

    return _find_elbow(ks, inertias)


def _find_elbow(ks: list[int], inertias: list[float]) -> int:
    points = np.array(list(zip(ks, inertias)), dtype=np.float64)
    first, last = points[0], points[-1]
    line_vec = last - first
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    distances = []
    for point in points:
        vec_from_first = point - first
        proj_len = np.dot(vec_from_first, line_vec_norm)
        proj_point = first + proj_len * line_vec_norm
        distances.append(float(np.linalg.norm(point - proj_point)))

    elbow_idx = int(np.argmax(distances))
    return int(np.clip(ks[elbow_idx], MIN_K, MAX_K))
