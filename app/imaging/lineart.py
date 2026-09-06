import cv2
import numpy as np

from app.imaging.quantize import to_lab

# 원화 잉크 선은 국소적으로 가장 어둡고(L 낮음) 가장 무채색(chroma 낮음)인 자리다.
# L+chroma에 black-hat을 걸면 "주변 면보다 얼마나 어둡고 탁한가"가 나온다. JPEG·확대로
# 검은 선이 배경(청록)과 같은 밝기까지 흐려져도, 채도가 떨어지므로 이 지표로는 잡힌다.
# 잠정값 — octocat/clover/spongebob 스윕으로 확정 필요

# black-hat 커널 반지름. 가장 굵은 외곽선 폭보다 커야 그 선을 통째로 잡는다
INK_BLACKHAT_RADIUS = 15

# black-hat 값(L+chroma 단위)이 이보다 커야 선으로 본다
INK_CONTRAST = 12.0

# 선은 무채색에 가깝다. 채도가 이보다 높으면 유색 선/짙은 채도색이라 이번엔 제외
CHROMA_MAX = 26.0

# 이 반지름 원이 들어가는 어두운 덩어리는 선이 아니라 면 (동공·신발)
MAX_LINE_HALFWIDTH = 2

# 면으로 뺄 덩어리의 최소 넓이. 벨트·눈썹처럼 촘촘한 선 뭉치가 opening을 통과해도
# 넓이가 작아 여기서 걸리고 선으로 남는다
MIN_FILL_AREA = 120

# 이보다 작은 부스러기(JPEG 잡티)는 선으로 치지 않는다
MIN_LINE_AREA = 25


def detect_line_layer(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """원화의 검은 선을 별도 마스크로 뽑는다.

    (선 마스크, 선/면 분리 전 잉크 마스크)를 돌려준다 — 두 번째는 디버그에서 어떤
    덩어리가 면으로 빠졌는지 보기 위함.
    """
    h, w = rgb.shape[:2]
    lab = to_lab(rgb.reshape(-1, 3)).reshape(h, w, 3)
    chroma = np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2).astype(np.float32)
    inkiness = np.clip((lab[:, :, 0].astype(np.float32) + chroma) * 2, 0, 255).astype(np.uint8)

    radius = INK_BLACKHAT_RADIUS
    blackhat = cv2.morphologyEx(
        inkiness,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    ).astype(np.float32) / 2.0
    ink = ((blackhat > INK_CONTRAST) & (chroma < CHROMA_MAX)).astype(np.uint8)

    # opening으로 이 반지름 원이 들어가는 자리(동공·신발)만 남긴다. 연결요소 단위로
    # 빼면 굵은 모서리 하나 때문에 외곽선 전체가 면으로 사라진다
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * MAX_LINE_HALFWIDTH + 1, 2 * MAX_LINE_HALFWIDTH + 1)
    )
    opened = cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel), np.ones((3, 3), np.uint8))
    is_fill = _keep_large(opened.astype(bool), MIN_FILL_AREA)

    line = ink.astype(bool) & ~is_fill
    line = cv2.morphologyEx(
        line.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)
    return _drop_small(line, MIN_LINE_AREA), ink.astype(bool)


def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    return _keep_large(mask, min_area)


def _keep_large(mask: np.ndarray, min_area: int) -> np.ndarray:
    _, components, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[components]
