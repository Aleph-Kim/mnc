import cv2
import numpy as np

from app.imaging.quantize import MIN_CONTENT_DISTANCE, palette_distances


def denoise_label_map(
    label_map: np.ndarray,
    palette: np.ndarray,
    kernel_size: int = 9,
    max_shift: float = MIN_CONTENT_DISTANCE,
) -> np.ndarray:
    """양자화가 만든 너덜너덜한 경계만 다듬는다.

    이 단계가 필요한 이유는 그라데이션이다. 하늘처럼 매끄럽게 변하는 면을 k색으로
    자르면 원본에 없던 띠 경계가 생기는데, 그 경계는 양자화 잡음을 따라 픽셀 단위로
    흔들려서 윤곽선이 실선이 아니라 지저분한 띠로 그려진다.

    그런데 중앙값 필터는 창 안의 다수결일 뿐이라 그 자리에 무엇이 있었는지 모른다.
    9px 창은 폭 7px짜리 잎맥이나 고양이 수염도 통계적으로 눌러버린다. 그래서 필터
    결과를 그대로 받지 않고, "색이 얼마나 뛰었는가"로 채택 여부를 가른다. 양자화가
    만든 띠 경계는 원래 이웃한 색끼리의 작은 단차라 바뀌어도 색이 조금만 움직이지만,
    작가가 그려 넣은 선을 지우려면 색이 크게 튀어야 하기 때문이다.
    """
    # label 값은 KMeans 클러스터 번호일 뿐 색 순서가 아니라서, 라벨맵에 직접
    # medianBlur를 걸면 "숫자상 중앙값"이 실제로는 그 자리와 무관한 라벨을
    # 골라버릴 수 있다. 실제 색 공간에서 블러링한 뒤 가장 가까운 팔레트
    # 색으로 다시 스냅해서 그 문제를 피한다.
    color_img = palette[label_map].astype(np.uint8)
    blurred = cv2.medianBlur(color_img, kernel_size)

    # 픽셀마다 팔레트 전체와의 거리를 쌓으면 (높이 × 너비 × 색수 × 3) 배열이 되어
    # 1200px·30색에서 1.4GB를 잡는다. 블러 결과에 실제로 나타나는 색은 몇 천 개뿐이라
    # 고유색에 대해서만 계산하고 되돌린다.
    colors, inverse = np.unique(blurred.reshape(-1, 3), axis=0, return_inverse=True)
    diff = colors[:, None, :].astype(np.int32) - palette[None, :, :].astype(np.int32)
    nearest = np.argmin(np.sum(diff * diff, axis=-1), axis=-1).astype(np.uint8)
    smoothed = nearest[inverse].reshape(label_map.shape)

    shift = palette_distances(palette)[label_map, smoothed]
    return np.where(shift < max_shift, smoothed, label_map).astype(label_map.dtype)
