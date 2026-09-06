import cv2
import numpy as np
from sklearn.cluster import KMeans

# 두 팔레트 색이 이보다 가까우면 인쇄된 색상표에서 사람이 구분하지 못한다.
# (CIELAB dE76 기준)
#
# 전체 업로드 스윕(k=12, dE76): 옥토캣 안티에일리어싱 잔여색이 흰색과 4.31(중복 업로드
# 5건 전부 동일), 클로버 고양이의 가장 가까운 진짜 잎사귀 색 쌍이 3.98(둘 다 수만~수십만
# 픽셀짜리 큰 면적이라 사람 눈엔 어차피 같은 초록으로 보임), 그다음이 5.2(산/노을/구름
# 사진 다수) · 9.25 · 14.92. 4.31~5.2 사이가 빈 구간이라 그 안에서 잡았다.
MIN_PALETTE_DISTANCE = 4.6

# 이보다 작은 색 단차는 "원본에 있던 색"이 아니라 양자화가 만든 경계로 본다.
#
# 부드러운 그라데이션을 k색으로 자르면 실제로는 없던 띠 경계가 생기고, 그 경계는
# 양자화 잡음을 따라 너덜너덜하게 흔들린다. 반대로 원본에 진짜 있는 경계(잎맥, 수염,
# 로고 실루엣)는 색이 뚜렷하게 튀고 경계도 깔끔하게 떨어진다. 즉 "색이 얼마나
# 뛰는가"가 지워도 되는 경계와 작가가 그린 선을 가르는 기준이 된다.
#
# 실측(uploads 전체, dE76): 그라데이션 사진에서 양자화가 만든 단차는 최대 11.8,
# 반대로 클로버 잎맥 15.7 · 고양이 눈코 최대 67.9 · 라인아트 16.9~54.3. 그 사이
# 빈 구간의 가운데를 잡았다.
MIN_CONTENT_DISTANCE = 13.5

# 이 기울기(픽셀당 dE76)보다 완만한 자리의 색만 팔레트 후보로 쓴다.
#
# 안티에일리어싱과 확대 보간이 만드는 중간색은 원본에 없는 색이다. 그 색이 팔레트에
# 들어가면 실루엣을 따라 도는 가짜 띠가 되는데, 세 라운드 동안 사후 휴리스틱으로
# 잡으려던 그 링이 애초에 팔레트에 그 색이 없으면 생길 수가 없다.
#
# 예전에는 "슈퍼픽셀 내부의 색 분산이 작은가"로 이 색들을 걸렀다. 그런데 창 안 분산은
# 창 크기에 딸린 값이라 기준이 될 수 없다 — 실측(합성 램프): 폭 40px 램프 위에서
# 10x10 창의 분산은 7.29로 "균일" 판정을 받지만 30x30 창은 22.07로 걸린다. 그래서
# 램프보다 창을 키우는 수밖에 없었고(SUPERPIXEL_AREA=900, 즉 30x30), 그 큰 창은
# 곧 "30x30보다 가는 것은 색을 가질 수 없다"는 뜻이 되어 고양이 줄무늬가 통째로
# 사라졌다(실측: 줄무늬 안에 80% 이상 들어앉는 조각이 900 면적에서 0개).
#
# 기울기는 창 크기와 무관하다. 폭 W 램프의 기울기는 어느 자리에서 재도 dE/W다.
# 실측: 옥토캣의 안티에일리어싱 중간톤 36,643px의 기울기가 15.6~27.6(10~90 백분위)이라
# 3.0에서 단 하나도 살아남지 못하고, 같은 값에서 고양이 줄무늬는 심(core)이 남는다.
FLAT_GRADIENT = 3.0

# 후보 색을 추리는 중간 단계의 색 수.
#
# 팔레트를 고르는 greedy는 후보끼리의 거리표를 통째로 쥐어야 해서 후보 수의 제곱으로
# 무거워진다. 평탄한 픽셀 수십만 개를 그대로 넣을 수 없으니 먼저 이 수만큼으로 뭉친다.
# 실측(128/256, 시드 3개): 결과 팔레트가 사실상 같고 256은 시간만 3배가 되어 128을 썼다.
ATLAS_COLORS = 128

# 후보를 뭉칠 때 KMeans에 넣는 표본 수 상한.
#
# 실측: 전량(수십만)을 넣으면 이미지당 7초, 이 표본 수에서는 1.4초인데 팔레트는 같다.
ATLAS_SAMPLE = 60000


def quantize_colors(
    image: np.ndarray, k: int, trace: list | None = None, legacy_pick: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """이미지에 실제로 있는 색만 골라 팔레트를 만들고 k색으로 나눈다.

    두 가지를 따로 정한다. 어떤 색이 후보가 될 자격이 있는가(평탄한 자리의 색만)와,
    후보 중 무엇에 번호를 줄 것인가(가려서 대신할 수 없는 색부터)이다.

    후보 자격은 기울기로 가른다. 경계의 안티에일리어싱 그라데이션은 원본에 없는
    혼합색이고, 그 색이 팔레트에 들어가면 실루엣을 따라 도는 가짜 띠가 된다.

    번호를 줄 색은 오차 최소화로 고르지 않는다. KMeans는 제곱오차를 줄이는데, 그건
    "그림을 얼마나 비슷하게 재현하는가"의 기준이지 "칠하는 사람에게 어떤 물감이
    필요한가"의 기준이 아니다. 넓은 잎사귀의 초록을 dE 5로 쪼개는 쪽이 제곱오차는
    훨씬 많이 줄지만, 그 5는 사람이 보지 못하는 차이라 물감 한 통을 쓸 값어치가 없다.
    반대로 그림 0.36%를 차지하는 고양이 줄무늬는 제곱오차 기여가 미미해도 그 색이
    없으면 줄무늬 자체가 사라진다. 그래서 '보이는 오차'(MIN_PALETTE_DISTANCE를 넘는
    부분만)를 줄이는 순서로, 그중에서도 대신할 색이 아예 없는 것부터 고른다.
    """
    height, width = image.shape[:2]
    pixels = to_lab(image.reshape(-1, 3))

    flat = pixels[_flat_mask(image).ravel()]
    # 그림 전체가 경계일 수는 없지만, 표본이 마르면 기준을 포기하고 전부 쓴다.
    if len(flat) < max(k, ATLAS_COLORS):
        flat = pixels

    atlas, weights = _atlas(flat)
    centers = _pick_palette_legacy(atlas, weights, k) if legacy_pick else _pick_palette(atlas, weights, k)
    _record(trace, "pick", centers)
    centers = _refine(atlas, weights, centers)
    _record(trace, "refine", centers)
    centers = _merge_indistinct(atlas, weights, centers)
    _record(trace, "merge_indistinct", centers)

    label_map = _nearest(pixels, centers).reshape(height, width)
    used = np.unique(label_map)
    if len(used) < len(centers):
        lookup = np.zeros(len(centers), dtype=np.uint8)
        lookup[used] = np.arange(len(used))
        label_map, centers = lookup[label_map], centers[used]

    palette = np.clip(np.round(to_rgb(centers)), 0, 255).astype(np.uint8)
    _record(trace, "final", centers)
    return label_map, palette


def _record(trace: list | None, stage: str, centers: np.ndarray) -> None:
    # 팔레트 색이 pick→refine→merge에서 얼마나 움직이는지 추적 (흰자 탁해짐 진단용)
    if trace is not None:
        trace.append((stage, np.round(to_rgb(centers)).astype(int).tolist()))


def _flat_mask(image: np.ndarray) -> np.ndarray:
    """색이 완만하게 놓인 자리만 True. 경계를 건너는 그라데이션은 여기서 빠진다."""
    lab = to_lab(image.reshape(-1, 3)).reshape(*image.shape[:2], 3).astype(np.float32)
    dx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    # Sobel 3x3 커널의 합이 4라 픽셀당 dE로 읽으려면 4로 나눠야 한다
    gradient = np.sqrt((dx**2 + dy**2).sum(axis=2)) / 4.0
    return gradient < FLAT_GRADIENT


def _atlas(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """평탄한 픽셀을 ATLAS_COLORS개의 후보색과 그 넓이로 뭉친다.

    격자로 자르지 않고 KMeans를 쓰는 건, 격자는 색 뭉치를 칸 경계에서 임의로 쪼개기
    때문이다. 고양이 줄무늬처럼 표본이 수백 픽셀뿐인 색은 그렇게 쪼개지면 조각마다
    무게가 바닥나 후보에서 사라진다.
    """
    step = max(1, len(flat) // ATLAS_SAMPLE)
    sample = flat[::step]
    count = min(ATLAS_COLORS, len(np.unique(sample, axis=0)))
    kmeans = KMeans(n_clusters=count, random_state=0, n_init=3).fit(sample)

    labels = _nearest(flat, kmeans.cluster_centers_)
    weights = np.bincount(labels, minlength=count).astype(np.float64)
    sums = np.stack(
        [np.bincount(labels, weights=flat[:, c], minlength=count) for c in range(3)], axis=1
    )
    alive = weights > 0
    return sums[alive] / weights[alive, None], weights[alive]


def _pick_palette(atlas: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """대신할 색이 없는 후보부터 팔레트에 넣는다.

    두 바퀴를 돈다. 첫 바퀴의 기준선은 MIN_CONTENT_DISTANCE로, "이만큼 떨어져 있으면
    이웃 색으로 대신 칠할 수 없다"는 뜻이다. 여기서 걸리는 색은 팔레트에 없으면
    그림에서 통째로 사라지므로 먼저 자리를 준다. 두 번째 바퀴는 남은 자리로 이미
    대신할 색이 있는 자리의 오차를 줄인다 — 하늘 사진처럼 매끄러운 그라데이션은
    첫 바퀴가 다섯 색이면 다 덮어버려서, 사용자가 요청한 k색을 채우려면 이 바퀴가 있어야
    한다(실측: 구름 사진의 6~11번째 색이 전부 여기서 나온다).

    1라운드는 면적 가중 gain을 쓰지 않는다. gain을 쓰면 그림의 0.67%뿐인 흰자·치아·
    셔츠가 매 라운드 더 넓은 배경 그라데이션 색에 밀려 아예 안 뽑힌다(실측: spongebob
    k=11에서 밝은 색이 하나도 안 들어오고 흰자가 khaki로 표현됨). 대신 "대신할 색이
    없고(모든 선택색과 MIN_CONTENT_DISTANCE 밖) 잡티가 아닌(noise_floor 초과)" 후보를
    면적 큰 순으로 채운다.

    2라운드는 그대로 남은 자리로 '보이는 오차'(MIN_PALETTE_DISTANCE 초과분)를 줄인다.
    이미 고른 색과 MIN_PALETTE_DISTANCE 안에 드는 후보는 어느 라운드든 못 고른다.
    """
    distances = np.linalg.norm(atlas[:, None] - atlas[None, :], axis=-1)
    nearest = np.full(len(atlas), np.inf)
    chosen: list[int] = []

    noise_floor = max(50.0, 0.002 * float(weights.sum()))
    while len(chosen) < k:
        eligible = (nearest > MIN_CONTENT_DISTANCE) & (weights > noise_floor)
        eligible[chosen] = False
        if not eligible.any():
            break
        best = int(np.argmax(np.where(eligible, weights, -1.0)))
        chosen.append(best)
        nearest = np.minimum(nearest, distances[best])

    while len(chosen) < k:
        before = np.maximum(nearest - MIN_PALETTE_DISTANCE, 0)
        after = np.maximum(np.minimum(nearest[None, :], distances) - MIN_PALETTE_DISTANCE, 0)
        gains = (weights[None, :] * (before[None, :] - after)).sum(axis=1)
        gains[nearest < MIN_PALETTE_DISTANCE] = 0
        gains[chosen] = 0

        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        chosen.append(best)
        nearest = np.minimum(nearest, distances[best])

    return atlas[chosen].copy()


def _pick_palette_legacy(atlas: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """ablation 비교용 — 이전(면적 가중 gain 2라운드) 방식."""
    distances = np.linalg.norm(atlas[:, None] - atlas[None, :], axis=-1)
    nearest = np.full(len(atlas), np.inf)
    chosen: list[int] = []
    for floor in (MIN_CONTENT_DISTANCE, MIN_PALETTE_DISTANCE):
        while len(chosen) < k:
            before = np.maximum(nearest - floor, 0)
            after = np.maximum(np.minimum(nearest[None, :], distances) - floor, 0)
            gains = (weights[None, :] * (before[None, :] - after)).sum(axis=1)
            gains[nearest < MIN_PALETTE_DISTANCE] = 0
            best = int(np.argmax(gains))
            if gains[best] <= 0:
                break
            chosen.append(best)
            nearest = np.minimum(nearest, distances[best])
    return atlas[chosen].copy()


def _refine(atlas: np.ndarray, weights: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """고른 색을 자기가 맡은 후보들의 무게중심으로 옮긴다.

    _pick_palette는 어떤 색에 번호를 줄지는 잘 고르지만 그 색의 정확한 위치는 못 잡는다.
    한 뭉치 안이 이미 다 덮인 상태에서는 뭉치의 중심이든 가장자리든 '보이는 오차'가
    똑같이 0이라, 기준선 바깥의 외딴 후보 쪽으로 끌려간다(실측: 옥토캣의 남색이
    실제 (34,38,44) 대신 훨씬 어두운 (6,13,21)로 잡혀 그림 전체가 검게 눌렸다).
    """
    for _ in range(12):
        labels = _nearest(atlas, centers)
        moved = centers.copy()
        for index in range(len(centers)):
            owned = labels == index
            total = weights[owned].sum()
            if total > 0:
                moved[index] = (atlas[owned] * weights[owned, None]).sum(axis=0) / total
        if np.allclose(moved, centers):
            break
        centers = moved
    return centers


def _merge_indistinct(
    atlas: np.ndarray, weights: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """무게중심으로 옮기다가 서로 붙어버린 색을 합친다.

    _pick_palette가 고를 때는 MIN_PALETTE_DISTANCE 밖이었어도 _refine이 둘 다 같은
    뭉치로 끌어당기면 육안으로 구분되지 않는 번호 두 개가 색상표에 남는다.
    """
    while len(centers) > 1:
        labels = _nearest(atlas, centers)
        owned = np.bincount(labels, weights=weights, minlength=len(centers))

        distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        if distance[i, j] >= MIN_PALETTE_DISTANCE:
            break

        if owned[j] > owned[i]:
            i, j = j, i
        total = owned[i] + owned[j]
        if total > 0:
            centers[i] = (centers[i] * owned[i] + centers[j] * owned[j]) / total
        centers = np.delete(centers, j, axis=0)

    return centers


def _nearest(colors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """각 색에서 가장 가까운 중심점 번호. 전부 LAB 좌표다.

    거리표를 한 번에 쌓으면 (픽셀 수 x 색 수 x 3)이라 1200px·30색에서 수백 MB가 된다.
    중심점 하나씩 훑으며 최솟값만 남긴다.
    """
    best = np.zeros(len(colors), dtype=np.uint8)
    closest = np.full(len(colors), np.inf)
    for index, center in enumerate(centers):
        distance = np.square(colors - center).sum(axis=1)
        nearer = distance < closest
        closest[nearer] = distance[nearer]
        best[nearer] = index
    return best


def drop_unused_colors(
    region_labels: np.ndarray, palette: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """그림에 실제로 남지 않은 색을 팔레트에서 뺀다.

    얇은 조각 영역이 이웃에 흡수되면서 색 하나가 통째로 사라질 수 있다(아이콘의
    안티에일리어싱 회색들이 그렇다). 그대로 두면 색상표에는 번호가 찍혀 있는데
    그림 어디에도 그 번호가 없어서, 칠하는 사람이 빠진 곳을 찾아 헤매게 된다.
    """
    used = np.unique(region_labels)
    if len(used) == len(palette):
        return region_labels, palette

    lookup = np.zeros(len(palette), dtype=region_labels.dtype)
    lookup[used] = np.arange(len(used))
    return lookup[region_labels], palette[used]


def to_lab(rgb: np.ndarray) -> np.ndarray:
    # cvtColor는 float 입력이 0~1일 때만 진짜 CIELAB(L 0~100)을 준다.
    # uint8로 넘기면 L이 0~255로 스케일돼 거리 기준이 달라진다.
    scaled = (np.clip(rgb, 0, 255) / 255.0).astype(np.float32).reshape(1, -1, 3)
    return cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)


def to_rgb(lab: np.ndarray) -> np.ndarray:
    """to_lab의 역변환. 번호별 대표색은 평탄한 자리에서 뽑은 중심점 그대로 쓴다.

    배정된 픽셀의 평균으로 대표색을 다시 계산하면 안 된다. 경계의 그라데이션 픽셀이
    전부 양쪽 클러스터 중 하나로 딸려 들어오기 때문에, 검정 클러스터의 평균이 회색
    쪽으로 끌려간다(실측: 옥토캣의 흰색 중심점 L 95.9가 배정 픽셀 평균으로는 81.0이
    되어, 팔레트에 없어야 할 회색이 도로 생긴다).
    """
    scaled = lab.astype(np.float32).reshape(1, -1, 3)
    return cv2.cvtColor(scaled, cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float64) * 255.0


def palette_distances(palette: np.ndarray) -> np.ndarray:
    """팔레트 색끼리의 dE76 거리표. [i, j]로 색 단차의 크기를 바로 읽는다."""
    lab = to_lab(palette)
    return np.linalg.norm(lab[:, None] - lab[None, :], axis=-1)
