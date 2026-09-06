import cv2
import numpy as np

from app.imaging.quantize import MIN_CONTENT_DISTANCE, palette_distances

# 이 반지름의 원을 품지 못하는 영역은 붓이 들어가지 않는 얇은 조각으로 본다.
#
# 번호 글자가 들어가는 크기(한 자리 5.0)와는 다른 질문이라 따로 둔다. 고양이 줄무늬는
# 폭 8px이라 붓은 들어가지만 번호는 못 품는데, 번호 기준으로 지워버리면 줄무늬가
# 통째로 사라진다(실측: 번호 기준으로 자르자 살아남은 줄무늬가 26개에서 1개가 됐다).
MIN_REGION_THICKNESS = 3

# 두께 기준에 걸린 얇은 영역을 그래도 살려볼지 가르는 크기(외접 사각형 대각선, px).
#
# 잎맥이나 감은 눈처럼 작가가 그은 가는 선은 붓이 들어갈 폭은 아니어도 선다운
# 길이로 뻗는다. 반대로 잎사귀 질감에서 튀는 부스러기는 짧게 머문다.
MIN_STROKE_EXTENT = 15.0

# 붓이 안 들어가는 '가는 선'을 그림 전체에서 몇 개까지 남길지.
#
# 감은 눈처럼 작가가 그은 선은 붓 폭이 안 나와도 지우면 그림이 무너진다. 그런데 개별
# 영역마다 "색 단차가 크고 길게 뻗었는가"만 따지면 살아남는 총량에 아무 제동이 없다.
# 잎맥처럼 똑같이 생긴 선이 수백 개인 그림에서는 그게 전부 통과해서, 클로버 그림은
# 영역 984개 중 337개가 번호도 안 들어가는 부스러기인 그물이 됐다.
#
# 상한이지 할당량이 아니다. 자격(MIN_STROKE_EXTENT·색 단차)을 먼저 통과해야 경쟁에
# 들어오고, 자격을 갖춘 선이 상한보다 적으면 그냥 다 산다 — 구름 사진은 자격 통과가
# 0개라 이 값이 무엇이든 결과가 같다(실측: 자격 검사 없이 상한만 두자 구름 사진이
# 70영역에서 109영역으로 늘고 그중 42개가 부스러기가 됐다).
#
# 실측(0/20/40/80/160/320 스윕, 클로버 k=11 — 살아남은 영역 수와 번호가 실제로 찍히는
# 비율): 0은 466영역 86%인데 감은 눈과 입이 통째로 없고, 20은 485영역 83%로 한쪽 눈만
# 돌아온다. 40에서 두 눈과 입이 모두 살아나고 505영역 80%다. 80 이상은 이목구비가 더
# 늘지 않으면서 잎맥만 들어와 545영역 74% · 625영역 64%로 읽기만 나빠진다.
# 이미 깨끗하다고 확인된 옥토캣(5영역 100%)·구름 사진(72영역 99%)은 어느 값에서도
# 그대로다.
DETAIL_BUDGET = 40


def segment_regions(label_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """색 라벨맵을 연결 요소로 쪼개고, 각 영역의 팔레트 색 번호를 함께 돌려준다.

    영역이 어떤 색인지는 region_map과 짝이 되는 이 배열이 유일한 근거다. 병합으로
    영역이 합쳐지면 라벨도 같이 갱신되므로, 미리보기·윤곽선·번호가 항상 같은 그림을
    가리킨다.
    """
    h, w = label_map.shape
    region_map = np.full((h, w), -1, dtype=np.int32)
    region_labels: list[int] = []

    next_id = 0
    for color_label in np.unique(label_map):
        selected = label_map == color_label
        # 경계·두께 계산(_shared_borders, extract_seams의 crack 격자)이 전부 4-이웃
        # 모델이라 영역 생성도 4-연결로 맞춘다. 8-연결은 대각선 한 점으로만 닿은 두
        # 덩어리를 한 영역=한 번호로 묶어, 화면상 떨어져 보이는 칸에 번호가 하나만 생긴다
        count, components = cv2.connectedComponents(selected.astype(np.uint8), connectivity=4)
        region_map[selected] = components[selected] - 1 + next_id
        region_labels.extend([int(color_label)] * (count - 1))
        next_id += count - 1

    return region_map, np.array(region_labels, dtype=np.int32)


def merge_small_regions(
    region_map: np.ndarray,
    region_labels: np.ndarray,
    palette: np.ndarray,
    min_area: int = 400,
    min_thickness: int = MIN_REGION_THICKNESS,
    min_content_distance: float = MIN_CONTENT_DISTANCE,
    detail_budget: int = DETAIL_BUDGET,
) -> tuple[np.ndarray, np.ndarray]:
    """칠할 수 없는 영역을 가장 길게 맞닿은 이웃에 흡수시킨다.

    거르는 기준이 두 개고, 성격이 다르다.

    두께: 품을 수 있는 원의 반지름이 min_thickness에 못 미치면 붓이 들어가지 않는다.
    면적만으로 거르면 안 되는 이유가 여기 있다 — 색 경계를 따라 생기는 몇 px 폭의 띠는
    둘레가 길어서 총 면적이 기준을 가볍게 넘지만, 실제로는 윤곽선과 나란히 달리는
    이중선으로 보인다.

    면적: 두께는 되는데 작은 조각은 "굳이 번호를 매길 값어치가 있나"를 묻는 것뿐이라,
    이웃과의 색 단차가 크면 살린다. 그라데이션을 자르다 생긴 자투리는 흡수될 이웃과
    색이 거의 같아서(실측 dE 11.8 이하) 그대로 걸러진다.

    두께에 걸린 것 중 감은 눈처럼 지우면 안 되는 선은 detail_budget 안에서만 살린다.
    개별 조건으로 거르면 잎맥처럼 똑같이 생긴 선이 수백 개인 그림에서 전부 통과한다.
    """
    count = len(region_labels)
    if count <= 1:
        return region_map, region_labels

    owners, neighbours, shared = _shared_borders(region_map, count)
    # 맞닿은 영역 쌍이 없으면(단색 이미지 등) 흡수시킬 이웃이 없다
    if len(owners) == 0:
        return region_map, region_labels
    starts = np.searchsorted(owners, np.arange(count))
    ends = np.searchsorted(owners, np.arange(count), side="right")

    areas = np.bincount(region_map.ravel(), minlength=count)
    contrast = _neighbour_contrast(
        region_labels, palette, owners, neighbours, shared, starts, ends
    )
    negligible = (areas < min_area) & (contrast < min_content_distance)
    thin = np.bincount(region_map[_interior_mask(region_map, min_thickness)], minlength=count) == 0
    eligible = thin & ~negligible & _is_stroke(
        region_map, count, contrast, min_content_distance
    )
    doomed = negligible | (thin & ~_worth_keeping(
        region_map, region_labels, count, contrast, eligible, detail_budget
    ))
    if not doomed.any():
        return region_map, region_labels

    parent = np.arange(count)

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return int(node)

    # 작은 영역부터 흡수시킨다. 큰 쪽을 먼저 없애면 작은 조각이 갈 곳을 잃는다.
    for region in np.argsort(areas):
        if not doomed[region]:
            continue
        home = root(int(region))
        span = slice(starts[region], ends[region])

        totals: dict[int, int] = {}
        for neighbour, length in zip(neighbours[span], shared[span]):
            target = root(int(neighbour))
            if target != home:
                totals[target] = totals.get(target, 0) + int(length)
        if not totals:
            continue

        # 살아남을 이웃이 있으면 그쪽을 우선한다. 없으면 사라질 이웃끼리라도 뭉쳐서
        # 다음 차례에 함께 흡수되게 둔다.
        surviving = {node: length for node, length in totals.items() if not doomed[node]}
        pool = surviving or totals
        parent[home] = max(pool, key=pool.get)

    final = np.array([root(node) for node in range(count)])
    survivors, compact = np.unique(final, return_inverse=True)
    # region_map의 -1(미배정)은 그대로 -1로 남긴다
    lookup = np.append(compact.astype(np.int32), -1)
    return lookup[region_map], region_labels[survivors]


def _is_stroke(
    region_map: np.ndarray, count: int, contrast: np.ndarray, min_distance: float
) -> np.ndarray:
    """번호는 못 품어도 작가가 그은 선으로 볼 자격이 있는 영역만 True.

    이웃과의 색 단차가 크고, 1px 헤어라인은 아니고, 선이라 할 만큼 뻗어 있어야 한다.
    여기를 통과해도 바로 살아남지는 않고 _worth_keeping의 순위 경쟁으로 넘어간다.
    """
    thick_enough = np.bincount(region_map[_interior_mask(region_map, 1)], minlength=count) > 0
    return (
        (contrast >= min_distance)
        & thick_enough
        & (_extents(region_map, count) >= MIN_STROKE_EXTENT)
    )


def _worth_keeping(
    region_map: np.ndarray,
    region_labels: np.ndarray,
    count: int,
    contrast: np.ndarray,
    eligible: np.ndarray,
    budget: int,
) -> np.ndarray:
    """자격을 갖춘 선 중 예산 안에 드는 것만 True.

    중요도는 이웃과의 색 단차 x 뻗은 길이다. 둘 다 필요하다 — 단차만 보면 잎사귀
    질감의 짧은 부스러기가 끼어들고, 길이만 보면 색이 거의 같아 지워도 티가 안 나는
    양자화 자투리가 끼어든다.

    다만 순위대로 위에서 자르면 안 된다. 되풀이되는 질감이 예산을 통째로 먹기 때문이다
    (실측: 클로버의 자격 통과 선 229개 중 91개가 잎맥 한 색이고, 고양이 눈·입·수염이
    쓰는 검정은 11개뿐이다. 전체 순위로는 잎맥 87개가 눈보다 위라 예산을 160까지
    올려야 눈이 들어오는데, 그때는 번호 있는 영역이 64%로 떨어져 다시 못 읽는다).

    그래서 색깔별로 한 개씩 돌아가며 뽑는다. 수백 번 되풀이되는 질감은 자기 순위표
    안에서만 경쟁하게 되고, 몇 개 없는 이목구비는 자기 색의 1등이라 일찍 들어온다.
    """
    rank = np.where(eligible, contrast * _extents(region_map, count), 0.0)
    keep = np.zeros(count, dtype=bool)
    if budget <= 0:
        return keep

    ranked = np.argsort(-rank)
    ranked = ranked[rank[ranked] > 0]

    by_color: dict[int, list[int]] = {}
    for region in ranked:
        by_color.setdefault(int(region_labels[region]), []).append(int(region))

    taken = 0
    for turn in range(max((len(v) for v in by_color.values()), default=0)):
        for color in sorted(by_color):
            if turn >= len(by_color[color]):
                continue
            keep[by_color[color][turn]] = True
            taken += 1
            if taken >= budget:
                return keep
    return keep


def _extents(region_map: np.ndarray, count: int) -> np.ndarray:
    """영역별 외접 사각형의 대각선 길이. 조각이 그림 위에 얼마나 넓게 뻗었는지를 잰다."""
    height, width = region_map.shape
    flat = region_map.ravel()
    inside = np.nonzero(flat >= 0)[0]
    ids = flat[inside]
    rows, cols = np.divmod(inside, width)

    top = np.full(count, height, dtype=np.int32)
    left = np.full(count, width, dtype=np.int32)
    bottom = np.zeros(count, dtype=np.int32)
    right = np.zeros(count, dtype=np.int32)
    np.minimum.at(top, ids, rows)
    np.minimum.at(left, ids, cols)
    np.maximum.at(bottom, ids, rows)
    np.maximum.at(right, ids, cols)

    return np.hypot(bottom - top + 1, right - left + 1)


def _neighbour_contrast(
    region_labels: np.ndarray,
    palette: np.ndarray,
    owners: np.ndarray,
    neighbours: np.ndarray,
    shared: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """흡수될 이웃과의 색 단차(dE76).

    비교 상대는 '가장 길게 맞닿은 이웃', 즉 실제로 이 영역을 삼킬 쪽이다. 어느
    이웃과든 비교해서 판단하면, 잎맥처럼 잎몸과 어두운 배경에 동시에 닿은 조각이
    엉뚱한 쪽 색으로 판정된다.
    """
    # owners 기준으로 이미 묶여 있으니, 같은 묶음 안에서 경계가 긴 순으로 다시
    # 정렬하면 각 묶음의 마지막이 곧 최장 경계 이웃이다.
    order = np.lexsort((shared, owners))
    dominant = neighbours[order][ends - 1]

    distances = palette_distances(palette)
    contrast = distances[region_labels, region_labels[dominant]]
    # 이웃이 아예 없는 영역(그림 전체가 한 색)은 비교할 대상이 없다.
    return np.where(ends > starts, contrast, 0.0)


def _interior_mask(region_map: np.ndarray, radius: int) -> np.ndarray:
    """반지름 radius인 원이 통째로 들어가는 픽셀만 True.

    영역별로 거리변환을 돌리면 영역 수만큼 전체 이미지를 훑어야 한다. 대신 파티션
    전체에 최소/최대 필터를 한 번씩 걸어서, 창 안이 전부 같은 영역인 픽셀을 찾는다.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    field = region_map.astype(np.float32)
    lowest = cv2.erode(field, kernel, borderType=cv2.BORDER_REPLICATE)
    highest = cv2.dilate(field, kernel, borderType=cv2.BORDER_REPLICATE)
    return lowest == highest


def _shared_borders(
    region_map: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """맞닿은 영역 쌍과 그 경계 길이. 흡수될 영역이 어느 이웃으로 갈지 정하는 근거다."""
    horizontal = np.stack([region_map[:, :-1].ravel(), region_map[:, 1:].ravel()])
    vertical = np.stack([region_map[:-1, :].ravel(), region_map[1:, :].ravel()])
    pairs = np.concatenate([horizontal, vertical], axis=1)
    pairs = pairs[:, (pairs[0] != pairs[1]) & (pairs[0] >= 0) & (pairs[1] >= 0)]

    both = np.concatenate([pairs, pairs[::-1]], axis=1)
    keys, lengths = np.unique(both[0].astype(np.int64) * count + both[1], return_counts=True)
    return keys // count, (keys % count).astype(np.int32), lengths
