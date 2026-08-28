import cv2
import numpy as np


def extract_contours(region_map: np.ndarray) -> dict[int, np.ndarray]:
    contours_by_region: dict[int, np.ndarray] = {}

    for region_id in np.unique(region_map):
        if region_id == -1:
            continue
        mask = (region_map == region_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest, 2.0, True)
        contours_by_region[int(region_id)] = approx.reshape(-1, 2)

    return contours_by_region


def extract_seams(region_map: np.ndarray, epsilon: float = 0.3) -> list[np.ndarray]:
    """영역 경계를 '픽셀 사이의 틈(crack)' 격자 위에서 한 번만 추출한다.

    영역별 마스크에서 각자 윤곽을 뽑으면 같은 경계가 양쪽에서 1px 어긋난 두 선으로
    그려져 미세하게 겹친다. 경계선을 공유 표현으로 한 번만 만들어 그 문제를 없앤다.
    """
    # 바깥 테두리도 하나의 경계로 잡히도록 이미지 밖을 가상의 영역으로 채운다
    padded = np.pad(region_map, 1, constant_values=-2)

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def link(a: tuple[int, int], b: tuple[int, int]) -> None:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    rows, cols = np.nonzero(padded[:-1, :] != padded[1:, :])
    for row, col in zip(rows + 1, cols):
        link((int(row), int(col)), (int(row), int(col) + 1))

    rows, cols = np.nonzero(padded[:, :-1] != padded[:, 1:])
    for row, col in zip(rows, cols + 1):
        link((int(row), int(col)), (int(row) + 1, int(col)))

    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def walk(start: tuple[int, int], first: tuple[int, int]) -> list[tuple[int, int]] | None:
        edge = (start, first) if start < first else (first, start)
        if edge in visited:
            return None
        visited.add(edge)
        path = [start, first]
        previous, current = start, first
        while current != start and len(adjacency[current]) == 2:
            a, b = adjacency[current]
            nxt = a if b == previous else b
            edge = (current, nxt) if current < nxt else (nxt, current)
            if edge in visited:
                break
            visited.add(edge)
            path.append(nxt)
            previous, current = current, nxt
        return path

    paths: list[tuple[list[tuple[int, int]], bool]] = []
    for node, neighbours in adjacency.items():
        if len(neighbours) == 2:
            continue
        for neighbour in neighbours:
            path = walk(node, neighbour)
            if path is not None:
                # 분기점에서 출발해 제자리로 돌아온 경로는 열린 호가 아니라 고리다.
                # 열린 호로 넘기면 시작점과 끝점이 같아 근사가 무너지고 마지막 변이 통째로
                # 빠진다. 폭이 몇 px뿐인 영역은 그 한 변이 윤곽 전체의 절반이라 구멍이 난다.
                paths.append((path, path[0] == path[-1]))
    for node, neighbours in adjacency.items():
        for neighbour in neighbours:
            path = walk(node, neighbour)
            if path is not None:
                paths.append((path, True))

    h, w = region_map.shape
    seams: list[np.ndarray] = []
    for path, closed in paths:
        if len(path) < 2:
            continue
        pts = np.array([(x - 1, y - 1) for y, x in path], dtype=np.int32)
        np.clip(pts, [0, 0], [w - 1, h - 1], out=pts)
        # crack 격자에서 꺾인 점이 현을 벗어나는 최소 거리가 정확히 0.5px이다. 그래서
        # 0.5 미만이면 공선인 점만 지워져 래스터 결과가 원본과 완전히 같고, 0.5부터는
        # 꺾임이 지워지며 선이 이웃 영역으로 밀린다. 0.5 양쪽으로 여유를 둔 값을 쓴다.
        simplified = cv2.approxPolyDP(pts.reshape(-1, 1, 2), epsilon, closed).reshape(-1, 2)
        if closed:
            simplified = np.vstack([simplified, simplified[:1]])
        seams.append(simplified)

    return seams
