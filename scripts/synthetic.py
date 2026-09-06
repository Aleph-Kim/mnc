"""합성 입력 검증 (프로덕션 코드 아님).

각 케이스마다 무엇을 보존하고 무엇을 지워야 하는지 기대를 명시하고, 파이프라인
결과가 그 기대를 만족하는지 수치로 확인한다.

실행: docker compose run --rm -v "$PWD/scripts:/app/scripts" web python scripts/synthetic.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

from app.imaging.pipeline import _git_rev, generate_design

OUT = Path("/app/storage/outputs/_synthetic") / _git_rev()


def _save(name: str, rgb: np.ndarray) -> Path:
    p = OUT / f"{name}.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return p


def case_gradient_with_shape() -> tuple[np.ndarray, str]:
    # 부드러운 세로 그라데이션 위에 경계가 뚜렷한 빨간 원
    h, w = 600, 400
    g = np.linspace(40, 210, h).astype(np.uint8)
    rgb = np.stack([np.tile(g[:, None], (1, w))] * 3, axis=-1).copy()
    rgb[..., 2] = 230
    cv2.circle(rgb, (200, 300), 90, (220, 30, 30), -1)
    return rgb, "그라데이션은 1~2색으로 평탄화, 빨간 원은 경계·색 보존"


def case_variable_outline() -> tuple[np.ndarray, str]:
    # 굵기가 변하는 검은 외곽선의 흰 사각형, 회색 배경
    rgb = np.full((500, 500, 3), 170, np.uint8)
    cv2.rectangle(rgb, (120, 120), (380, 380), (255, 255, 255), -1)
    cv2.rectangle(rgb, (120, 120), (380, 380), (10, 10, 10), 3)
    cv2.line(rgb, (120, 120), (380, 120), (10, 10, 10), 9)  # 위쪽만 굵게
    return rgb, "검은 선은 선 레이어로(끊김 없이), 흰 면은 흰색 유지, 선/면 왕복 금지"


def case_thin_branch_and_face() -> tuple[np.ndarray, str]:
    # 가는 초록 가지 + 작은 검은 눈/입
    rgb = np.full((500, 500, 3), 220, np.uint8)
    cv2.line(rgb, (250, 60), (250, 300), (40, 150, 40), 3)
    cv2.line(rgb, (250, 150), (330, 90), (40, 150, 40), 3)
    cv2.circle(rgb, (180, 380), 10, (10, 10, 10), -1)
    cv2.circle(rgb, (250, 380), 10, (10, 10, 10), -1)
    cv2.ellipse(rgb, (215, 430), (40, 15), 0, 0, 180, (10, 10, 10), 3)
    return rgb, "가는 가지 보존, 검은 눈은 면으로 유지, 입선은 선"


def case_hole_and_diagonal() -> tuple[np.ndarray, str]:
    rgb = np.full((500, 500, 3), 235, np.uint8)
    cv2.circle(rgb, (250, 250), 150, (50, 90, 200), -1)
    cv2.circle(rgb, (250, 250), 60, (235, 235, 235), -1)  # 구멍
    rgb[60:110, 60:110] = (200, 60, 60)
    rgb[110:160, 110:160] = (200, 60, 60)  # 대각선으로만 닿는 두 블록
    return rgb, "도넛의 구멍 유지, 대각선으로만 닿는 두 블록은 별도 영역+각자 번호"


def case_flat() -> tuple[np.ndarray, str]:
    return np.full((400, 400, 3), (120, 180, 90), np.uint8), "1색, 영역 1개, 오류 없이 완료"


def case_jpeg_background() -> tuple[np.ndarray, str]:
    # JPEG는 8x8 블록 양자화라 잡음이 공간적으로 상관됨 — blur로 근사
    rng = np.random.default_rng(0)
    noise = rng.integers(-30, 30, (500, 500, 3)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 3)
    base = np.full((500, 500, 3), (90, 130, 190), np.float32) + noise
    rgb = np.clip(base, 0, 255).astype(np.uint8)
    cv2.rectangle(rgb, (180, 180), (320, 320), (230, 210, 60), -1)
    return rgb, "질감 배경은 1~2색으로, 노란 사각형은 보존"


CASES = [
    case_gradient_with_shape, case_variable_outline, case_thin_branch_and_face,
    case_hole_and_diagonal, case_flat, case_jpeg_background,
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for fn in CASES:
        name = fn.__name__.replace("case_", "")
        rgb, expect = fn()
        src = _save(f"{name}_input", rgb)
        run_dir = OUT / name
        try:
            res = generate_design(src, 8, run_dir, mode="illustration")
            s = json.loads((run_dir / "debug" / "summary.json").read_text())
            results[name] = {
                "expectation": expect,
                "final_regions": s["region_count"]["final"],
                "palette_size": s["palette_size"],
                "numbers_placed": s["numbers_placed"],
                "numbers_skipped": s["numbers_skipped"],
                "edge_merges": s.get("edge_merges_applied"),
                "boundary_unresolved": s.get("boundary", {}).get("unresolved_px"),
            }
            print(f"OK  {name}: regions={s['region_count']['final']} pal={s['palette_size']}")
        except Exception as exc:  # noqa: BLE001
            results[name] = {"expectation": expect, "error": repr(exc)}
            print(f"FAIL {name}: {exc!r}")
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
