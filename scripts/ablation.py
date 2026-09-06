"""단계별 효과 분리 검증 (프로덕션 코드 아님).

동일 입력·색 수·해상도로 파이프라인을 여러 ablation 설정으로 돌려 별도 디렉터리에 저장한다.
기존 storage/outputs/<uuid> 는 건드리지 않는다.

실행: docker compose run --rm -v "$PWD/scripts:/app/scripts" \
        -v "$HOME/Desktop/test2.jpeg:/data/test2.jpeg:ro" web python scripts/ablation.py
"""

import json
import shutil
from pathlib import Path

from app.imaging.pipeline import _git_rev, generate_design

SRC = Path("/data/test2.jpeg")
K = 11
ROOT = Path("/app/storage/outputs/_ablation") / _git_rev()

CONFIGS = {
    "0_baseline_legacy": frozenset({"no_edge_merge", "no_line_layer", "legacy_palette"}),
    "1_edge_merge_only": frozenset({"no_line_layer", "legacy_palette"}),
    "2_line_boundary_only": frozenset({"no_edge_merge", "legacy_palette"}),
    "3_palette_only": frozenset({"no_edge_merge", "no_line_layer"}),
    "4_combined": frozenset(),
}


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    index = {"git_rev": _git_rev(), "k": K, "runs": {}}

    for name, ablation in CONFIGS.items():
        out = ROOT / name
        try:
            generate_design(SRC, K, out, mode="illustration", ablation=ablation)
            summary = json.loads((out / "debug" / "summary.json").read_text())
            manifest = json.loads((out / "debug" / "manifest.json").read_text())
            index["runs"][name] = {
                "ablation": sorted(ablation),
                "region_counts": summary["region_count"],
                "palette_size": summary["palette_size"],
                "palette_min_dE": summary["palette_min_dE"],
                "numbers_placed": summary["numbers_placed"],
                "numbers_skipped": summary["numbers_skipped"],
                "numbers_skipped_by_reason": summary["numbers_skipped_by_reason"],
                "boundary": summary.get("boundary", {}),
                "edge_merges_applied": summary.get("edge_merges_applied", 0),
                "timings_s": manifest["timings_s"],
                "final_palette": summary["regions"] and None,
            }
            print(f"OK  {name}: {summary['region_count']}  skip {summary['numbers_skipped']}")
        except Exception as exc:  # noqa: BLE001
            index["runs"][name] = {"error": repr(exc)}
            print(f"FAIL {name}: {exc!r}")

    (ROOT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
