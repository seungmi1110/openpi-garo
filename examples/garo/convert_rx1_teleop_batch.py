"""rx1_teleop_* Raw_Data 여러 세트를 convert_garo_to_lerobot.py로 일괄 변환.

예시:
python openpi/examples/garo/convert_rx1_teleop_batch.py \
  --datasets-root /home/reality/seungmi/datasets \
  --output-root /home/reality/seungmi/datasets/processed
"""

from __future__ import annotations

from pathlib import Path

import tyro

try:
    from .convert_garo_to_lerobot import main as convert_one_dataset
except ImportError:
    from convert_garo_to_lerobot import main as convert_one_dataset


DEFAULT_SOURCES = (
    "rx1_teleop_jj",
    "rx1_teleop_hard",
    "rx1_teleop_star",
    "rx1_teleop_star2",
    "rx1_teleop_star3",
)


def find_episode_roots(source_root: Path) -> list[Path]:
    """source_root 아래에서 episode_*를 직접 포함하는 디렉토리들을 찾는다."""
    parents: set[Path] = set()
    for episode_dir in source_root.rglob("episode_*"):
        if episode_dir.is_dir():
            parents.add(episode_dir.parent)
    return sorted(parents)


def make_output_dir_name(source_name: str, episode_root: Path, source_root: Path) -> str:
    rel = episode_root.relative_to(source_root)
    suffix = "_".join(rel.parts)
    return f"pi05_{source_name}_{suffix}"


def main(
    datasets_root: str = "/home/reality/seungmi/datasets",
    output_root: str = "/home/reality/seungmi/datasets/processed",
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    *,
    dry_run: bool = False,
):
    """
    Args:
        datasets_root: rx1_teleop_* 폴더들이 있는 상위 경로
        output_root: 변환된 LeRobot 데이터셋 저장 상위 경로
        sources: 변환할 source 폴더 이름 목록
        dry_run: True면 실제 변환 없이 매핑만 출력
    """
    datasets_root_path = Path(datasets_root)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    planned: list[tuple[str, Path, Path]] = []

    for source_name in sources:
        source_root = datasets_root_path / source_name
        if not source_root.exists():
            print(f"skip: source not found -> {source_root}")
            continue

        episode_roots = find_episode_roots(source_root)
        if not episode_roots:
            print(f"skip: no episode_* found under {source_root}")
            continue

        for episode_root in episode_roots:
            out_name = make_output_dir_name(source_name, episode_root, source_root)
            out_dir = output_root_path / out_name
            planned.append((source_name, episode_root, out_dir))

    if not planned:
        raise ValueError("No conversion targets found. Check datasets_root/sources.")

    print("conversion targets:")
    for source_name, episode_root, out_dir in planned:
        print(f"- [{source_name}] {episode_root} -> {out_dir}")

    if dry_run:
        print("dry_run=True: no conversion executed.")
        return

    success = 0
    failed = 0

    for source_name, episode_root, out_dir in planned:
        print(f"\n=== convert start [{source_name}] ===")
        print(f"input : {episode_root}")
        print(f"output: {out_dir}")
        try:
            convert_one_dataset(
                data_dir=str(episode_root),
                output_dir=str(out_dir),
                push_to_hub=False,
            )
            success += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"error: failed to convert {episode_root}: {exc}")

    print(f"\ndone: success={success}, failed={failed}")


if __name__ == "__main__":
    tyro.cli(main)
