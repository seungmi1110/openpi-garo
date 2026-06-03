"""
Print the 10th-from-last wrist-right frame and command for each episode.

Usage:
    python print_wrist_right_10th_from_end.py <dataset_dir>

Example:
    python print_wrist_right_10th_from_end.py /home/reality/seungmi/datasets/rx1_teleop_star3/2026_05_20/play_block
"""

import argparse
import json
from pathlib import Path

REL_IMAGE_KEY = "observation.images.wrist_right"


def pick_command(meta: dict, frame: dict) -> str:
    return (
        meta.get("language_instruction")
        or meta.get("task")
        or frame.get("task")
        or "<NO_COMMAND>"
    )


def process_episode(ep_dir: Path, offset_from_end: int) -> None:
    data_path = ep_dir / "episode_data.json"
    meta_path = ep_dir / "metadata.json"

    if not data_path.exists():
        print(f"[SKIP] {ep_dir.name}: episode_data.json 없음")
        return

    with open(data_path) as f:
        frames = json.load(f)

    if len(frames) < offset_from_end:
        print(
            f"[SKIP] {ep_dir.name}: 프레임 수 부족 "
            f"(len={len(frames)}, 뒤에서 {offset_from_end}번째 불가)"
        )
        return

    target_frame = frames[-offset_from_end]
    rel_path = target_frame.get(REL_IMAGE_KEY)

    if not rel_path:
        print(f"[SKIP] {ep_dir.name}: 키 없음 ({REL_IMAGE_KEY})")
        return

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    command = pick_command(meta, target_frame)
    abs_frame_path = (ep_dir / rel_path).resolve()
    frame_index = target_frame.get("frame_index", "N/A")

    print(f"{ep_dir.name}")
    print(f"  command: {command}")
    print(f"  frame_index: {frame_index}")
    print(f"  frame_path: {abs_frame_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="각 에피소드에서 wrist_right 프레임 뒤에서 10번째와 명령어 출력"
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="episode_XXXX 폴더들이 있는 데이터셋 경로",
    )
    parser.add_argument(
        "--offset-from-end",
        type=int,
        default=10,
        help="뒤에서 몇 번째 프레임을 가져올지 (기본값: 10)",
    )
    args = parser.parse_args()

    if args.offset_from_end <= 0:
        raise ValueError("--offset-from-end 는 1 이상이어야 합니다.")

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"경로가 존재하지 않습니다: {args.dataset_dir}")

    ep_dirs = sorted(
        p for p in args.dataset_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")
    )
    if not ep_dirs:
        print(f"episode_XXXX 폴더를 찾을 수 없습니다: {args.dataset_dir}")
        return

    for ep_dir in ep_dirs:
        process_episode(ep_dir, args.offset_from_end)


if __name__ == "__main__":
    main()
