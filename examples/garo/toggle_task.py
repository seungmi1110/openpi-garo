"""
Toggle task between "blue moon" and "red heart" for specified episodes.

Usage:
    python toggle_task.py <dataset_dir> <ep1> [ep2 ep3 ...]

Example:
    python toggle_task.py /home/reality/seungmi/datasets/rx1_teleop_star/2026_05_19/play_block 5 7 64
"""

import argparse
import json
from pathlib import Path

TASK_TOGGLE = {
    "Move the yellow star to the red heart.": "Move the yellow star to the blue moon.",
    "Move the yellow star to the blue moon.": "Move the yellow star to the red heart.",
}


def toggle_episode(dataset_dir: Path, ep_num: int):
    ep_name = f"episode_{ep_num:04d}"
    ep_dir = dataset_dir / ep_name

    if not ep_dir.exists():
        print(f"  [SKIP] {ep_name}: 디렉토리 없음")
        return

    data_path = ep_dir / "episode_data.json"
    meta_path = ep_dir / "metadata.json"

    with open(data_path) as f:
        frames = json.load(f)

    old_task = frames[0]["task"]
    new_task = TASK_TOGGLE.get(old_task)

    if new_task is None:
        print(f"  [SKIP] {ep_name}: 알 수 없는 task \"{old_task}\"")
        return

    for frame in frames:
        frame["task"] = new_task
    with open(data_path, "w") as f:
        json.dump(frames, f, indent=2)

    with open(meta_path) as f:
        meta = json.load(f)
    meta["task"] = new_task
    meta["language_instruction"] = new_task
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [OK]   {ep_name}: \"{old_task}\" → \"{new_task}\"")


def main():
    parser = argparse.ArgumentParser(description="Toggle task between blue moon / red heart")
    parser.add_argument("dataset_dir", type=Path, help="데이터셋 폴더 경로")
    parser.add_argument("episodes", type=int, nargs="+", help="에피소드 번호 (공백으로 구분)")
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        print(f"오류: 경로가 존재하지 않습니다 — {args.dataset_dir}")
        return

    print(f"데이터셋: {args.dataset_dir}")
    for ep_num in args.episodes:
        toggle_episode(args.dataset_dir, ep_num)


if __name__ == "__main__":
    main()
