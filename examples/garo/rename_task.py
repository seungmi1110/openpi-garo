"""
Rename task strings across all episodes in a dataset.

Replacements applied:
  "Move the yellow star to the blue moon."
    → "Pick up the yellow star and move it next to the blue moon."
  "Move the yellow star to the red heart."
    → "Pick up the yellow star and move it next to the red heart."

Updates episode_data.json (all frames) and metadata.json for every episode found.

Usage:
    python rename_task.py <dataset_dir>

Example:
    python rename_task.py /home/reality/seungmi/datasets/rx1_teleop_star/2026_05_19/play_block
"""

import argparse
import json
from pathlib import Path

TASK_MAP = {
    "Move the yellow star to the blue moon.": "Pick up the yellow star and move it next to the blue moon.",
    "Move the yellow star to the red heart.": "Pick up the yellow star and move it next to the red heart.",
}

META_TASK_FIELDS = ["task", "language_instruction", "task_type"]


def rename_episode(ep_dir: Path):
    data_path = ep_dir / "episode_data.json"
    meta_path = ep_dir / "metadata.json"

    changed = False

    # episode_data.json
    if data_path.exists():
        with open(data_path) as f:
            frames = json.load(f)

        old_task = frames[0].get("task", "")
        new_task = TASK_MAP.get(old_task)

        if new_task:
            for frame in frames:
                frame["task"] = new_task
            with open(data_path, "w") as f:
                json.dump(frames, f, indent=2)
            print(f"  [data ] {ep_dir.name}: \"{old_task}\" → \"{new_task}\"")
            changed = True
        else:
            print(f"  [skip ] {ep_dir.name}: task 변경 대상 아님 (\"{old_task}\")")

    # metadata.json
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        meta_changed = False
        for field in META_TASK_FIELDS:
            if field in meta and meta[field] in TASK_MAP:
                meta[field] = TASK_MAP[meta[field]]
                meta_changed = True

        if meta_changed:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  [meta ] {ep_dir.name}: metadata 업데이트")
            changed = True

    return changed


def main():
    parser = argparse.ArgumentParser(description="Rename task strings across all episodes in a dataset")
    parser.add_argument("dataset_dir", type=Path, help="데이터셋 폴더 경로 (episode_XXXX 폴더들이 있는 곳)")
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        print(f"오류: 경로가 존재하지 않습니다 — {args.dataset_dir}")
        return

    ep_dirs = sorted(p for p in args.dataset_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))

    if not ep_dirs:
        print(f"episode_ 폴더를 찾을 수 없습니다: {args.dataset_dir}")
        return

    print(f"데이터셋: {args.dataset_dir}")
    print(f"에피소드 수: {len(ep_dirs)}\n")

    changed_count = sum(rename_episode(ep_dir) for ep_dir in ep_dirs)
    print(f"\n완료: {changed_count}/{len(ep_dirs)}개 에피소드 수정됨")


if __name__ == "__main__":
    main()
