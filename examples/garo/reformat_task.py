"""
Convert task strings from "Move the [A] to the [B]."
                       to "Pick up the [A] and move it next to the [B]."

Applies to all episodes under the given dataset directory.
Updates episode_data.json (all frames) and metadata.json.

Usage:
    python reformat_task.py <dataset_dir>

Example:
    python reformat_task.py /home/reality/seungmi/datasets/rx1_teleop_jj/2026_05_18/play_block
"""

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"^Move the (.+) to the (.+)\.$")
META_TASK_FIELDS = ["task", "language_instruction", "task_type"]


def convert(task: str) -> str | None:
    m = PATTERN.match(task)
    if not m:
        return None
    return f"Pick up the {m.group(1)} and move it next to the {m.group(2)}."


def reformat_episode(ep_dir: Path):
    data_path = ep_dir / "episode_data.json"
    meta_path = ep_dir / "metadata.json"

    changed = False

    if data_path.exists():
        with open(data_path) as f:
            frames = json.load(f)

        old_task = frames[0].get("task", "")
        new_task = convert(old_task)

        if new_task:
            for frame in frames:
                frame["task"] = new_task
            with open(data_path, "w") as f:
                json.dump(frames, f, indent=2)
            print(f"  [data ] {ep_dir.name}: \"{old_task}\" → \"{new_task}\"")
            changed = True
        else:
            print(f"  [skip ] {ep_dir.name}: 변환 대상 아님 (\"{old_task}\")")

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        meta_changed = False
        for field in META_TASK_FIELDS:
            if field in meta:
                converted = convert(meta[field])
                if converted:
                    meta[field] = converted
                    meta_changed = True

        if meta_changed:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  [meta ] {ep_dir.name}: metadata 업데이트")
            changed = True

    return changed


def main():
    parser = argparse.ArgumentParser(description="Reformat task strings from 'Move the A to the B.' pattern")
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

    changed_count = sum(reformat_episode(ep_dir) for ep_dir in ep_dirs)
    print(f"\n완료: {changed_count}/{len(ep_dirs)}개 에피소드 수정됨")


if __name__ == "__main__":
    main()
