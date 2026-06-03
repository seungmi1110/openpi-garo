"""
language_table Raw_Data를 LeRobot format으로 변환하는 코드.

Raw_Data 구조 예시:
episode_0000/
 ├── episode_data.json
 ├── metadata.json
 ├── observation_images_top/
 ├── observation_images_wrist_right/
 └── observation_images_wrist_left/
"""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_REPO_ID = "language_table_panda"
CAMERA_ORDER = [
    "observation.images.top",
    "observation.images.wrist_right",
    "observation.images.wrist_left",
]
CAMERA_FOLDERS = {
    "observation.images.top": "observation_images_top",
    "observation.images.wrist_right": "observation_images_wrist_right",
    "observation.images.wrist_left": "observation_images_wrist_left",
}


def load_image(image_path: Path, out_size: int = 224) -> np.ndarray:
    """이미지를 RGB로 읽고 out_size x out_size 로 리사이즈한다."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((out_size, out_size), Image.LANCZOS)
    return np.array(image)


def read_steps(episode_json: Path) -> list[dict[str, Any]]:
    with open(episode_json, "r") as f:
        loaded = json.load(f)

    steps = loaded["steps"] if isinstance(loaded, dict) and "steps" in loaded else loaded
    if not isinstance(steps, list):
        raise ValueError(f"Invalid steps format in {episode_json}")
    return steps


def read_metadata(episode_dir: Path) -> dict[str, Any]:
    meta_path = episode_dir / "metadata.json"
    if not meta_path.exists():
        return {}

    with open(meta_path, "r") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def resolve_image_path(episode_dir: Path, camera_key: str, rel_path: str | None) -> Path | None:
    """
    이미지 경로를 보정한다.
    - 케이스1: step에 observation_images_top/frame_xxx.png 형태로 저장
    - 케이스2: step에 frame_xxx.png 만 저장
    """
    if not rel_path:
        return None

    rel = Path(rel_path)

    direct = episode_dir / rel
    if direct.exists():
        return direct

    folder = CAMERA_FOLDERS[camera_key]
    fallback = episode_dir / folder / rel.name
    if fallback.exists():
        return fallback

    return None


def discover_schema(episode_dirs: list[Path], include_wrist_left: bool) -> dict[str, Any]:
    for episode_dir in episode_dirs:
        episode_json = episode_dir / "episode_data.json"
        if not episode_json.exists():
            continue

        steps = read_steps(episode_json)
        if not steps:
            continue

        first_step = None
        for step in steps:
            if isinstance(step, dict):
                first_step = step
                break
        if first_step is None:
            continue

        metadata = read_metadata(episode_dir)

        state_dim = metadata.get("state_dim")
        if state_dim is None:
            state_dim = len(first_step.get("observation.state", []))

        action_dim = metadata.get("action_dim")
        if action_dim is None:
            action_dim = len(first_step.get("action", []))

        robot_type = metadata.get("robot_type", "panda")
        fps = int(metadata.get("fps", 10))

        metadata_camera_keys = set(metadata.get("camera_keys", []))
        camera_keys = []
        for key in CAMERA_ORDER:
            if key in first_step or key in metadata_camera_keys:
                if key == "observation.images.wrist_left" and not include_wrist_left:
                    continue
                camera_keys.append(key)

        if not camera_keys:
            raise ValueError("No camera keys found in sample episode")

        return {
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "robot_type": robot_type,
            "fps": fps,
            "camera_keys": camera_keys,
        }

    raise ValueError("Could not discover schema from episodes. Check dataset path.")


def main(
    data_dir: str,
    output_dir: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    max_episodes: int | None = 400,
    include_wrist_left: bool = True,
    image_size: int = 224,
    target_state_dim: int | None = None,
    target_action_dim: int | None = None,
    allow_dim_truncation: bool = False,
    target_robot_type: str | None = None,
    target_fps: int | None = None,
    push_to_hub: bool = False,
):
    data_dir_path = Path(data_dir)
    output_dir_path = Path(output_dir)

    episode_dirs = sorted(p for p in data_dir_path.glob("episode_*") if p.is_dir())
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]

    if not episode_dirs:
        raise ValueError(f"No episode_* dirs found in {data_dir_path}")

    schema = discover_schema(episode_dirs, include_wrist_left=include_wrist_left)
    camera_keys = schema["camera_keys"]

    output_state_dim = schema["state_dim"] if target_state_dim is None else int(target_state_dim)
    output_action_dim = schema["action_dim"] if target_action_dim is None else int(target_action_dim)
    if output_state_dim < schema["state_dim"] and not allow_dim_truncation:
        raise ValueError(
            f"target_state_dim({output_state_dim}) must be >= discovered state_dim({schema['state_dim']}) "
            "unless allow_dim_truncation=True"
        )
    if output_action_dim < schema["action_dim"] and not allow_dim_truncation:
        raise ValueError(
            f"target_action_dim({output_action_dim}) must be >= discovered action_dim({schema['action_dim']}) "
            "unless allow_dim_truncation=True"
        )

    output_robot_type = schema["robot_type"] if target_robot_type is None else target_robot_type
    output_fps = schema["fps"] if target_fps is None else int(target_fps)

    if output_dir_path.exists():
        shutil.rmtree(output_dir_path)

    features: dict[str, Any] = {}
    for key in camera_keys:
        features[key] = {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        }

    features["state"] = {
        "dtype": "float32",
        "shape": (output_state_dim,),
        "names": ["state"],
    }
    features["actions"] = {
        "dtype": "float32",
        "shape": (output_action_dim,),
        "names": ["actions"],
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir_path,
        robot_type=output_robot_type,
        fps=output_fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    print(
        f"start convert: episodes={len(episode_dirs)} "
        f"state_dim={schema['state_dim']}->{output_state_dim} "
        f"action_dim={schema['action_dim']}->{output_action_dim} "
        f"robot_type={schema['robot_type']}->{output_robot_type} "
        f"fps={schema['fps']}->{output_fps}"
    )

    saved_episodes = 0
    skipped_episodes = 0

    for episode_dir in episode_dirs:
        episode_json = episode_dir / "episode_data.json"
        if not episode_json.exists():
            print(f"skip: {episode_dir.name} has no episode_data.json")
            skipped_episodes += 1
            continue

        steps = read_steps(episode_json)
        valid_frames = 0
        skipped_frames = 0

        for step_idx, step in enumerate(steps):
            if not isinstance(step, dict):
                skipped_frames += 1
                continue

            frame: dict[str, Any] = {}

            missing_camera = False
            for camera_key in camera_keys:
                image_path = resolve_image_path(episode_dir, camera_key, step.get(camera_key))
                if image_path is None:
                    missing_camera = True
                    break
                frame[camera_key] = load_image(image_path, out_size=image_size)

            if missing_camera:
                skipped_frames += 1
                if skipped_frames <= 3:
                    print(f"warn: {episode_dir.name} step={step_idx} missing camera image")
                continue

            state = np.array(step.get("observation.state", []), dtype=np.float32)
            actions = np.array(step.get("action", []), dtype=np.float32)

            if state.ndim != 1 or actions.ndim != 1:
                skipped_frames += 1
                if skipped_frames <= 3:
                    print(
                        f"warn: {episode_dir.name} step={step_idx} non-1d vector "
                        f"state={state.shape} action={actions.shape}"
                    )
                continue

            if state.shape[0] > output_state_dim:
                if allow_dim_truncation:
                    state = state[:output_state_dim]
                else:
                    skipped_frames += 1
                    if skipped_frames <= 3:
                        print(
                            f"warn: {episode_dir.name} step={step_idx} state dim overflow "
                            f"state={state.shape} target={output_state_dim}"
                        )
                    continue

            if actions.shape[0] > output_action_dim:
                if allow_dim_truncation:
                    actions = actions[:output_action_dim]
                else:
                    skipped_frames += 1
                    if skipped_frames <= 3:
                        print(
                            f"warn: {episode_dir.name} step={step_idx} action dim overflow "
                            f"action={actions.shape} target={output_action_dim}"
                        )
                    continue

            if state.shape[0] != output_state_dim:
                padded_state = np.zeros((output_state_dim,), dtype=np.float32)
                padded_state[: state.shape[0]] = state
                state = padded_state

            if actions.shape[0] != output_action_dim:
                padded_actions = np.zeros((output_action_dim,), dtype=np.float32)
                padded_actions[: actions.shape[0]] = actions
                actions = padded_actions

            task = step.get("task") or step.get("language_instruction") or ""

            frame["state"] = state
            frame["actions"] = actions
            frame["task"] = task
            dataset.add_frame(frame)
            valid_frames += 1

        if valid_frames == 0:
            print(f"skip: {episode_dir.name} has no valid frames")
            skipped_episodes += 1
            continue

        dataset.save_episode()
        saved_episodes += 1
        print(f"saved {episode_dir.name}, frames={valid_frames}, skipped={skipped_frames}")

    print(f"done: saved_episodes={saved_episodes}, skipped_episodes={skipped_episodes}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["language-table", "panda"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
