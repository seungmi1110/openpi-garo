"""
GARO Raw_Data를 LeRobot format으로 변환하는 코드
Raw_Data 구조:
episode_0000/
 ├── episode_data.json
 ├── observation_images_top/
 ├── observation_images_wrist_right/
 └── observation_images_wrist_left/
"""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import tyro

# 기존 LIBERO용 코드라 사용 안 함
# from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 기존 LIBERO는 tfds 기반이라 GARO 데이터와 맞지 않음
# import tensorflow_datasets as tfds


# 기존 LIBERO용 repo 이름
# REPO_NAME = "your_hf_username/libero"

# GARO 데이터셋 이름
REPO_NAME = "garo_pi05"


# 기존 LIBERO dataset 목록이라 사용 안 함
# RAW_DATASET_NAMES = [
#     "libero_10_no_noops",
#     "libero_goal_no_noops",
#     "libero_object_no_noops",
#     "libero_spatial_no_noops",
# ]


def load_image(image_path: Path):
    """jpg 이미지 파일을 읽어서 256x256 RGB numpy array로 변환"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))
    return np.array(image)


def main(
    data_dir: str,
    output_dir: str = "/data1/seungseo/datasets/processed/pi05_v01",
    *,
    push_to_hub: bool = False,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # 기존 LIBERO용 저장 위치
    # output_path = HF_LEROBOT_HOME / REPO_NAME

    # GARO용 저장 위치
    output_path = output_dir

    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=output_path,
        robot_type="rx1_dual_arm",
        fps=11,
        features={
            # 기존 LIBERO image
            # "image": {
            #     "dtype": "image",
            #     "shape": (256, 256, 3),
            #     "names": ["height", "width", "channel"],
            # },

            # GARO top camera
            "observation.images.top": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },

            # 기존 LIBERO wrist_image 1개
            # "wrist_image": {
            #     "dtype": "image",
            #     "shape": (256, 256, 3),
            #     "names": ["height", "width", "channel"],
            # },

            # GARO wrist camera 2개
            "observation.images.wrist_right": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist_left": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },

            # 기존 LIBERO state는 8차원
            # "state": {
            #     "dtype": "float32",
            #     "shape": (8,),
            #     "names": ["state"],
            # },

            # GARO state는 16차원
            "state": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["state"],
            },

            # 기존 LIBERO action은 7차원
            # "actions": {
            #     "dtype": "float32",
            #     "shape": (7,),
            #     "names": ["actions"],
            # },

            # GARO action은 16차원
            "actions": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 기존 LIBERO용 반복문
    # for raw_dataset_name in RAW_DATASET_NAMES:
    #     raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
    #     for episode in raw_dataset:
    #         for step in episode["steps"].as_numpy_iterator():
    #             dataset.add_frame(
    #                 {
    #                     "image": step["observation"]["image"],
    #                     "wrist_image": step["observation"]["wrist_image"],
    #                     "state": step["observation"]["state"],
    #                     "actions": step["action"],
    #                     "task": step["language_instruction"].decode(),
    #                 }
    #             )
    #         dataset.save_episode()

    # GARO용 반복문: episode_0000, episode_0001 ... 직접 읽기
    episode_dirs = sorted(data_dir.glob("episode_*"))

    for episode_dir in episode_dirs:
        episode_json = episode_dir / "episode_data.json"

        if not episode_json.exists():
            print(f"skip: {episode_dir} has no episode_data.json")
            continue

        with open(episode_json, "r") as f:
            steps = json.load(f)

        for step in steps:
            top_path = episode_dir / step["observation.images.top"]
            wrist_right_path = episode_dir / step["observation.images.wrist_right"]
            wrist_left_path = episode_dir / step["observation.images.wrist_left"]

            dataset.add_frame(
                {
                    "observation.images.top": load_image(top_path),
                    "observation.images.wrist_right": load_image(wrist_right_path),
                    "observation.images.wrist_left": load_image(wrist_left_path),
                    "state": np.array(step["observation.state"], dtype=np.float32),
                    "actions": np.array(step["action"], dtype=np.float32),
                    "task": step["task"],
                }
            )

        dataset.save_episode()
        print(f"saved {episode_dir.name}, frames={len(steps)}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["garo", "pi05", "rx1_dual_arm"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)